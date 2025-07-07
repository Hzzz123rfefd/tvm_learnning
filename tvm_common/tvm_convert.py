import json
import logging
import os
import shutil
import numpy as np
import torch
import tvm
import onnx
from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.frontend.onnx import from_onnx
from tvm.ir import IRModule
from tvm.runtime import Executable
from tvm.relax import get_pipeline
from typing import List, Union
from typing import Dict, List, Tuple

from tvm_common.utils import torch2onnx, gc_collect, compare_ndarray
from tvm_common.pipeline import _pipeline


class TVMConvert:
    def __init__(self, backend:str, cross_sm:str = None, cross_host:str = None, cross_cc:str = None):
        self.backend = backend
        self.cross_sm = cross_sm
        self.cross_host = cross_host
        self.cross_cc = cross_cc
        self.cross_compiling = self.cross_sm is not None
        self.dev = tvm.device(self.backend)
        self.target = self.get_target()

    def convert_pipline_from_pytorch(
        self, model: torch.nn.Module, 
        example_args:Tuple[torch.Tensor], 
        output: str, 
        meta_schedule: bool = False,
        pipeline_name: str = None,
        inject_hyper_params_path:str = None,
        to_onnx:bool = False,
        dynamic_batch_size: bool = False,
        fp16: bool = False
    ):
        model_info = {}
        output = os.path.abspath(output)
        output_basename = os.path.basename(output)
        os.makedirs(output, exist_ok = True)

        logging.info("torch model inference")
        with torch.no_grad():
            torch_output = model(*example_args)
            torch_output = torch_output.detach().cpu().numpy()
        torch_output_fp16 = None
        if fp16:
            fp16_args = []
            for arg in example_args:
                if isinstance(arg, torch.Tensor):
                    fp16_args.append(arg.to(torch.float16))
                else:
                    fp16_args.append(arg)
            example_args = tuple(fp16_args)

            logging.info("Converting model to FP16")
            model = model.to(torch.float16)

            logging.info("Skip Running FP16 model due to slow inference speed")
            # with torch.no_grad():
            #     torch_output_fp16 = model(*example_args)
            #     torch_output_fp16 = torch_output_fp16.detach().cpu().numpy()

        if to_onnx:
            logging.info("Dumping ONNX model")
            tmp_dir = os.path.join(output, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_onnx = os.path.join(tmp_dir, "tmp.onnx")
            try:
                # NOTE: TVM's from_exported_program conversion has limitations in data type inference for underlying hardcoding.
                # Therefore, when FP16 is enabled, we convert IRModule through ONNX.  
                torch2onnx(
                    model, example_args, tmp_onnx, dynamic_batch_size
                )
                del model
                gc_collect()
                logging.info("Converting ONNX model to IRModule")
                mod = self.get_ir_module_from_onnx_model(tmp_onnx)
            finally:
                shutil.rmtree(tmp_dir)
                gc_collect()
        else:
            logging.info("Converting pytorch model to IRModule")
            mod = self.get_ir_module_from_pytorch_modol(model, example_args)
            del model
            gc_collect()

        mod, params = relax.frontend.detach_params(mod)
        mod = relax.transform.BundleModelParams()(mod)
        mod.show()

        logging.info("save model weight")
        assert len(params) == 1, f"Only one function is supported, but got {len(params)}"
        param_path = os.path.join(output, f"{output_basename}.bin")
        self.save_params(params, param_path)
        model_info["weight"] = os.path.basename(param_path)

        if inject_hyper_params_path != None:
            logging.info("add inject hyper params to IRModule")
            inputs_bits = int(32 if not fp16 else 16)
            mod = self.inject_hyper_params(mod, inputs_bits, inject_hyper_params_path)

        with self.target:
            logging.info("Optimizing model with TVM pipeline")
            mod = self.opt_mod(mod, output, meta_schedule, pipeline_name, dynamic_batch_size)

            logging.info("Compiling model")
            ex = tvm.compile(mod, self.target)

            logging.info(f"Exporting model to {output}")
            export_path = os.path.join(output, f"{output_basename}.so")
            self.save_mod(ex, export_path)

            logging.info(f"Compiled model exported to: {export_path}")
            model_info["model_lib"] = os.path.basename(export_path)

            logging.info("tvm model inference and check")
            if self.cross_compiling:
                logging.info("Cross-compiling done, skipping running model natively")
            else:
                x_tvm = tvm.nd.array(example_args[0].numpy(), self.dev)
                y_tvm = self.inference(ex, params, x_tvm)
                logging.info(f"TVM output array: \n{y_tvm}")
                logging.info("\n\nCompare TVM output with torch FP32 output:")
                compare_ndarray(torch_output, y_tvm)
                if torch_output_fp16 is not None:
                    logging.info("\n\nCompare TVM output with torch FP16 output:")
                    compare_ndarray(torch_output_fp16, y_tvm)
        
        with open(
            os.path.join(output, f"{output_basename}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(model_info, f, indent=4, ensure_ascii=False)

    def get_target(self)->tvm.target.Target :
        if self.cross_compiling:
            target_config = {
                "kind": self.backend,
                "host": self.cross_host,
            }
            target = tvm.target.Target(target_config)
        else:
            target = tvm.target.Target(self.backend)
        return target

    def get_ir_module_from_onnx_model(self, onnx_model_path: str)-> IRModule:
        model = onnx.load(onnx_model_path)
        mod = from_onnx(model, keep_params_in_input=True)
        return mod

    def get_ir_module_from_pytorch_modol(self, model: torch.nn.Module, example_args:Tuple[torch.Tensor])-> IRModule:
        with torch.no_grad():
            exported_program = export(model, example_args)
            mod = from_exported_program(
                exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
            )
        return mod

    def inject_hyper_params(self, mod: IRModule, input_bits:int, inject_hyper_params_path: str):
        assert inject_hyper_params_path is not None, "inject hyper params path must be provided."

        from tvm.script import ir as I
        from tvm.script import relax as R

        if not os.path.exists(inject_hyper_params_path):
            raise FileNotFoundError(f"inject hyper params file is not exist: {inject_hyper_params_path}")
        
        with open(inject_hyper_params_path, "r") as f:
            try:
                hyper_params = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"inject hyper params file format error, unable to resolve to json: {e}")
            
        @I.ir_module
        class HyperParamModule: 
            # default hyper params
            @R.function
            def get_device_type() -> R.Prim("int32"):
                R.func_attr({"relax.force_pure": True})
                return self.dev.device_type

            @R.function
            def get_input_bits() -> R.Prim("int32"):
                R.func_attr({"relax.force_pure": True})
                return input_bits

        def make_func(typ):
            if typ == "int32":
                @R.function
                def func() -> R.Prim("int32"):
                    R.func_attr({"relax.force_pure": True})
                    return int(val)
            elif typ == "float32":
                @R.function
                def func() -> R.Prim("float32"):
                    R.func_attr({"relax.force_pure": True})
                    return float(val)
            elif typ == "bool":
                @R.function
                def func() -> R.Prim("bool"):
                    R.func_attr({"relax.force_pure": True})
                    return bool(val)
            else:
                raise ValueError(f"Unsupported type: {typ} for key: {key}")
            return func

        for key, param in hyper_params.items():
            typ = param["type"]
            val = param["value"]
            func_name = f"get_{key}"
            func = make_func(typ)
            setattr(HyperParamModule, func_name, func)

        mod.update(HyperParamModule)
        mod.show()
        return mod

    def opt_mod(self, mod: IRModule, output:str, meta_schedule: bool = False, pipeline_name: str = None, dynamic_batch_size: bool = False):
        if meta_schedule:
            assert pipeline_name is None, "When meta_schedule is True, pipeline_name must be None."
        # else:
        #     assert pipeline_name is not None, "When meta_schedule is False, pipeline_name must be provided."

        if pipeline_name:
            mod = relax.get_pipeline(pipeline_name)(mod)
        elif meta_schedule:
            assert (
                not dynamic_batch_size
            ), "Dynamic batch size is not supported for meta-schedule"
            mod = relax.transform.LegalizeOps()(mod)
            mod= tvm.tir.transform.DefaultGPUSchedule()(mod) 
            work_dir = os.path.join(output, "tuning_logs")
            mod = get_pipeline(
                "static_shape_tuning",
                total_trials = 4,    # should be 8000, it waste time
                target = self.target,
                work_dir = work_dir,
            )(mod)
        else:
            mod = relax.transform.LegalizeOps()(mod)
            mod= tvm.tir.transform.DefaultGPUSchedule()(mod) 
        return mod

    def save_mod(self, ex: Executable, mod_path: str):
        if self.cross_compiling:
            ex.export_library(mod_path, cc = self.cross_cc)
        else:
            ex.export_library(mod_path)
    
    def save_params(self, params: Dict[str, List[tvm.nd.NDArray]], param_path: str):
        assert len(params) == 1, f"Only one function is supported, but got {len(params)}"
        func_name = list(params.keys())[0]
        param_dict = {}
        num_params = len(params[func_name])
        for i in range(num_params):
            param_dict[f"{i}"] = params[func_name][i]
        logging.info(f"Saving param {func_name} to {param_path}")
        tvm.runtime.save_param_dict_to_file(param_dict, param_path)
    
    def load_params(self, param_path: str)-> Dict[str, List[tvm.nd.NDArray]]:
        with open(param_path, "rb") as f:
            loaded_bytes = f.read()
        param_dict = tvm.runtime.load_param_dict(loaded_bytes)
        params = [tvm.nd.array(param_dict[str(i)]) for i in range(len(param_dict))]
        return {"main": params}

    def inference(self, ex: Executable, params: Dict[str, List[Union[np.ndarray, tvm.nd.NDArray]]], x_tvm: tvm.nd.NDArray):
        x_tvm = tvm.nd.array(x_tvm, self.dev)
        vm = relax.VirtualMachine(ex, self.dev)
        for func_name, param_list in params.items():
            logging.info(f"Converting params of {func_name} to target device: {self.dev}")
            for i in range(len(param_list)):
                param_list[i] = tvm.nd.array(param_list[i], self.dev)

        y_tvm = vm["main"](x_tvm, params["main"])

        if isinstance(y_tvm, tvm.ir.container.Array):
            y_tvm = y_tvm[0].numpy()
        elif isinstance(y_tvm, tvm.nd.NDArray):
            y_tvm = y_tvm.numpy()
        else:
            raise ValueError(
                f"Unexpected tvm model output type: {type(y_tvm)}, please add code to handle this case."
            )
        timing_res = vm.time_evaluator("main", self.dev)(x_tvm, params["main"])
        logging.info(timing_res)
        return y_tvm

