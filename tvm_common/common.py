import argparse
import logging
import os

import tvm
from tvm import relax
from tvm_common.tvm_convert import TVMConvert

def args_sanity_check(args):
    out_dir = os.path.dirname(args.output)
    if not os.path.exists(out_dir):
        raise ValueError(f"Output directory does not exist: {out_dir}")
    del out_dir

    any_cross_specified = (
        (args.cross_sm is not None)
        or (args.cross_host is not None)
        or (args.cross_cc is not None)
    )
    all_cross_specified = (
        (args.cross_sm is not None)
        and (args.cross_host is not None)
        and (args.cross_cc is not None)
    )

    if any_cross_specified:
        assert (
            all_cross_specified
        ), "If any cross-compilation parameter is specified, all three parameters (--cross_sm, --cross_host, --cross_cc) must be specified"
        assert (
            not args.meta_schedule
        ), "meta-schedule is not supported for cross-compilation now, please develop it with RPC by yourself"

def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output file of tvm model, should be a directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 precision for the model",
    )
    parser.add_argument(
        "--dynamic-batch-size",
        action="store_true",
        help="Use dynamic batch size for the model",
    )
    parser.add_argument(
        "--meta-schedule",
        action="store_true",
        help="Use meta-schedule to optimize the model",
    )
    parser.add_argument(
        "--pipeline",
        type = str,
        default = None, 
        help="Use the name of pipeline to optimize the model",
    )
    parser.add_argument(
        "--inject_hyper_params",
        type = str,
        default = None, 
        help="Path to the inject hyper params"
    )
    parser.add_argument(
        "--from_onnx", 
        type=str, 
        default=None, 
        help="Path to the onnx model"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend",
    )
    parser.add_argument(
        "--cross_sm",
        type=str,
        default=None,
        help="Cross compile the tvm model to the specified sm architecture, e.g. sm_75",
    )
    parser.add_argument(
        "--cross_host",
        type=str,
        default=None,
        help="Cross compile the tvm model to the specified host, e.g. llvm -mtriple=aarch64-linux-gnu",
    )
    parser.add_argument(
        "--cross_cc",
        type=str,
        default=None,
        help="Cross compile the tvm model host code with this compiler, e.g. aarch64-none-linux-gnu-gcc",
    )
    return parser

def convert(args, model, example_args):
    args_sanity_check(args)

    tvm_convert = TVMConvert(
        args.backend,
        args.cross_sm, 
        args.cross_host, 
        args.cross_cc,
    )

    tvm_convert.convert_pipline_from_pytorch(
        model, 
        example_args, 
        args.output, 
        args.meta_schedule,
        pipeline_name = args.pipeline,
        inject_hyper_params_path = args.inject_hyper_params,
        to_onnx = True, 
        dynamic_batch_size = args.dynamic_batch_size, 
        fp16 = args.fp16
    )

def inference(args, x):
    tvm_convert = TVMConvert(
        args.backend,
        args.cross_sm, 
        args.cross_host, 
        args.cross_cc,
    )
    model_path = os.path.join(args.output, f"{args.output}.so")
    weight_path = os.path.join(args.output, f"{args.output}.bin")


    logging.info(f"load tvm module")
    rt_mod = tvm.runtime.load_module(model_path)

    logging.info(f"load weight")
    param_dict = tvm_convert.load_params(weight_path)

    logging.info(f"inference")
    x_tvm = tvm.nd.array(x)
    y_tvm = tvm_convert.inference(rt_mod, param_dict, x_tvm)
    return y_tvm
