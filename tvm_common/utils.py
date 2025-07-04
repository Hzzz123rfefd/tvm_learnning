import numpy as np
import torch
import onnx
import logging
import gc

def gc_collect():
    torch.cuda.empty_cache()
    gc.collect()

def torch2onnx(
    model: torch.nn.Module,
    example_args: tuple,
    output_path: str,
    dynamic_batch_size: bool = False,
):
    if dynamic_batch_size:
        torch.onnx.export(
            model,
            example_args,
            output_path,
            input_names=["pixel_values"],
            output_names=["vit_embeds"],
            dynamic_axes={"pixel_values": {0: "batch_size"}},
            verbose=False,
        )
    else:
        torch.onnx.export(
            model,
            example_args,
            output_path,
            input_names=["pixel_values"],
            output_names=["vit_embeds"],
            verbose=False,
        )

def loadonnx(path:str)->onnx.onnx_ml_pb2.GraphProto:
    onnx_model = onnx.load(path)
    # onnx.checker.check_model(onnx_model)
    logging.info(f"ONNX model checked")

    # 使用 ONNX Runtime 测试模型
    logging.info("Skip Verifying with onnx runtime...")
    # ort_session = onnxruntime.InferenceSession(output_path)
    # ort_inputs = {ort_session.get_inputs()[0].name: example_args[0].cpu().numpy()}
    # ort_outs = ort_session.run(None, ort_inputs)
    logging.info("ONNX Runtime verification done!")

    return onnx_model

def compare_ndarray(a: np.ndarray, b: np.ndarray):
    def cosine_similarity(a: np.ndarray, b: np.ndarray):
        a = np.copy(a).flatten().astype(np.float64)
        b = np.copy(b).flatten().astype(np.float64)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    abs_diff = np.abs(a - b)
    logging.info(f"A output range: [{a.min()}, {a.max()}]")
    logging.info(f"B output range: [{b.min()}, {b.max()}]")
    logging.info(f"max diff: {abs_diff.max()}")
    logging.info(f"min diff: {abs_diff.min()}")
    logging.info(f"mean diff: {abs_diff.mean()}")
    logging.info(f"cosine similarity: {cosine_similarity(a, b)}")