import logging
import types
import torch
from transformers import AutoModel,AutoConfig
import math
import torch
import logging

from tvm_common import gc_collect, get_common_parser, convert

class ModelInternViTWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("image_mean", torch.Tensor([0.485, 0.456, 0.406]))
        self.register_buffer("image_std", torch.Tensor([0.229, 0.224, 0.225]))

    def forward(self, x: torch.Tensor):
        x = (x - self.image_mean) / self.image_std
        x = x.permute(0, 3, 1, 2)
        internvit = self.model.vision_model
        embd = internvit(
            pixel_values=x, output_hidden_states=False, return_dict=True
        ).last_hidden_state

        embd = embd[:, 1:, :]

        h = w = int(embd.shape[1] ** 0.5)
        embd = embd.reshape(embd.shape[0], h, w, -1)
        embd = self.model.pixel_shuffle(embd, scale_factor=self.model.downsample_ratio)
        embd = embd.reshape(embd.shape[0], -1, embd.shape[-1])
        embd = self.model.mlp1(embd)
        embd = embd.float()

        return embd

    def get_force_image_size(self) -> int:
        return self.model.config.force_image_size

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map

def _get_pos_embed(self, pos_embed, H, W):
    target_dtype = pos_embed.dtype
    pos_embed = (
        pos_embed.float()
        .reshape(
            1,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
            -1,
        )
        .permute(0, 3, 1, 2)
    )
    pos_embed_size = self.image_size // self.patch_size
    if H != pos_embed_size or W != pos_embed_size:
        pos_embed = tnn.functional.interpolate(
            pos_embed, size=(H, W), mode="bicubic", align_corners=False
        )

    pos_embed = pos_embed.reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
    return pos_embed

    def get_force_image_size(self) -> int:
        return self.model.config.force_image_size

def load_torch_model(weight_path: str) -> torch.nn.Module:
    # NOTE: BF16 7B Model, need 14GB memory at least
    device_map = split_model(weight_path)
    model = AutoModel.from_pretrained(
        weight_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()

    del model.language_model
    gc_collect()

    model.vision_model.embeddings._get_pos_embed = types.MethodType(
        _get_pos_embed, model.vision_model.embeddings
    )

    model = ModelInternViTWrapper(model)
    model.eval()
    model.to(torch.float32)
    return model

def main(args):
    model = load_torch_model(args.weight)

    force_image_size = model.get_force_image_size()
    example_args = (
        torch.randn(1, force_image_size, force_image_size, 3, dtype=torch.float32),
    )

    logging.info("Convert pytorch model to tvm")
    convert(args, model, example_args)

if __name__ == "__main__":
    parser = get_common_parser()
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        required=True,
        help="Path to InternVL3 model weights, in huggingface format",
    )
    args = parser.parse_args()
    main(args)