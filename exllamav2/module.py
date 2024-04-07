from __future__ import annotations
import torch
import torch.nn as nn
from exllamav2.config import ExLlamaV2Config
from exllamav2.fasttensors import STFile

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2
    from exllamav2.lora import ExLlamaV2Lora

def _torch_device(idx: int) -> str:
    if idx == -1: return "cpu"
    return f"cuda:{idx}"


class ExLlamaV2Module:

    config: ExLlamaV2Config
    key: str
    alt_key: str | None
    device_idx: int
    footprint: int
    submodules: list[ExLlamaV2Module]
    assumed_footprint: int

    def __init__(self,
                 model: ExLlamaV2,
                 key: str):

        self.model = model
        self.key = key
        self.alt_key = None
        self.footprint = -1
        self.submodules = []


    def numel(self): raise(NotImplementedError())
    def load(self): raise(NotImplementedError())
    def unload(self): raise(NotImplementedError())
    def scratch_space_fixed(self): raise(NotImplementedError())
    def scratch_space(self): raise(NotImplementedError())

    def forward(self,
                hidden_states,
                cache = None,
                attn_params = None,
                past_len = None,
                intermediates = None,
                loras = None):
        raise(NotImplementedError())


    def device(self) -> str:
        return _torch_device(self.device_idx)


    def load_multi(self,
                   key: str,
                   keys: list[str],
                   measure: bool = False) -> int | dict[str: torch.Tensor]:

        tensors = {}
        submap = {}
        submap_i = {}
        size = 0

        # key = self.key if override_key is None else override_key

        for k in keys:
            ck = key + "." + k
            if ck in self.model.config.tensor_file_map:
                submap[k] = self.model.config.tensor_file_map[ck]

        for k, v in submap.items():
            if v not in submap_i:
                submap_i[v] = []
            submap_i[v].append(k)

        for v, ks in submap_i.items():
            stfile = STFile.open(v, fast = self.model.config.fasttensors, keymap = self.model.config.arch.keymap)
            for k in ks:
                if measure:
                    size += stfile.measure(key + "." + k)
                else:
                    tensors[k] = stfile.get_tensor(key + "." + k, device = self.device())

        return size if measure else tensors


    def load_weight(self,
                    override_key: str | None = None):

        if override_key is not None:
            keys = [override_key]
        else:
            keys = [self.key]
            if self.alt_key is not None:
                keys += [self.alt_key]

        for key in keys:

            # EXL2

            if key + ".q_weight" in self.model.config.tensor_file_map:
                qtensors = self.load_multi(key, ["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm", "bias"])
                qtensors["q_perm"] = torch.argsort(qtensors["q_invperm"]).to(torch.int)
                return qtensors

            # GPTQ

            if key + ".qweight" in self.model.config.tensor_file_map:
                qtensors = self.load_multi(key, ["qweight", "qzeros", "scales", "g_idx", "bias"])
                if "bias" in qtensors and torch.all(qtensors["bias"].eq(0)):
                    del qtensors["bias"]
                qtensors["scales"] = qtensors["scales"].half()
                return qtensors

            # Torch

            if key + ".weight" in self.model.config.tensor_file_map:
                if key + ".bias" in self.model.config.tensor_file_map:
                    tensors = self.load_multi(key, ["weight", "bias"])
                    tensor = tensors["weight"].half()
                    bias = tensors["bias"].half()
                    return nn.Parameter(tensor), nn.Parameter(bias)
                else:
                    tensors = self.load_multi(key, ["weight"])
                    tensor = tensors["weight"].half()
                    return nn.Parameter(tensor)

            # No weights found for key

        return None


    def load_weight_fused(self,
                          f_key: str,
                          f_beg: int,
                          f_end: int,
                          in_feat: int,
                          out_feat: int):

        for key in [f_key, f_key + ".weight"]:

            filename = self.model.config.tensor_file_map.get(key)
            if not filename: continue

            stfile = STFile.open(filename, fast = self.model.config.fasttensors, keymap = self.model.config.arch.keymap)
            # tensor = stfile.get_tensor(key, device = self.device()).half()
            tensor = stfile.get_tensor(key, device = "cpu", cached = True, out_dtype = torch.half)
            tensor = tensor[f_beg:f_end, :]
            if in_feat != out_feat and \
                tensor.shape[1] == out_feat and \
                tensor.shape[0] == in_feat:
                tensor = tensor.T
            tensor = tensor.contiguous().to(self.device())
            return nn.Parameter(tensor)

        return None


    def weight_footprint(self) -> int:

        if self.footprint == -1:

            keys = [self.key]
            if self.alt_key is not None:
                keys += [self.alt_key]

            for key in keys:

                # EXL2

                if key + ".q_weight" in self.model.config.tensor_file_map:
                    self.footprint = self.load_multi(key, ["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm", "q_perm", "bias"], measure = True)

                # GPTQ

                elif key + ".qweight" in self.model.config.tensor_file_map:
                    self.footprint = self.load_multi(key, ["qweight", "qzeros", "scales", "g_idx", "bias"], measure = True)

                # Torch

                elif key + ".weight" in self.model.config.tensor_file_map:
                    self.footprint = self.load_multi(key, ["weight", "bias"], measure = True)

                if self.footprint != -1: break

            # Error

            if self.footprint == -1:
                # raise ValueError("Unknown tensor type: " + self.key)
                return self.assumed_footprint

        return self.footprint


    def set_device_idx(self, idx: int):
        self.device_idx = idx


    def is_quant(self) -> bool:
        return False


    def reload(self):
        self.unload()
        self.load()
