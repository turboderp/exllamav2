import torch
import torch.nn as nn
from exllamav2.config import ExLlamaV2Config
from safetensors import safe_open

def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"

class ExLlamaV2Module:

    model = None
    config: ExLlamaV2Config
    key: str
    device_idx: int

    def __init__(self, model, key):

        self.model = model
        self.key = key


    def device(self):

        return _torch_device(self.device_idx)


    def load_multi(self, keys):

        tensors = {}
        submap = {}
        submap_i = {}

        for k in keys:
            ck = self.key + "." + k
            if ck in self.model.config.tensor_file_map:
                submap[k] = self.model.config.tensor_file_map[ck]

        for k, v in submap.items():
            if v not in submap_i:
                submap_i[v] = []
            submap_i[v].append(k)

        for v, ks in submap_i.items():
            with safe_open(v, framework="pt", device="cpu") as st:
                for k in ks:
                    tensors[k] = st.get_tensor(self.key + "." + k).to(self.device())

        return tensors


    def load_weight(self):

        # EXL2

        if self.key + ".q_weight" in self.model.config.tensor_file_map:
            qtensors = self.load_multi(["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm"])
            qtensors["q_perm"] = torch.argsort(qtensors["q_invperm"]).to(torch.int)
            return qtensors

        # GPTQ

        if self.key + ".qweight" in self.model.config.tensor_file_map:
            qtensors = self.load_multi(["qweight", "qzeros", "scales", "g_idx"])
            return qtensors

        # Torch

        if self.key + ".weight" in self.model.config.tensor_file_map:
            tensor = self.load_multi(["weight"])["weight"]
            tensor = tensor.half()
            return nn.Parameter(tensor)


    def set_device_idx(self, idx):

        self.device_idx = idx
