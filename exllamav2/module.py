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


    def load_weight(self):

        # EXL2

        if self.key + ".q_weight" in self.model.config.tensor_file_map:
            filename = self.model.config.tensor_file_map[self.key + ".q_weight"]
            with safe_open(filename, framework = "pt", device = "cpu") as st:
                qtensors = {}
                qtensors["q_weight"] = st.get_tensor(self.key + ".q_weight").to(self.device())
                qtensors["q_invperm"] = st.get_tensor(self.key + ".q_invperm").to(self.device())
                qtensors["q_scale"] = st.get_tensor(self.key + ".q_scale").to(self.device())
                qtensors["q_scale_max"] = st.get_tensor(self.key + ".q_scale_max").to(self.device())
                qtensors["q_groups"] = st.get_tensor(self.key + ".q_groups").to(self.device())
                qtensors["q_perm"] = torch.argsort(qtensors["q_invperm"]).to(torch.int)
                return qtensors

        # GPTQ

        if self.key + ".qweight" in self.model.config.tensor_file_map:
            filename = self.model.config.tensor_file_map[self.key + ".qweight"]
            with safe_open(filename, framework = "pt", device = "cpu") as st:
                qtensors = {}
                qtensors["qweight"] = st.get_tensor(self.key + ".qweight").to(self.device())
                qtensors["qzeros"] = st.get_tensor(self.key + ".qzeros").to(self.device())
                qtensors["scales"] = st.get_tensor(self.key + ".scales").to(self.device())
                if self.key + ".g_idx" in self.model.config.tensor_file_map:
                    qtensors["g_idx"] = st.get_tensor(self.key + ".g_idx").to(self.device())
                return qtensors

        # Torch

        if self.key + ".weight" in self.model.config.tensor_file_map:
            filename = self.model.config.tensor_file_map[self.key + ".weight"]
            with safe_open(filename, framework = "pt", device = "cpu") as st:
                weight = st.get_tensor(self.key + ".weight")
                weight = weight.half()
                weight = weight.to(self.device())
                return nn.Parameter(weight)


    def set_device_idx(self, idx):

        self.device_idx = idx
