import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor

class ExLlamaV2LayerNorm(ExLlamaV2Module):

    layernorm: nn.LayerNorm or None = None
    weight: nn.Parameter or None = None
    bias: nn.Parameter or None = None
    variance_epsilon: float

    name: str = "LayerNorm"


    def __init__(self, model, key):
        super().__init__(model, key)


    def load(self):

        w = self.load_weight()

        if isinstance(w, tuple):
            weight = w[0]
            bias = w[1]
        else:
            weight = w
            bias = None

        assert isinstance(weight, nn.Parameter)
        assert bias is None or isinstance(bias, nn.Parameter)

        self.layernorm = nn.LayerNorm(self.model.config.hidden_size, elementwise_affine = True, bias = bias is not None)

        self.layernorm.weight = weight
        self.weight = weight

        if bias is not None:
            self.layernorm.bias = bias
            self.bias = bias

        self.variance_epsilon = self.model.config.norm_eps


    def unload(self):

        if self.layernorm is not None:
            self.layernorm = None
        self.weight = None
        self.bias = None


    def get_weight(self):

        if self.bias is not None: return self.weight, self.bias
        return self.weight


    def weight_footprint(self):

        hidden_size = self.model.config.hidden_size
        return hidden_size * 2


    def scratch_space_fixed(self):

        return 0


    def scratch_space(self):

        return 0


    def forward(self, hidden_states, cache = None, attn_params = None, past_len = None, intermediates = False, loras = None, position_offsets = None):

        output_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        norm = torch.empty_like(hidden_states)
        ext_c.layer_norm(hidden_states,
                         self.weight.data,
                         self.bias.data if self.bias is not None else none_tensor,
                         norm,
                         self.variance_epsilon)

        hidden_states = norm.view(output_shape)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


    def forward_torch(self, hidden_states, cache = None, attn_params = None, past_len = None, intermediates = False, loras = None, position_offsets = None):

        # output_shape = hidden_states.shape
        # hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # mean = torch.mean(hidden_states, dim = -1, keepdim = True)
        # var = torch.square(hidden_states - mean).mean(dim = -1, keepdim = True)
        # hidden_states = (hidden_states - mean) / torch.sqrt(var + self.variance_epsilon)
        # hidden_states *= self.weight
        # hidden_states += self.bias

        hidden_states = self.layernorm(hidden_states)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states


