
from exllamav2 import ExLlamaV2
from exllamav2.config import ExLlamaV2Config
from exllamav2.module import ExLlamaV2Module
from exllamav2.mlp import ExLlamaV2MLP

class ExLlamaV2MultimodalProjector(ExLlamaV2):

    config: ExLlamaV2Config
    modules: list[ExLlamaV2Module]

    # noinspection PyMissingConstructor
    def __init__(
        self,
        config: ExLlamaV2Config
    ):
        self.config = config
        cfg = self.config
        self.archparams = cfg.arch.mmp

        self.modules = [
            ExLlamaV2MLP(
                self,
                cfg.arch.mmp_prefix,
                0,
                archparams = cfg.arch.mmp,
                in_features = cfg.vision_hidden_size,
                out_features = cfg.hidden_size,
                interm_features = cfg.hidden_size,
                has_bias=cfg.multimodal_projector_bias,
                has_norm = False,
                has_residual = False,
            )
        ]

    # noinspection PyMethodOverriding
    def forward(self, x):

        for m in self.modules:
            x = m.forward(x)
        return x

    def load_tp(self, **kwargs):
        raise ValueError("load_tp not supported for multimodal projector")
    def load_tp_gen(self, **kwargs):
        raise ValueError("load_tp not supported for multimodal projector")
    def load_autosplit(self, **kwargs):
        raise ValueError("load_autosplit not supported for multimodal projector")
    def load_autosplit_gen(self, **kwargs):
        raise ValueError("load_autosplit not supported for multimodal projector")