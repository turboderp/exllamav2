
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator.filters import ExLlamaV2Filter
from functools import lru_cache
from lmformatenforcer.integrations.exllamav2 import build_token_enforcer_tokenizer_data
from lmformatenforcer import TokenEnforcer, CharacterLevelParser
from typing import List


# Temporary wrapper for lm-format-enforcer, until the integration in LMFE itself is updated


@lru_cache(10)
def _get_lmfe_tokenizer_data(tokenizer: ExLlamaV2Tokenizer):
    return build_token_enforcer_tokenizer_data(tokenizer)


class ExLlamaV2TokenEnforcerFilter(ExLlamaV2Filter):

    token_sequence: List[int]

    def __init__(
        self,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
        character_level_parser: CharacterLevelParser,
    ):
        super().__init__(model, tokenizer)
        tokenizer_data = _get_lmfe_tokenizer_data(tokenizer)
        self.token_enforcer = TokenEnforcer(tokenizer_data, character_level_parser)
        self.token_sequence = []

    def begin(self, prefix_str: str) -> None:
        self.token_sequence = []

    def feed(self, token) -> None:
        self.token_sequence.append(int(token[0][0]))

    def next(self):
        allowed_tokens = self.token_enforcer.get_allowed_tokens(self.token_sequence)
        return sorted(allowed_tokens), []

    def use_background_worker(self):
        return True