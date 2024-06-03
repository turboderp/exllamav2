from __future__ import annotations
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator.filters.base import ExLlamaV2Filter

class ExLlamaV2PrefixFilter(ExLlamaV2Filter):

    prefix_strings: list[str]
    current_prefixes: set[str]
    current_str: str

    def __init__(self,
                 model: ExLlamaV2,
                 tokenizer: ExLlamaV2Tokenizer,
                 prefix_strings: str | list[str]):
        """
        :param prefix_strings:
            Force generation to start with one of the specified strings. Note that if two strings have a shared
            prefix, only the shorter of the two is effective, since matching the shorter prefix is enough to fully
            satisfy the constraint. I.e. ["story", "storytime"] is effectively the same constraint as ["story"]

        """

        super().__init__(model, tokenizer)

        self.prefix_strings = prefix_strings if isinstance(prefix_strings, list) else [prefix_strings]
        self.current_prefixes = set()
        self.current_str = ""


    def clone(self, c = None):
        if c is None:
            c = ExLlamaV2PrefixFilter.__new__(ExLlamaV2PrefixFilter)
        super().clone(c)
        c.prefix_strings = self.prefix_strings
        c.current_prefixes = self.current_prefixes
        c.current_str = self.current_str
        return c


    def begin(self, prefix_str: str = ""):

        self.current_prefixes = set(self.prefix_strings)
        self.current_str = ""


    def feed(self, token: int):

        id_to_piece = self.tokenizer.get_id_to_piece_list()
        piece = id_to_piece[token]
        self.current_str += piece

        end_prefixes = set()
        for prefix in self.current_prefixes:
            if not prefix[:len(self.current_str)] == self.current_str:
                end_prefixes.add(prefix)
        self.current_prefixes -= end_prefixes


    def next(self):

        min_valid_length = 0 if not self.current_prefixes else min(len(s) for s in self.current_prefixes)
        if len(self.current_str) >= min_valid_length:
            return None, set()

        pass_tokens_all = set()
        for prefix in self.current_prefixes:

            char_trie = self.tokenizer.get_char_trie()
            prefix_to_ids = self.tokenizer.get_prefix_to_ids_dict()

            rem_str = prefix[len(self.current_str):]

            # Use prefix dict if string could be completed by one token

            if rem_str in prefix_to_ids:
                pass_tokens = set(prefix_to_ids[rem_str])
            else:
                pass_tokens = set()

            # Find tokens that would advance along the prefix from the current offset

            for c in rem_str:
                if c in char_trie.children:
                    char_trie = char_trie.children[c]
                else:
                    break
                pass_tokens |= set(char_trie.leaf)

            pass_tokens_all |= pass_tokens

        return pass_tokens_all, set()