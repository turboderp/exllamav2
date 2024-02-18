from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Tokenizer
)

from exllamav2.generator.filters.base import ExLlamaV2Filter

class ExLlamaV2PrefixFilter(ExLlamaV2Filter):

    offset: int
    prefix_string: str

    def __init__(self, model, tokenizer, prefix_string):
        super().__init__(model, tokenizer)

        self.prefix_string = prefix_string
        self.offset = 0


    def begin(self, prefix_str = ""):

        self.offset = 0


    def feed(self, token):

        id_to_piece = self.tokenizer.get_id_to_piece_list()
        piece = id_to_piece[token]
        self.offset += len(piece)


    def next(self):

        if self.offset >= len(self.prefix_string):
            return None, set()

        char_trie = self.tokenizer.get_char_trie()
        prefix_to_ids = self.tokenizer.get_prefix_to_ids_dict()

        rem_str = self.prefix_string[self.offset:]

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

        return pass_tokens, set()