from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Tokenizer
)

from exllamav2.generator.filters.base import ExLlamaV2Filter

class ExLlamaV2SelectFilter(ExLlamaV2Filter):

    options: list[str]
    offset: int
    prefix: str
    case_insensitive: bool
    sequence_str_cmp: str

    def __init__(self,
                 model: ExLlamaV2,
                 tokenizer: ExLlamaV2Tokenizer,
                 options: list[str],
                 case_insensitive: bool = False):
        """
        :param options:
            List of possible strings that may be generated.

        :param case_insensitive:
            Ignore case.
        """
        
        super().__init__(model, tokenizer)

        self.options = options if not case_insensitive else [o.lower() for o in options]
        self.case_insensitive = case_insensitive
        self.offset = 0
        self.prefix = ""
        self.sequence_str_cmp = ""


    def clone(self, c = None):
        if c is None:
            c = ExLlamaV2SelectFilter.__new__(ExLlamaV2SelectFilter)
        super().clone(c)
        c.options = self.options
        c.offset = self.offset
        c.prefix = self.prefix
        c.case_insensitive = self.case_insensitive
        c.sequence_str_cmp = self.sequence_str_cmp
        return c


    def begin(self, prefix_str: str = ""):

        self.sequence_str = ""
        self.sequence_str_cmp = ""
        self.prefix = prefix_str if prefix_str is not None else ""
        self.offset = 0


    def feed(self, token: int):

        id_to_piece = self.tokenizer.get_id_to_piece_list()
        piece = id_to_piece[token]
        self.sequence_str += piece
        if self.case_insensitive:
            split = max(len(self.prefix) - self.offset, 0)
            piece_l = piece[:split]
            piece_r = piece[split:].lower()
            self.sequence_str_cmp += piece_l + piece_r
        else:
            self.sequence_str_cmp += piece
        self.offset += len(piece)


    def next(self):

        # prefix_to_ids = self.tokenizer.get_prefix_to_ids_dict()
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        # pass_str = set()
        # end_str = set()

        char_trie = self.tokenizer.get_char_trie_ci() if self.case_insensitive else self.tokenizer.get_char_trie()
        pass_tokens = set()
        end_tokens = set()

        for o in self.options:

            option = (self.prefix + o)
            if option[:self.offset] != self.sequence_str_cmp: continue

            option = option[self.offset:]

            if self.case_insensitive:
                option_cased = option
                option = option.lower()
            else:
                option_cased = None
            if option_cased == option: option_cased = None

            w = char_trie
            while option != "":

                c = option[0]
                option = option[1:]

                if c in w.children: w = w.children[c]
                else: break

                if len(w.leaf) > 0:

                    # Add tokens to pass set

                    if option_cased is None:
                        pass_tokens.update(w.leaf)
                        # pass_str.update([id_to_piece[l] for l in w.leaf])
                        if option == "":
                            end_tokens.update(w.leaf)
                            # end_str.update([id_to_piece[l] for l in w.leaf])

                    # Special case if prefix is cased but continuation is case-insensitive

                    else:
                        for l in list(w.leaf):
                            s = id_to_piece[l]
                            if option_cased.startswith(s):
                                pass_tokens.add(l)
                                # pass_str.add(s)
                                if option == "":
                                    end_tokens.add(l)
                                    # end_str.add(s)

        return pass_tokens, end_tokens
