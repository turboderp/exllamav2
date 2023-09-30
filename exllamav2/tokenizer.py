from exllamav2.config import ExLlamaV2Config
from sentencepiece import SentencePieceProcessor
import torch

class ExLlamaV2Tokenizer:

    class Trie:

        children: dict = {}
        leaf: list = []

        def __init__(self, children = None, leaf = None):
            self.children = children if children is not None else {}
            self.leaf = leaf if leaf is not None else []


    config: ExLlamaV2Config
    tokenizer: SentencePieceProcessor

    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = ""
    newline_token: str = "\n"
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    newline_token_id: int = 13

    id_to_piece: list = None
    piece_to_id: dict = None
    prefix_to_ids: dict = None
    prefix_id_to_ids: dict = None
    char_trie: Trie = None
    char_trie_ci: Trie = None

    tokenized_str_cache = {}
    max_cached_strings = 100

    def __init__(self, config, lazy_init = False):

        self.config = config

        self.tokenizer = SentencePieceProcessor(model_file = self.config.tokenizer_path)

        self.unk_token_id = self.tokenizer.unk_id()
        self.eos_token_id = self.tokenizer.eos_id()
        self.bos_token_id = self.tokenizer.bos_id()
        self.pad_token_id = 0

        # Create dictionaries on init

        if not lazy_init:

            self.get_id_to_piece_list()
            self.get_piece_to_id_dict()
            self.get_prefix_to_ids_dict()
            self.get_prefix_id_to_ids_dict()
            self.get_char_trie()
            self.get_char_trie_ci()


    # Get single token

    def single_token(self, token_id: int):

        return torch.tensor([[token_id]], dtype = torch.long)


    # Encode string

    # TODO: Handle added tokens for "special" models

    def encode(self, text, add_bos = False, add_eos = False):

        if isinstance(text, list):

            # text is a list of strings

            list_ids = self.tokenizer.EncodeAsIds(text)

            if add_bos:
                for ids in list_ids: ids.insert(0, self.bos_token_id)
            if add_eos:
                for ids in list_ids: ids.append(self.eos_token_id)

            max_length = max([len(ids) for ids in list_ids])

            padded_ids = []
            for ids in list_ids:
                padding = torch.full((max_length - len(ids),), self.pad_token_id)
                sequence = torch.tensor(ids)
                padded_ids.append(torch.cat((padding, sequence), dim = 0))

            return torch.stack(padded_ids, dim = 0)

        else:

            # text is a single string

            ids = self.tokenizer.EncodeAsIds(text)

            if add_bos:
                ids.insert(0, self.bos_token_id)
            if add_eos:
                ids.append(self.eos_token_id)

            return torch.tensor(ids).to(torch.long).unsqueeze(0)


    # Decode IDs

    def decode(self, ids):

        if ids.dim() > 1:

            texts = []
            for i in range(ids.shape[0]):
                seq = ids[i].tolist()
                seq = [t for t in seq if t != self.pad_token_id]
                if self.eos_token_id in seq: seq = seq[:seq.index(self.eos_token_id)]
                texts.append(self.tokenizer.Decode(seq))
            return texts

        else:

            ids = ids.tolist()
            text = self.tokenizer.Decode(ids)
            return text


    # Create padding mask

    def padding_mask(self, ids):

        mask = (ids == self.pad_token_id)
        mask = mask.int()
        mask *= -65504
        mask = mask.half()
        return mask


    def num_tokens(self, text):

        ids = self.tokenizer.Encode(text)
        return len(ids)


    # Copy vocabulary from SP model

    def get_id_to_piece_list(self):

        if self.id_to_piece is not None: return self.id_to_piece

        all_tokens = list(range(self.tokenizer.vocab_size()))
        self.id_to_piece = \
        [
            (p.replace("▁", " ") if not p.startswith("<") else self.tokenizer.decode(idx))
            for idx, p in enumerate(self.tokenizer.id_to_piece(all_tokens))
        ]
        return self.id_to_piece


    def get_piece_to_id_dict(self):

        if self.piece_to_id is not None: return self.piece_to_id

        all_pieces = self.get_id_to_piece_list()
        self.piece_to_id = { piece: idx for idx, piece in enumerate(all_pieces) }
        return self.piece_to_id


    # Create dictionary mapping prefixes to token IDs

    def get_prefix_to_ids_dict(self):

        if self.prefix_to_ids is not None: return self.prefix_to_ids

        piece_to_id = self.get_piece_to_id_dict()
        pieces = sorted(list(piece_to_id.keys()))
        pieces = [p for p in pieces if len(p) > 0]
        self.prefix_to_ids = {}

        for i in range(len(pieces)):
            piece = pieces[i]
            if len(piece) == 0: continue
            piece_id = piece_to_id[pieces[i]]
            self.prefix_to_ids[piece] = [piece_id]

            for j in range(1, len(piece)):
                fpiece = piece[:-j]
                if fpiece in self.prefix_to_ids:
                    self.prefix_to_ids[fpiece].append(piece_id)

        self.prefix_to_ids = { prefix: sorted(ids) for prefix, ids in self.prefix_to_ids.items() }

        return self.prefix_to_ids


    # Create dictionary mapping each ID to any IDs that it prefixes

    def get_prefix_id_to_ids_dict(self):

        if self.prefix_id_to_ids is not None: return self.prefix_id_to_ids

        piece_to_id = self.get_piece_to_id_dict()
        prefix_to_ids = self.get_prefix_to_ids_dict()

        self.prefix_id_to_ids = { piece_to_id[piece]: ids for piece, ids in prefix_to_ids.items() }

        return self.prefix_id_to_ids


    # Create trie mapping chars to token IDs

    def _make_trie(self, ci):

        id_to_piece = self.get_id_to_piece_list()
        trie = ExLlamaV2Tokenizer.Trie()

        for idx, piece in enumerate(id_to_piece):

            if ci: piece = piece.lower()

            w = trie
            while piece != "":

                p = piece[0]
                piece = piece[1:]

                if p not in w.children: w.children[p] = ExLlamaV2Tokenizer.Trie()
                w = w.children[p]

                if piece == "": w.leaf.append(idx)

        return trie


    def get_char_trie(self):

        if self.char_trie is not None: return self.char_trie

        self.char_trie = self._make_trie(False)
        return self.char_trie


    def get_char_trie_ci(self):

        if self.char_trie_ci is not None: return self.char_trie_ci

        self.char_trie_ci = self._make_trie(True)
        return self.char_trie_ci


    # Cached tokenization

    def cached_encode_str(self, text: str):

        if text in self.tokenized_str_cache:
            return self.tokenized_str_cache[text]

        while len(self.tokenized_str_cache) >= self.max_cached_strings:
            del self.tokenized_str_cache[next(iter(self.tokenized_str_cache))]  # Always removes oldest entry as of Python 3.7

        new_enc = self.encode(text)
        self.tokenized_str_cache[text] = new_enc
        return new_enc
