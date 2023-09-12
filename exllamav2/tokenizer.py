from exllamav2.config import ExLlamaV2Config
from sentencepiece import SentencePieceProcessor
import torch

class ExLlamaV2Tokenizer:

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

    def __init__(self, config):

        self.config = config

        self.tokenizer = SentencePieceProcessor(model_file = self.config.tokenizer_path)

        self.unk_token_id = self.tokenizer.unk_id()
        self.eos_token_id = self.tokenizer.eos_id()
        self.bos_token_id = self.tokenizer.bos_id()
        self.pad_token_id = 0


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