import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, ds, token_src, token_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.token_src = token_src
        self.token_tgt = token_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # start of sentence
        self.sos_token = torch.tensor(
            [token_src.token_to_id('[SOS]')], dtype=torch.int64)
        # end of sentence
        self.eos_token = torch.tensor(
            [token_src.token_to_id('[EOS]')], dtype=torch.int64)
        # padding
        self.pad_token = torch.tensor(
            [token_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: any) -> any:
        src_tgt_pair = self.ds[index]

        src_txt = src_tgt_pair['translation'][self.src_lang]
        tgt_txt = src_tgt_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.token_src.encode(src_txt).ids
        dec_input_tokens = self.token_src.encode(tgt_txt).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence too long')

        # Add the STOS and EOS to the src_txt

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),  # Ensure dtype is Long (int64)
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )


        #encoder_input = torch.cat(
        #    [
        #        self.sos_token,
        #        torch.tensor(enc_input_tokens, dtype=torch.int64),
        #        self.eos_token,
        #        torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        #    ]
        #)

        # Add SOS to decoder
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64),
            ]
        )

        # Add EOS to label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
