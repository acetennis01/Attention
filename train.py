import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from model import transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import warnings

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path


def greedy_decode(model, source, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, src_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(
            decoder_input.size(1)).type_as(src_mask).to(device)

        out = model.decode(encoder_output, src_mask,
                           decoder_input, decoder_mask)

        prob = model.project(out[:, -1])

        _, next_word = torch.maz(prob, dim=1)

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(
            source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(
                model_out.detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f'Source: {source_text}')
            print_msg(f'Target: {target_text}')
            print_msg(f'Predicted: {model_out_text}')

            if count == num_examples:
                break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    token_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(token_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(token_path))
    else:
        tokenizer = Tokenizer.from_file(str(token_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        'opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # tokenizers
    token_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    token_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split 90-10

    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size

    # create split
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, token_src, token_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, token_src, token_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = token_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = token_src.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length for src sent.: {max_len_src}')
    print(f'Max length for tgt sent.: {max_len_tgt}')

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, token_src, token_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = transformer(vocab_src_len, vocab_tgt_len,
                        config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, token_src, token_tgt = get_ds(config)
    model = get_model(config, token_src.get_vocab_size(),
                      token_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print("preloading the model")
        state = torch.load(model_filename)

        initial_epoch = state['epoch'] + 1
        optim.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_func = nn.CrossEntropyLoss(ignore_index=token_src.token_to_id(
        '[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        # model.train()
        batch_iterator = tqdm(
            train_dataloader, desc=f'Precessing epoch {epoch:02d}')

        for batch in batch_iterator:
            model.train()

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)

            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # run tensor throught the transformer

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask)

            projection_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_func(
                projection_output.view(-1, token_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backpropagate
            loss.backward()

            optim.step()
            optim.zero_grad(set_to_none=True)

            global_step += 1

        run_validation(model, val_dataloader, token_src, token_tgt,
                       config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer_state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
