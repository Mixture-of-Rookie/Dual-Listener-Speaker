#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import wandb
import argparse
from tqdm import tqdm

# torch
import torch

# model
from models.speaker import Speaker
# dataset
from datasets.dataloader import DataLoader
# solver
from solver.loss import SpeakerCriterion
# config
from config.default import get_cfg_defaults
# utils
from utils.utils import (
    set_lr,
    dump_cfg,
    set_seed_logger,
)
# pyutils
from pyutils.cap_eval.eval import evaluate

def train_epoch(model, dataloader, optimizer, crit, epoch, global_steps, split, logger, cfg):# {{{
    # Set mode for training
    model.train()

    logger.info('=====> Start epoch {}'.format(epoch + 1))

    print_steps = cfg['MONITOR']['PRINT_FREQ']
    clip_max_norm = cfg['OPTIMIZATION']['CLIP_MAX_NORM']

    steps, train_loss = 0, 0.
    while True:
        # Load batch of data
        data = dataloader.getSpeakerBatch(split, cfg)
        obj_feat = data['obj_feat']
        cxt_feat = data['cxt_feat']
        loc_feat = data['loc_feat']
        txt_input_labels = data['txt_input_labels']
        txt_target_labels = data['txt_target_labels']

        # Forward
        pred, _ = model(obj_feat, cxt_feat, loc_feat, txt_input_labels)
        loss = crit(pred, txt_target_labels)

        # Backward
        loss.backward()
        steps += 1
        train_loss += float(loss)

        # Update parameters
        if clip_max_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Print log
        if (steps + 1) % print_steps == 0:
            wandb.log(data={'train_loss': float(loss)},
                      step=steps+global_steps)
            logger.info('Epoch [%d], step [%d], train_loss: %.5f' % (
                        epoch + 1, steps + 1, float(loss)))

        if data['wrapped']:
            dataloader.shuffle_refs(split)
            break

    train_loss = train_loss / steps
    logger.info('** ** Epoch [%d] done! Training loss: %.4f.'
            % (epoch + 1, train_loss))
    return steps #}}}

@torch.no_grad()
def validate(model, dataloader, epoch, global_steps, split, save_dir, logger, cfg):# {{{
    # Set mode for evaluation
    model.eval()

    eos_token = dataloader.word2idx['<EOS>']
    max_seq_len = cfg['INPUT']['MAX_SEQ_LEN']

    results = []
    while True:
        # Load batch of data
        data = dataloader.getSpeakerBatch(split, cfg)
        obj_feat = data['obj_feat']
        cxt_feat = data['cxt_feat']
        loc_feat = data['loc_feat']
        ann_id_list = data['ann_id_list']

        word_index = model.sample(obj_feat, cxt_feat, loc_feat, max_seq_len, eos_token)
        for i, ann_id in enumerate(ann_id_list):
            tmp_phrase = []
            for j in range(word_index.size(1)):
                if word_index[i][j] == eos_token:
                    break
                else:
                    tmp_idx = word_index[i][j].item()
                    tmp_phrase.append(dataloader.idx2word[tmp_idx])
            results.append({'ann_id': ann_id,
                            'caption': ' '.join(tmp_phrase)})

        if data['wrapped']:
            break

    result_file = os.path.join(save_dir, 'eval_result_{}.json'.format(epoch + 1))
    groundtruth_file = cfg['DATASET']['EVAL_SPEAKER_JSON']
    with open(result_file, 'w') as f:
        json.dump(results, f)

    metric_dict = evaluate(groundtruth_file, result_file)
    wandb.log(data={'Bleu_1': metric_dict['Bleu_1'], 'Bleu_2': metric_dict['Bleu_2'],
                    'Bleu_3': metric_dict['Bleu_3']},
              step=global_steps)
    logger.info('Eval at Epoch [%d]! Bleu_1: %.2f, Bleu_2: %.2f, Bleu_3: %.2f.'
            % (epoch + 1, metric_dict['Bleu_1'], metric_dict['Bleu_2'], metric_dict['Bleu_3']))# }}}

def main(cfg):
    # 1. Preparation
    logger, save_dir = set_seed_logger(cfg)
    # Set up wandb
    wandb.login(key='19f80ed6515d4e1c36807094906af71dc9e99898')
    wandb.init(project='dual-listener-speaker', name=args.wandb_name)
    wandb.config.save_dir = save_dir
    # Backup config
    cfg_file = os.path.join(save_dir, 'config.yaml')
    dump_cfg(cfg, cfg_file)

    # 2. Create train/dev dataloader
    data_pickle = cfg['DATASET']['DATA_PICKLE']
    vocab_pickle = cfg['DATASET']['VOCAB_PICKLE']
    dataloader = DataLoader(data_pickle, vocab_pickle)
    dataloader.shuffle_refs(split='trainval')

    # 3. Build model
    speaker = Speaker(cfg)
    speaker.cuda()

    # 4. Set up optimizer & criterion
    lr = cfg['OPTIMIZATION']['SPEAKER_LR']
    optimizer = torch.optim.Adam(speaker.parameters(), lr)
    crit = SpeakerCriterion()
    crit.cuda()

    # 5. Training
    epochs = cfg['OPTIMIZATION']['EPOCHS']
    trn_split = cfg['DATASET']['TRAIN']
    dev_split = cfg['DATASET']['DEV']
    global_steps = 0
    for epoch in range(epochs):
        # Train one epoch
        epoch_steps = train_epoch(speaker, dataloader, optimizer, crit, epoch, global_steps, trn_split, logger, cfg)
        global_steps += epoch_steps

        # Perform evaluation
        if (epoch + 1) % cfg['MONITOR']['EVAL_FREQ'] == 0:
            validate(speaker, dataloader, epoch, global_steps, dev_split, save_dir, logger, cfg)

        # LR decay
        if (epoch + 1) % cfg['OPTIMIZATION']['LR_DECAY_FREQ'] == 0:
            lr = lr * 0.1
            set_lr(optimizer, lr)

        # Save checkpoint
        if (epoch + 1) % cfg['OUTPUT']['SAVE_FREQ'] == 0:
            model_path = os.path.join(save_dir, 'pytorch_model_{}.bin'.format(epoch + 1))
            checkpoint = {}
            checkpoint['speaker'] = speaker
            torch.save(checkpoint, model_path)
            logger.info('** * ** Saving trained model to {} ** * **'.format(model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True,
        help='Path to the config file for a specific experiment.')
    parser.add_argument('--wandb_name', type=str, required=True,
        help='Specify name of wandb experiment for visualization.')
    args = parser.parse_args()

    # get default config & merge from cfg_file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # call main
    main(cfg)
