#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import wandb
import argparse
from tqdm import tqdm

# torch
import torch

# model
from models.listener import Listener
# dataset
from datasets.dataloader import DataLoader
# solver
from solver.loss import ListenerCriterion
# config
from config.default import get_cfg_defaults
# utils
from utils.utils import (
    set_lr,
    dump_cfg,
    set_seed_logger,
)

def train_epoch(model, dataloader, optimizer, crit, epoch, global_steps, split, logger, cfg):# {{{
    # Set mode for training
    model.train()

    logger.info('=====> Start epoch {}'.format(epoch + 1))

    print_steps = cfg['MONITOR']['PRINT_FREQ']
    clip_max_norm = cfg['OPTIMIZATION']['CLIP_MAX_NORM']

    steps, train_loss = 0, 0.
    num_correct, num_total = 0., 0.
    while True: 
        # Load batch of data
        data = dataloader.getListenerBatch(split, cfg)
        obj_feat = data['obj_feat']
        cxt_feat = data['cxt_feat']
        loc_feat = data['loc_feat']
        pos_labels = data['pos_labels']
        txt_labels = data['txt_labels']
        gt_index_list = data['gt_index_list']

        # Forward
        pred = model(obj_feat, cxt_feat, loc_feat, txt_labels)
        loss = crit(pred, pos_labels)

        # Compute accuracy
        pred = torch.argmax(pred, dim=1)
        for i in range(pred.size(0)):
            if pred[i] in gt_index_list[i]:
                num_correct += 1
            num_total += 1

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
            logger.info('Epoch [%d], step [%d], train_loss: %.5f, train_accuracy: %.4f' % (
                        epoch + 1, steps + 1, float(loss), num_correct/num_total))

        if data['wrapped']:
            dataloader.shuffle_refs(split)
            break

    train_loss = train_loss / steps
    wandb.log(data={'train_accuracy': num_correct/num_total},
              step=steps+global_steps)
    logger.info('** ** Epoch [%d] done! Training loss: %.4f, Training accuracy: %.4f'
            % (epoch + 1, train_loss, num_correct/num_total))
    return steps# }}}


@torch.no_grad()
def validate(model, dataloader, epoch, global_steps, split, logger, cfg):# {{{
    # Set mode for evaluation
    model.eval()

    num_correct, num_total = 0., 0.
    while True:
        # Load batch of data
        data = dataloader.getListenerBatch(split, cfg)
        obj_feat = data['obj_feat']
        cxt_feat = data['cxt_feat']
        loc_feat = data['loc_feat']
        txt_labels = data['txt_labels']
        gt_index_list = data['gt_index_list']

        # Forward
        pred = model(obj_feat, cxt_feat, loc_feat, txt_labels)
        pred = torch.argmax(pred, dim=1)

        # Compute accuracy
        for i in range(pred.size(0)):
            if pred[i] in gt_index_list[i]:
                num_correct += 1
            num_total += 1

        if data['wrapped']:
            break

    wandb.log(data={'val_accuracy': num_correct/num_total},
              step=global_steps)
    logger.info('Eval at Epoch [%d]! Eval samples: %d, Eval accuracy: %.2f' %
            (epoch + 1, num_total, num_correct / num_total))# }}}


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
    listener = Listener(cfg)
    listener.cuda()

    # 4. Set up optimizer & criterion
    lr = cfg['OPTIMIZATION']['LISTENER_LR']
    optimizer = torch.optim.Adam(listener.parameters(), lr)
    crit = ListenerCriterion()

    # 5. Training
    epochs = cfg['OPTIMIZATION']['EPOCHS']
    trn_split = cfg['DATASET']['TRAIN']
    dev_split = cfg['DATASET']['DEV']
    global_steps = 0
    for epoch in range(epochs):
        # Train one epoch
        epoch_steps = train_epoch(listener, dataloader, optimizer, crit, epoch, global_steps, trn_split, logger, cfg)
        global_steps += epoch_steps

        # Perform evaluation
        if (epoch + 1) % cfg['MONITOR']['EVAL_FREQ'] == 0:
            validate(listener, dataloader, epoch, global_steps, dev_split, logger, cfg)

        # LR decay
        if (epoch + 1) % cfg['OPTIMIZATION']['LR_DECAY_FREQ'] == 0:
            lr = lr * 0.1
            set_lr(optimizer, lr)

        # Save checkpoint
        if (epoch + 1) % cfg['OUTPUT']['SAVE_FREQ'] == 0:
            model_path = os.path.join(save_dir, 'pytorch_model_{}.bin'.format(epoch + 1))
            checkpoint = {}
            checkpoint['listener'] = listener
            torch.save(checkpoint, model_path)
            logger.info('** * ** Saving trained model to {} ** * **'.format(model_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True,
        help='Path to the config file for a specific experiment.')
    parser.add_argument('--wandb_name', type=str, required=True,
        help='Specify name of wandb experiment for visualization.')
    args = parser.parse_args()

    # Get default config & merge from cfg_file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # call main
    main(cfg)
