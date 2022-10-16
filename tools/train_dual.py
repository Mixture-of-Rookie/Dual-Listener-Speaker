#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import nltk
import wandb
import random
import argparse
from tqdm import tqdm

# torch
import torch
from torch.autograd import Variable

# dataset
from datasets.dataloader import DataLoader
# solver
from solver.loss import (
    DualSpeakerCriterion,
    DualListenerCriterion,
)
# config
from config.default import get_cfg_defaults
from datasets.dataloader import DataLoader
# utils
from utils.utils import (
    set_lr,
    dump_cfg,
    set_seed_logger,
)

def train_epoch(speaker, listener, dataloader, speaker_optimizer, listener_optimizer,# {{{
        speaker_crit, listener_crit, epoch, global_steps, split, logger, cfg):
    logger.info('=====> Start epoch {}'.format(epoch + 1))

    print_steps = cfg['MONITOR']['PRINT_FREQ']
    max_seq_len = cfg['INPUT']['MAX_SEQ_LEN']
    clip_max_norm = cfg['OPTIMIZATION']['CLIP_MAX_NORM']

    steps, train_loss = 0, 0.
    while True:
        # Load batch of data
        data = dataloader.getDualBatch(split, cfg)
        obj_feat = data['obj_feat']
        cxt_feat = data['cxt_feat']
        loc_feat = data['loc_feat']
        txt_labels = data['txt_labels']
        txt_target_labels = data['txt_target_labels']
        
        #########################################################
        ############   Phrase --> Region --> Phrase  ############
        #########################################################
        if cfg['OPTIMIZATION']['OPTIMIZE_LISTENER']:
            # Set mode
            speaker.eval()
            listener.train()

            ###############################
            ###### Phrase --> Region ######
            ###############################
            listener_pred = listener(obj_feat, cxt_feat, loc_feat, txt_labels)
            topK = cfg['OPTIMIZATION']['TOPK']
            reward_threshold = cfg['OPTIMIZATION']['REWARD_THRESHOLD']
            topk_listener_pred = torch.argsort(listener_pred, dim=1, descending=True)[:, :topK]
            # Fetch speaker visual inputs according the topk_listener_pred
            batch_size = obj_feat.size(0)
            feature_dim = obj_feat.size(-1)
            loc_feature_dim = loc_feat.size(-1)
            speaker_obj_feat = torch.zeros(batch_size * topK, feature_dim)
            speaker_cxt_feat = torch.zeros(batch_size * topK, feature_dim)
            speaker_loc_feat = torch.zeros(batch_size * topK, loc_feature_dim)

            for b in range(batch_size):
                for k in range(topK):
                    tmp_pred_index = topk_listener_pred[b][k]
                    tmp_cur_index = b * topK + k
                    speaker_obj_feat[tmp_cur_index, :] = obj_feat[b][tmp_pred_index]
                    speaker_cxt_feat[tmp_cur_index, :] = cxt_feat[b][tmp_pred_index]
                    speaker_loc_feat[tmp_cur_index, :] = loc_feat[b][tmp_pred_index]

            # Convert to Variable
            speaker_obj_feat = Variable(speaker_obj_feat.cuda())
            speaker_cxt_feat = Variable(speaker_cxt_feat.cuda())
            speaker_loc_feat = Variable(speaker_loc_feat.cuda())
            
            ###############################
            ###### Region --> Phrase ######
            ###############################
            with torch.no_grad():
                eos_token = dataloader.word2idx['<EOS>']
                word_index = speaker.sample(speaker_obj_feat, speaker_cxt_feat, speaker_loc_feat,
                                            max_seq_len, eos_token)
            
            ###############################
            ####### Compute Reward ########
            ###############################
            max_reward_list, max_reward_index_list = [], []
            gt_phrase_list = dataloader.decode_phrase(txt_target_labels)
            pred_phrase_list = dataloader.decode_phrase(word_index)
            for b in range(batch_size):
                cur_max_index, cur_max_reward = 0, 0
                for k in range(topK):
                    tmp_cur_index = b * topK + k
                    reference = gt_phrase_list[b].split(' ')
                    hypothesis = pred_phrase_list[tmp_cur_index].split(' ')
                    bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=[1])
                    if bleu_score > cur_max_reward:
                        cur_max_index = k
                        cur_max_reward = bleu_score

                max_reward_list.append(cur_max_reward)
                if cur_max_reward < reward_threshold:
                    max_reward_index_list.append(-1)  # -1 means no reward
                else:
                    max_reward_index_list.append(topk_listener_pred[b][cur_max_index])

            ############################################
            ####### Compute Loss & Optimization ########
            ############################################
            # Compute loss
            listener_loss = listener_crit(listener_pred, max_reward_list, max_reward_index_list)
            # Backward
            listener_loss.backward()

            # Update parameters
            if clip_max_norm > 0.:
                torch.nn.utils.clip_grad_norm_(listener.parameters(), clip_max_norm)
            listener_optimizer.step()
            listener_optimizer.zero_grad()
            
            ############################################
            ################ Print log #################
            ############################################
            if (steps + 1) % print_steps == 0:
                wandb.log(data={'listener_loss': float(listener_loss)},
                          step=steps+global_steps)
                logger.info('Epoch [%d], step [%d], listener_loss: %.5f' % (
                            epoch + 1, steps + 1, float(listener_loss)))

        #########################################################
        ############   Region --> Phrase --> Region  ############
        #########################################################
        if cfg['OPTIMIZATION']['OPTIMIZE_SPEAKER']:# {{{
            # Set mode
            speaker.train()
            listener.eval()

            ###############################
            ###### Region --> Phrase ######
            ###############################
            
            # Randomly sample one region for each image to generate phrase
            batch_size = obj_feat.size(0)
            feature_dim = obj_feat.size(-1)
            num_proposals = obj_feat.size(1)
            loc_feature_dim = loc_feat.size(-1)
            speaker_obj_feat = torch.zeros(batch_size, feature_dim)
            speaker_cxt_feat = torch.zeros(batch_size, feature_dim)
            speaker_loc_feat = torch.zeros(batch_size, loc_feature_dim)

            # Used for compute speaker reward
            random_region_index_list = []
            for b in range(batch_size):
                random_region_index = random.randint(0, num_proposals - 1)
                speaker_obj_feat[b, :] = obj_feat[b, random_region_index, :]
                speaker_cxt_feat[b, :] = cxt_feat[b, random_region_index, :]
                speaker_loc_feat[b, :] = loc_feat[b, random_region_index, :]
                random_region_index_list.append(random_region_index)

            # Convert to Variable
            speaker_obj_feat = Variable(speaker_obj_feat.cuda())
            speaker_cxt_feat = Variable(speaker_cxt_feat.cuda())
            speaker_loc_feat = Variable(speaker_loc_feat.cuda())

            word_index, word_mask, log_probs = speaker.sample_for_listener(
                speaker_obj_feat, speaker_cxt_feat, speaker_loc_feat, max_seq_len, dataloader.word2idx)

            ###############################
            ###### Phrase --> Region ######
            ###############################
            with torch.no_grad():
                listener_pred = listener(obj_feat, cxt_feat, loc_feat, word_index)
            listener_pred_index = torch.argmax(listener_pred, dim=1)
            
            ###############################
            ####### Compute Reward ########
            ###############################
            reward_list = []
            for b in range(batch_size):
                if random_region_index_list[b] == listener_pred_index[b].item():
                    reward_list.append(1)
                else:
                    reward_list.append(0)

            ############################################
            ####### Compute Loss & Optimization ########
            ############################################
            # Compute loss
            speaker_loss = speaker_crit(reward_list, log_probs, word_mask)
            # Backward
            speaker_loss.backward()

            # Update parameters
            if clip_max_norm > 0.:
                torch.nn.utils.clip_grad_norm_(speaker.parameters(), clip_max_norm)
            speaker_optimizer.step()
            speaker_optimizer.zero_grad()# }}}

        if data['wrapped']:
            dataloader.shuffle_refs(split)
            break

        steps += 1

    return steps# }}}


@torch.no_grad()
def validate(listener, dataloader, epoch, global_steps, split, logger, cfg):# {{{
    # Set mode fro evaluation
    listener.eval()

    num_correct, num_total = 0., 0.
    while True:
        # Load batch of data
        data = dataloader.getDualBatch(split, cfg)
        obj_feat = data['obj_feat']
        cxt_feat = data['cxt_feat']
        loc_feat = data['loc_feat']
        txt_labels = data['txt_labels']
        gt_index_list = data['gt_index_list']

        # Forward
        pred = listener(obj_feat, cxt_feat, loc_feat, txt_labels)
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
    data_pickle  = cfg['DATASET']['DATA_PICKLE']
    vocab_pickle = cfg['DATASET']['VOCAB_PICKLE']
    dataloader = DataLoader(data_pickle, vocab_pickle)
    dataloader.shuffle_refs(split='trainval')

    # 3. Build model
    speaker_checkpoint_path = cfg['PRETRAINED']['SPEAKER_CHECKPOINT_PATH']
    speaker_checkpoint = torch.load(speaker_checkpoint_path)
    logger.info('Load pretrained speaker from %s.' % speaker_checkpoint_path)
    speaker = speaker_checkpoint['speaker']
    speaker.cuda()

    listener_checkpoint_path = cfg['PRETRAINED']['LISTENER_CHECKPOINT_PATH']
    listener_checkpoint = torch.load(listener_checkpoint_path)
    logger.info('Load pretrained listener from %s.' % listener_checkpoint_path)
    listener = listener_checkpoint['listener']
    listener.cuda()

    # 4. Set up optimizer & criterion
    speaker_lr = cfg['OPTIMIZATION']['SPEAKER_LR']
    listener_lr = cfg['OPTIMIZATION']['LISTENER_LR']
    speaker_optimizer = torch.optim.Adam(speaker.parameters(),
                                         lr=speaker_lr)
    listener_optimizer = torch.optim.Adam(listener.parameters(),
                                         lr=listener_lr)

    speaker_crit = DualSpeakerCriterion()
    listener_crit = DualListenerCriterion()
    speaker_crit.cuda()
    listener_crit.cuda()

    # 5. Training
    epochs = cfg['OPTIMIZATION']['EPOCHS']
    trn_split = cfg['DATASET']['TRAIN']
    dev_split = cfg['DATASET']['DEV']
    global_steps = 0
    for epoch in range(epochs):
        # Train one epoch
        epoch_steps = train_epoch(speaker, listener, dataloader, speaker_optimizer, listener_optimizer,
            speaker_crit, listener_crit, epoch, global_steps, trn_split, logger, cfg)
        global_steps += epoch_steps

        # Perform evaluation
        if (epoch + 1) % cfg['MONITOR']['EVAL_FREQ'] == 0:
            validate(listener, dataloader, epoch, global_steps, dev_split, logger, cfg)

        # LR decay
        if (epoch + 1) % cfg['OPTIMIZATION']['LR_DECAY_FREQ'] == 0:
            speaker_lr = speaker_lr * 0.1
            listener_lr = listener_lr * 0.1
            set_lr(speaker_optimizer, speaker_lr)
            set_lr(listener_optimizer, listener_lr)

        # Save checkpoint
        if (epoch + 1) % cfg['OUTPUT']['SAVE_FREQ'] == 0:
            model_path = os.path.join(save_dir, 'pytorch_model_{}.bin'.format(epoch + 1))
            checkpoint = {}
            checkpoint['speaker'] = speaker
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

    # Get the default config & merge from cfg_file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # Call main
    main(cfg)
