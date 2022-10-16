#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import pickle
import logging
import numpy as np

# torch
import torch
from torch.autograd import Variable

logger = logging.getLogger(__name__)

Referit_Object_Feature_Dir = '/mnt/disk6T/Data/Research/Referring-Expression-Comprehension/Dual-Listener-Speaker/referit/vg_det_pool5_feats'
Referit_Context_Feature_Dir = '/mnt/disk6T/Data/Research/Referring-Expression-Comprehension/Dual-Listener-Speaker/referit/context_pool5_features'
Flickr30k_Object_Feature_Dir = '/mnt/disk6T/Data/Research/Referring-Expression-Comprehension/Dual-Listener-Speaker/flickr30k/vg_det_pool5_feats'
Flickr30k_Context_Feature_Dir = '/mnt/disk6T/Data/Research/Referring-Expression-Comprehension/Dual-Listener-Speaker/flickr30k/context_pool5_features'

class DataLoader(object):
    """The DataLoader responsible for loading weakly supervised phrase grounding dataset,
    i.e., flickr30k and referit, to train the dual-listener-speaker model.

    Args:
        data_pickle (str): The pickle file contains data items of the dataset.
        vocab_pickle (str): The pickle file contains vocabulary of the dataset.

    """
    def __init__(self, data_pickle, vocab_pickle):# {{{
        # Load the data pickle file which contains items of datasets
        logger.info('Loading dataset from %s' % data_pickle)
        with open(data_pickle, 'rb') as f:
            self.refs = pickle.load(f)
        self.Refs = {ref['refexp_id']: ref for ref in self.refs}

        # Ref iterators for each split
        self.split_ix = {}
        self.iterators = {}
        for ref in self.refs:
            split = ref['split']
            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0
            self.split_ix[split] += [ref['refexp_id']]
        for k, v in self.split_ix.items():
            logger.info('Assigned %d refs to %s split.' % (len(v), k))

        # Consturct mapping from an image id to its objects (i.e., refexp_id)
        self.Images = {}
        for ref in self.refs:
            image_name = ref['image_name']
            if image_name not in self.Images:
                self.Images[image_name] = [ref['refexp_id']]
            else:
                self.Images[image_name] += [ref['refexp_id']]
        logger.info('Found %d images in the dataset.' % (len(self.Images)))

        self.image_split_ix = {}
        self.image_iterators = {}
        for image_name, refexp_ids in self.Images.items():
            split = self.Refs[refexp_ids[0]]['split']
            if split not in self.image_split_ix:
                self.image_split_ix[split] = []
                self.image_iterators[split] = 0
            self.image_split_ix[split] += [image_name]
        for k, v in self.image_split_ix.items():
            logger.info('Assigned %d images to %s split.' % (len(v), k))

        # Load vocabulary from the vocab pickle file
        with open(vocab_pickle, 'rb') as f:
            self.word2idx, self.idx2word = pickle.load(f)# }}}

    @property
    def vocab_size(self):
        return len(self.word2idx)

    # Return length of the dataset
    def get_len(self, split):
        assert split in self.split_ix, "The supported splits are {}, but the \
                given one is {}".format(list(self.split_ix.keys()), split)
        return len(self.split_ix[split])

    # Shuffle refexp ids
    def shuffle_refs(self, split):
        assert split in self.split_ix, "The supported splits are {}, but the \
                given one is {}".format(list(self.split_ix.keys()), split)
        random.shuffle(self.split_ix[split])

    # Shuffle image names
    def shuffle_images(self, split):
        assert split in self.image_split_ix, "The supported splits are {}, but \
                the given one is {}".format(list(self.image_split_ix.keys()), split)
        random.shuffle(self.image_split_ix[split])

    # Reset iterator[split] to 0
    def resetIterator(self, split):
        assert split in self.iterators, "The supported splits are {}, but the \
                given one is {}".format(list(self.iterators.keys()), split)
        self.iterators[split] = 0

    # Reset image_iterators[split] to 0
    def resetImageIterator(self, split):
        assert split in self.image_iterators, "The supported splits are {}, but \
                the given one is {}".format(list(self.image_iterators.keys()), split)
        self.image_iterators[split] = 0

    def getSpeakerBatch(self, split, cfg):# {{{
        """Return batch data for training/dev the speaker."""
        # Options
        dataset = cfg['DATASET']['NAME']
        seq_len = cfg['INPUT']['MAX_SEQ_LEN']
        feat_type = cfg['INPUT']['FEATURE_TYPE']
        batch_size = cfg['OPTIMIZATION']['BATCH_SIZE']
        loc_feat_dim = cfg['SPEAKER']['ENCODER']['LOC_DIM']

        if feat_type == 'fc7':
            feat_dim = 4096
        elif feat_type == 'pool5':
            feat_dim = 2048
        else:
            raise ValueError('The supported feat_type are fc7 and pool5, \
                    but the given one is %s' % feat_type)

        if dataset == 'referit':
            obj_feat_dir = Referit_Object_Feature_Dir
            cxt_feat_dir = Referit_Context_Feature_Dir
        elif dataset == 'flickr30k':
            obj_feat_dir = Flickr30k_Object_Feature_Dir
            cxt_feat_dir = Flickr30k_Context_Feature_Dir
        else:
            raise ValueError('The supported dataset are referit and flickr30k, \
                    but the given one is %s' % dataset)

        # List contains all refexps of refs in this split
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False

        ann_id_list = []
        obj_feat = np.zeros((batch_size, feat_dim), dtype='float32')
        cxt_feat = np.zeros((batch_size, feat_dim), dtype='float32')
        loc_feat = np.zeros((batch_size, loc_feat_dim), dtype='float32')
        txt_input_labels = np.zeros((batch_size, seq_len), dtype='int32')
        txt_target_labels = np.zeros((batch_size, seq_len), dtype='int32')

        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next

            refexp_id = split_ix[ri]
            ref = self.Refs[refexp_id]
            tokens = ref['tokens']  # e.g., [10, 3, 4].
            image_name = ref['image_name']  # e.g., 1273.jpg.
            image_id = image_name[:-4]  # e.g., 1273.
            ann_id = ref['ann_id']  # e.g.,, 1273_1.
            # The index of the pseudo bbox in the proposals
            pseudo_bbox_index = ref['pseudo_bbox_index']  # e.g., 3.

            # Add ann_id to ann_id_list
            ann_id_list.append(ann_id)

            # Load object feature
            tmp_proposal_feats = np.load(os.path.join(obj_feat_dir, image_id + '.npz'))
            # Fetch the feature of target pseudo object with pseudo_bbox_index
            tmp_obj_feat = tmp_proposal_feats['local_feature'][pseudo_bbox_index, :]
            tmp_loc_feat = tmp_proposal_feats['spatial_feat'][pseudo_bbox_index, :]
            # Load context feature
            tmp_cxt_feat = np.load(os.path.join(cxt_feat_dir, image_id + '_fc7.npy'))
            # Fill the i-th item
            obj_feat[i, :] = tmp_obj_feat
            cxt_feat[i, :] = tmp_cxt_feat
            loc_feat[i, :] = tmp_loc_feat

            # Prepare txt data
            if len(tokens) > seq_len - 1:
                tokens = tokens[:seq_len - 1]
            input_tokens = [self.word2idx['<EOS>']] + tokens
            target_tokens = tokens + [self.word2idx['<EOS>']]
            txt_input_labels[i, :len(input_tokens)] = np.array(input_tokens, dtype='int32')
            txt_target_labels[i, :len(target_tokens)] = np.array(target_tokens, dtype='int32')

        # Convert to Variables
        obj_feat = Variable(torch.from_numpy(obj_feat).cuda())
        cxt_feat = Variable(torch.from_numpy(cxt_feat).cuda())
        loc_feat = Variable(torch.from_numpy(loc_feat).cuda())
        txt_input_labels = Variable(torch.from_numpy(txt_input_labels).long().cuda())
        txt_target_labels = Variable(torch.from_numpy(txt_target_labels).long().cuda())

        # Chunk txt_input_labels and txt_target_labels using max_len
        max_len = (txt_input_labels != 0).sum(1).max().item()
        txt_input_labels = txt_input_labels[:, :max_len]
        txt_target_labels = txt_target_labels[:, :max_len]

        # Return
        data = {}
        data['obj_feat'] = obj_feat
        data['cxt_feat'] = cxt_feat
        data['loc_feat'] = loc_feat
        data['ann_id_list'] = ann_id_list
        data['txt_input_labels'] = txt_input_labels
        data['txt_target_labels'] = txt_target_labels
        data['wrapped'] = wrapped
        return data# }}}

    def getListenerBatch(self, split, cfg): # {{{
        """Return batch data for training/dev the listener."""
        # Options
        dataset = cfg['DATASET']['NAME']
        seq_len = cfg['INPUT']['MAX_SEQ_LEN']
        feat_type = cfg['INPUT']['FEATURE_TYPE']
        batch_size = cfg['OPTIMIZATION']['BATCH_SIZE']
        loc_feat_dim = cfg['LISTENER']['VIS_ENC']['LOC_DIM']

        if feat_type == 'fc7':
            feat_dim = 4096
        elif feat_type == 'pool5':
            feat_dim = 2048
        else:
            raise ValueError('The supported feat_type are fc7 and pool5, \
                    but the given one is %s' % feat_type)

        if dataset == 'referit':
            topK_proposals = 30
            obj_feat_dir = Referit_Object_Feature_Dir
            cxt_feat_dir = Referit_Context_Feature_Dir
        elif dataset == 'flickr30k':
            topK_proposals = 30
            obj_feat_dir = Flickr30k_Object_Feature_Dir
            cxt_feat_dir = Flickr30k_Context_Feature_Dir
        else:
            raise ValueError('The supported dataset are referit and flickr30k, \
                    but the given one is %s' % dataset)

        # List contains all refexps of refs in this split
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False

        gt_index_list = []
        obj_feat = np.zeros((batch_size, topK_proposals, feat_dim), dtype='float32')
        cxt_feat = np.zeros((batch_size, topK_proposals, feat_dim), dtype='float32')
        loc_feat = np.zeros((batch_size, topK_proposals, loc_feat_dim), dtype='float32')
        pos_labels = np.zeros((batch_size, topK_proposals), dtype='float32')
        txt_labels = np.zeros((batch_size, seq_len), dtype='int32')

        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next

            refexp_id = split_ix[ri]
            ref = self.Refs[refexp_id]
            tokens = ref['tokens']  # e.g., [10, 3, 4].
            image_name = ref['image_name']  # e.g., 1273.jpg.
            gt_index = ref['gt_index'] if 'gt_index' in ref else [] # e.g., [10]
            pseudo_pos_all = ref['pseudo_positive_all'] \
                                       if 'pseudo_positive_all' in ref  else []  # e.g., [3, 19].


            # Add gt_index to gt_index_list
            gt_index_list.append(gt_index)
            
            # Load object feature
            tmp_proposal_feats = np.load(os.path.join(obj_feat_dir, image_name[:-4] + '.npz'))
            # Fetch the feature of target pseudo object with pseudo_bbox_index
            tmp_obj_feat = tmp_proposal_feats['local_feature']
            tmp_loc_feat = tmp_proposal_feats['spatial_feat']
            # Load context feature
            num_proposals = tmp_obj_feat.shape[0]
            tmp_cxt_feat = np.load(os.path.join(cxt_feat_dir, image_name[:-4] + '_fc7.npy'))
            tmp_cxt_feat = tmp_cxt_feat.reshape((1, -1))
            tmp_cxt_feat = tmp_cxt_feat.repeat(num_proposals, axis=0)
            # Fill the i-th item
            obj_feat[i, :num_proposals] = tmp_obj_feat
            cxt_feat[i, :num_proposals] = tmp_cxt_feat
            loc_feat[i, :num_proposals] = tmp_loc_feat

            # Prepare pos_all_labels
            for pos_id in pseudo_pos_all:
                pos_labels[i][pos_id] = 1

            # Prepare txt data
            txt_labels[i, :len(tokens)] = np.array(tokens, dtype='int32')

        # Convert to Variables
        obj_feat = Variable(torch.from_numpy(obj_feat).cuda())
        cxt_feat = Variable(torch.from_numpy(cxt_feat).cuda())
        loc_feat = Variable(torch.from_numpy(loc_feat).cuda())
        pos_labels = Variable(torch.from_numpy(pos_labels).cuda())
        txt_labels = Variable(torch.from_numpy(txt_labels).long().cuda())

        # Chunk txt_labels using max_len
        max_len = (txt_labels != 0).sum(1).max().item()
        txt_labels = txt_labels[:, :max_len]

        # Return
        data = {}
        data['obj_feat'] = obj_feat
        data['cxt_feat'] = cxt_feat
        data['loc_feat'] = loc_feat
        data['pos_labels'] = pos_labels
        data['txt_labels'] = txt_labels
        data['gt_index_list'] = gt_index_list
        data['wrapped'] = wrapped
        return data# }}}

    def getDualBatch(self, split, cfg): # {{{
        """Return batch data for training/dev the dual-listener-speaker."""
        # Options
        dataset = cfg['DATASET']['NAME']
        seq_len = cfg['INPUT']['MAX_SEQ_LEN']
        feat_type = cfg['INPUT']['FEATURE_TYPE']
        batch_size = cfg['OPTIMIZATION']['BATCH_SIZE']
        loc_feat_dim = cfg['LISTENER']['VIS_ENC']['LOC_DIM']

        if feat_type == 'fc7':
            feat_dim = 4096
        elif feat_type == 'pool5':
            feat_dim = 2048
        else:
            raise ValueError('The supported feat_type are fc7 and pool5, \
                    but the given one is %s' % feat_type)

        if dataset == 'referit':
            topK_proposals = 30
            obj_feat_dir = Referit_Object_Feature_Dir
            cxt_feat_dir = Referit_Context_Feature_Dir
        elif dataset == 'flickr30k':
            topK_proposals = 30
            obj_feat_dir = Flickr30k_Object_Feature_Dir
            cxt_feat_dir = Flickr30k_Context_Feature_Dir
        else:
            raise ValueError('The supported dataset are referit and flickr30k, \
                    but the given one is %s' % dataset)

        # List contains all refexps of refs in this split
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False

        gt_index_list = []
        obj_feat = np.zeros((batch_size, topK_proposals, feat_dim), dtype='float32')
        cxt_feat = np.zeros((batch_size, topK_proposals, feat_dim), dtype='float32')
        loc_feat = np.zeros((batch_size, topK_proposals, loc_feat_dim), dtype='float32')
        txt_labels = np.zeros((batch_size, seq_len), dtype='int32')
        txt_input_labels = np.zeros((batch_size, seq_len), dtype='int32')
        txt_target_labels = np.zeros((batch_size, seq_len), dtype='int32')

        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next

            refexp_id = split_ix[ri]
            ref = self.Refs[refexp_id]
            tokens = ref['tokens']  # e.g., [10, 3, 4].
            image_name = ref['image_name']  # e.g., 1273.jpg.
            gt_index = ref['gt_index'] if 'gt_index' in ref else [] # e.g., [10]

            # Add gt_index to gt_index_list
            gt_index_list.append(gt_index)
            
            # Load object feature
            tmp_proposal_feats = np.load(os.path.join(obj_feat_dir, image_name[:-4] + '.npz'))
            # Fetch the feature of target pseudo object with pseudo_bbox_index
            tmp_obj_feat = tmp_proposal_feats['local_feature']
            tmp_loc_feat = tmp_proposal_feats['spatial_feat']
            # Load context feature
            num_proposals = tmp_obj_feat.shape[0]
            tmp_cxt_feat = np.load(os.path.join(cxt_feat_dir, image_name[:-4] + '_fc7.npy'))
            tmp_cxt_feat = tmp_cxt_feat.reshape((1, -1))
            tmp_cxt_feat = tmp_cxt_feat.repeat(num_proposals, axis=0)
            # Fill the i-th item
            obj_feat[i, :num_proposals] = tmp_obj_feat
            cxt_feat[i, :num_proposals] = tmp_cxt_feat
            loc_feat[i, :num_proposals] = tmp_loc_feat

            # Prepare txt data
            txt_labels[i, :len(tokens)] = np.array(tokens, dtype='int32')
            if len(tokens) > seq_len - 1:
                tokens = tokens[:seq_len - 1]
            input_tokens = [self.word2idx['<EOS>']] + tokens
            target_tokens = tokens + [self.word2idx['<EOS>']]
            txt_input_labels[i, :len(input_tokens)] = np.array(input_tokens, dtype='int32')
            txt_target_labels[i, :len(target_tokens)] = np.array(target_tokens, dtype='int32')

        # Convert to Variables
        obj_feat = Variable(torch.from_numpy(obj_feat).cuda())
        cxt_feat = Variable(torch.from_numpy(cxt_feat).cuda())
        loc_feat = Variable(torch.from_numpy(loc_feat).cuda())
        txt_labels = Variable(torch.from_numpy(txt_labels).long().cuda())
        txt_input_labels = Variable(torch.from_numpy(txt_input_labels).long().cuda())
        txt_target_labels = Variable(torch.from_numpy(txt_target_labels).long().cuda())

        # Chunk txt_labels, txt_input_labels and txt_target_labels using max_len
        max_len = (txt_input_labels != 0).sum(1).max().item()
        txt_labels = txt_labels[:, :max_len - 1]
        txt_input_labels = txt_input_labels[:, :max_len]
        txt_target_labels = txt_target_labels[:, :max_len]

        # Return
        data = {}
        data['obj_feat'] = obj_feat
        data['cxt_feat'] = cxt_feat
        data['loc_feat'] = loc_feat
        data['txt_labels'] = txt_labels
        data['txt_input_labels'] = txt_input_labels
        data['txt_target_labels'] = txt_target_labels
        data['gt_index_list'] = gt_index_list
        data['wrapped'] = wrapped
        return data# }}}

    def decode_phrase(self, word_index):# {{{
        """Decode phrase based on the given word_index.

        The word_index is a torch.Tensor, whose shape is [N, seq_len].

        Args:
            word_index (torch.Tensor): The indices of words.

        Return:
            ~List

            * **decoded_phrase_list**: A List of length :obj:`N`, and each element
                is a string.

        """
        N, seq_len = word_index.size(0), word_index.size(1)
        decoded_phrase_list = []
        for n in range(N):
            # A phrase is a word list
            tmp_phrase = []
            for t in range(seq_len):
                tmp_index = int(word_index[n][t].item())
                tmp_phrase.append(self.idx2word[tmp_index])
                if tmp_index == self.word2idx['<EOS>']:
                    break
            decoded_phrase_list.append(' '.join(tmp_phrase[:-1]))
        return decoded_phrase_list# }}}
