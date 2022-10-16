#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Normalize_Scale(nn.Module):# {{{
    def __init__(self, dim, init_norm=20):
        super(Normalize_Scale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        # Input is variable of (batch_size, dim)
        assert isinstance(bottom, Variable), 'bottom must be variable'
        bottom_normalized = F.normalize(bottom)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled# }}}


class VisualEncoder(nn.Module):# {{{
    def __init__(self, cfg):
        super(VisualEncoder, self).__init__()
        self.dropout = cfg['LISTENER']['VIS_ENC']['DROPOUT']
        self.loc_dim = cfg['LISTENER']['VIS_ENC']['LOC_DIM']
        self.hidden_dim = cfg['LISTENER']['VIS_ENC']['HIDDEN_DIM']
        self.output_dim = cfg['LISTENER']['VIS_ENC']['OUTPUT_DIM']
        self.feat_dim = self.hidden_dim * 2 + self.loc_dim

        if cfg['INPUT']['FEATURE_TYPE'] == 'fc7':
            feature_dim = 4096
        elif cfg['INPUT']['FEATURE_TYPE'] == 'pool5':
            feature_dim = 2048
        else:
            raise ValueError('Supported feature types are fc7 and pool5, \
                    but the given one is {}'.format(cfg['INPUT']['FEATURE_TYPE']))

        self.obj_mlp = nn.Sequential(nn.Linear(feature_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout))
        self.cxt_mlp = nn.Sequential(nn.Linear(feature_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout))
        self.mlp = nn.Sequential(nn.Linear(self.feat_dim, self.output_dim),
                                 nn.ReLU(),
                                 nn.Dropout(self.dropout))

    def forward(self, obj_feat, cxt_feat, loc_feat):
        """Visual Encoder takes object, context and location features as input and output
        encoded feature.

        Args:
            obj_feat (torch.Tensor): Feature of object, which can be fc7 of vgg16
                or pool5 of faster-rcnn.
            cxt_feat (torch.Tensor): Feature of context. The feature of the whole
                image is used as context.
            loc_feat (torch.Tensor): Feature of location. The loc_feat encodes the
                location of object in the image.

        Returns:
            ~torch.Tensor:

            * **encoded_feat**: The output of the encoder, which takes three types of
                features as input, and output the combined feature.

        """
        # MLP projection
        obj_feat = self.obj_mlp(obj_feat)
        cxt_feat = self.cxt_mlp(cxt_feat)

        # The visaul encoder combines these three types of feature
        # to output a combined feature
        encoded_feat = torch.cat([obj_feat, cxt_feat, loc_feat], -1)
        encoded_feat = self.mlp(encoded_feat)
        return encoded_feat# }}}


class LanguageEncoder(nn.Module):# {{{
    def __init__(self, cfg):
        super(LanguageEncoder, self).__init__()
        self.vocab_size = cfg['DATASET']['VOCAB_SIZE']
        self.lstm_dim = cfg['SPEAKER']['DECODER']['LSTM_DIM']
        self.embd_dim = cfg['SPEAKER']['DECODER']['EMBD_DIM']

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embd_dim)
        self.lstm = nn.LSTM(input_size=self.embd_dim,
                            hidden_size=self.lstm_dim,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, txt_input):
        # Variable length
        input_lengths = (txt_input != 0).sum(1)

        # Make ixs
        input_lengths_list = input_lengths.data.cpu().numpy().tolist()
        # List of sorted input_lengths
        sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()
        # List of int sort ixs (descending order)
        sorted_ixs = np.argsort(input_lengths_list)[::-1].tolist()
        s2r = {s: r for r, s in enumerate(sorted_ixs)}
        # List of int recover ixs
        recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]
        assert max(input_lengths_list) == txt_input.size(1)

        # Move to long tensor
        sorted_ixs = txt_input.data.new(sorted_ixs).long()
        recover_ixs = txt_input.data.new(recover_ixs).long()

        # Sort txt_input by descending order
        txt_input = txt_input[sorted_ixs]

        # Word embedding
        txt_input = self.embedding_layer(txt_input)
        txt_input = nn.utils.rnn.pack_padded_sequence(
                txt_input, sorted_input_lengths_list, batch_first=True)

        # Forward LSTM
        _, hidden = self.lstm(txt_input)

        # Recover hidden
        # We only use hidden states for the final hidden representation
        hidden = hidden[0]
        hidden = hidden[:, recover_ixs, :]
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(hidden.size(0), -1)
        return hidden# }}}


class Listener(nn.Module):
    def __init__(self, cfg):
        super(Listener, self).__init__()
        self.normalize = cfg['LISTENER']['NORMALIZE']
        self.vis_encoder = VisualEncoder(cfg)
        self.lan_encoder = LanguageEncoder(cfg)

    def forward(self, obj_feat, cxt_feat, loc_feat, txt_input):
        vis_feat = self.vis_encoder(obj_feat, cxt_feat, loc_feat)
        lan_feat = self.lan_encoder(txt_input)
        lan_feat = lan_feat[:, None, :]
        lan_feat = lan_feat.repeat(1, vis_feat.shape[1], 1)

        if self.normalize:
            vis_feat = F.normalize(vis_feat)
            lan_feat = F.normalize(lan_feat)
        
        pred = vis_feat * lan_feat
        pred = torch.sum(pred, dim=2)
        return pred
