#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):# {{{
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.dropout = cfg['SPEAKER']['ENCODER']['DROPOUT']
        self.loc_dim = cfg['SPEAKER']['ENCODER']['LOC_DIM']
        self.hidden_dim = cfg['SPEAKER']['ENCODER']['HIDDEN_DIM']
        self.output_dim = cfg['SPEAKER']['ENCODER']['OUTPUT_DIM']
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
        """Encoder takes object, context and location features as input and output
        encoded feature.

        Args:
            obj_feat (torch.Tensor): Feature of object, which can be fc7 of vgg16
                or pool5 of faster-rcnn.
            cxt_feat (torch.Tensor): Feature of context. The feature of the whole
                image is used as context.
            loc_feat (torch.Tensor): Feature of location. The loc_feat encodes
                the location of object in the image.
        
        Returns:
            ~torch.Tensor:

            * **encoded_feat**: The output of the encoder, which takes three types
                of feature as input, and output the combined feature.

        """
        # MLP projection
        obj_feat = self.obj_mlp(obj_feat)
        cxt_feat = self.cxt_mlp(cxt_feat)

        # The encoder combines these three types of feature
        # to output a combined feature
        encoded_feat = torch.cat([obj_feat, cxt_feat, loc_feat], -1)
        encoded_feat = self.mlp(encoded_feat)
        return encoded_feat# }}}


class Decoder(nn.Module):# {{{
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.vocab_size = cfg['DATASET']['VOCAB_SIZE']
        self.lstm_dim = cfg['SPEAKER']['DECODER']['LSTM_DIM']
        self.embd_dim = cfg['SPEAKER']['DECODER']['EMBD_DIM']
        self.lstm_dropout = cfg['SPEAKER']['DECODER']['LSTM_DROPOUT']
        self.embd_dropout = cfg['SPEAKER']['DECODER']['EMBD_DROPOUT']
        self.encoder_out_dim = cfg['SPEAKER']['ENCODER']['OUTPUT_DIM']

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embd_dim)
        self.embedding_dropout = nn.Dropout(self.embd_dropout)
        self.lstm = nn.LSTM(input_size=self.encoder_out_dim + self.embd_dim,
                            hidden_size=self.lstm_dim,
                            batch_first=True)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout)
        self.fc = nn.Linear(self.lstm_dim, self.vocab_size)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(1, batch_size, self.lstm_dim),
                weight.new_zeros(1, batch_size, self.lstm_dim))

    def forward(self, vis_input, txt_input, init_state=None):
        """Decoder takes the visual features (vis_input) and text labels
        (txt_input) as input, and samples a sequence of words.

        The vis_input is the output of the encoder, which encodes enough info
            of the object.

        Args:
            vis_input (torch.Tensor): The output of the encoder.
            txt_input (torch.Tensor): The input text labels.
            init_state (tuple of torch.Tensor): Initial state of LSTM, default is None.

        Returns:
            ~torch.Tensor:

            * **fc_output**: The sampled words.

        """
        # Word embedding
        txt_input = self.embedding_layer(txt_input)
        txt_input = self.embedding_dropout(txt_input)
        # Forward LSTM
        lstm_input = torch.cat(
                (vis_input[:, None, :].repeat(1, txt_input.size(1), 1), txt_input), -1)
        if init_state is None:
            init_state = self.init_hidden(batch_size=txt_input.size(0))
            output, hidden = self.lstm(lstm_input, init_state)
        else:
            output, hidden = self.lstm(lstm_input, init_state)
        output = self.lstm_dropout(output)

        batch_size, seq_len, lstm_dim = output.size(0), output.size(1), output.size(2)
        fc_input = output.contiguous().view(batch_size * seq_len, lstm_dim)
        fc_output = self.fc(fc_input)
        fc_output = fc_output.view(batch_size, seq_len, self.vocab_size)
        return fc_output, hidden# }}}


class Speaker(nn.Module):# {{{
    """Speaker is a region-to-phrase model, which follows the standard encoder-decoder
    framework. It can generate a phrase that describes the given object."""
    def __init__(self, cfg):
        super(Speaker, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, obj_feat, cxt_feat, loc_feat, txt_input, init_state=None):
        vis_input = self.encoder(obj_feat, cxt_feat, loc_feat)
        output, hidden = self.decoder(vis_input, txt_input, init_state)
        return output, hidden

    def sample(self, obj_feat, cxt_feat, loc_feat, seq_len, eos_token):
        hidden = None
        batch_size = obj_feat.size(0)
        # Prepare visual features
        vis_input = self.encoder(obj_feat, cxt_feat, loc_feat)
        # Start token
        txt_input = np.ones((batch_size, 1), dtype='int64') * eos_token
        txt_input = Variable(torch.from_numpy(txt_input).cuda())
        # Word index (output)
        word_index = torch.zeros((batch_size, seq_len))
        for t in range(seq_len):
            pred, hidden = self.decoder(vis_input, txt_input, hidden)
            pred = F.softmax(pred, dim=2)  # [batch_size, 1, vocab_size]
            txt_input = torch.argmax(pred, dim=2)
            word_index[:, t] = txt_input[:, 0]
        return word_index# }}}

    def sample_for_listener(self, obj_feat, cxt_feat, loc_feat, seq_len, word2idx):
        hidden = None
        batch_size = obj_feat.size(0)
        # Prepare visual features
        vis_input = self.encoder(obj_feat, cxt_feat, loc_feat)
        # Start token
        eos_token = word2idx['<EOS>']
        txt_input = np.ones((batch_size, 1), dtype='int64') * eos_token
        txt_input = Variable(torch.from_numpy(txt_input).cuda())
        # Word index & log_probs are outputs
        word_index = torch.zeros((batch_size, seq_len))
        log_probs = Variable(torch.zeros((batch_size, seq_len)).cuda())

        for t in range(seq_len):
            pred, hidden = self.decoder(vis_input, txt_input, hidden)
            pred = F.softmax(pred, dim=2)
            m = torch.distributions.Categorical(pred)
            txt_input = m.sample()
            word_index[:, t] = txt_input[:, 0]
            log_probs[:, t] = m.log_prob(txt_input)[:, 0]

        # Post-processing
        generated_word_index = torch.zeros((batch_size, seq_len)).cuda()
        generated_phrase_mask = torch.zeros((batch_size, seq_len)).cuda()
        for b in range(batch_size):
            for t in range(seq_len):
                index = int(word_index[b][t].item())
                generated_phrase_mask[b, t] = 1.0
                if index == word2idx['<PAD>']:
                    generated_phrase_mask[b, t] = word2idx['<unk>']
                elif index == word2idx['<EOS>']:
                    # If sampled <EOS> token, we replace it with <PAD> since we do not
                    # feed <EOS> token to the listener, but we still fill the generated_phrase_mask
                    # to 1.0, since we hope the speaker to sample <EOS> token at the end of the phrase
                    if t == 0:
                        generated_word_index[b, t] = index
                    else:
                        generated_word_index[b, t] = word2idx['<PAD>']
                    break
                else:
                    generated_word_index[b, t] = index

        generated_word_index = Variable(generated_word_index.long().cuda())
        generated_phrase_mask = Variable(generated_phrase_mask.cuda())

        # Chunk word index using max_len
        max_len = (generated_word_index != 0).sum(1).max().item()
        generated_word_index = generated_word_index[:, :max_len]

        return generated_word_index, generated_phrase_mask, log_probs
