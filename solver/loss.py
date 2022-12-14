#! /usr/bin/env python
# -*- coding: utf-8 -*-

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerCriterion(nn.Module):# {{{
    """Criterion for speaker warm-up."""
    def __init__(self):
        super(SpeakerCriterion, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        """Compute loss for speaker warm-up.

        Args:
            pred (torch.Tensor): The output of the speaker, with the shape of
                [batch_size, seq_len, vocab_size].
            target (torch.Tensor): The target of the speaker, with the shape of
                [batch_size, seq_len].

        """
        pred = pred.transpose(1, 2)
        mask = (target != 0).float()
        loss = self.crossentropy(pred, target)
        loss *= mask
        return loss.sum() / mask.sum()# }}}


class ListenerCriterion(nn.Module):# {{{
    """Criterion for listener warm-up."""
    def __init__(self):
        super(ListenerCriterion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce_crit = nn.BCELoss()

    def forward(self, pred, target):
        """Compute loss for listener warm-up.

        Args:
            pred (torch.Tensor): The output of the listener, with the shape of
                [batch_size, num_proposals].
            target (torch.Tensor): The target of the listener, with the shape of
                [batch_size, num_proposals]. The target is the binary label for
                each proposal, where the item is set to 1 if the iou of the proposal
                and the groundtruth larger than 0.5.

        """
        pred = self.sigmoid(pred)
        loss = self.bce_crit(pred, target)
        return loss# }}}


class DualSpeakerCriterion(nn.Module):# {{{
    """Criterion for speaker of dual learning."""
    def __init__(self):
        super(DualSpeakerCriterion, self).__init__()

    def forward(self, reward, log_prob, mask):
        """Compute loss for speaker of dual learning.

        Args:
            reward (List): A list of reward generated by the listener.
            log_prob (torch.Tensor): The log probability of the sampled words,
                with the shape of [batch_size, seq_len].
            mask (torch.Tensor): The mask of the sampled words, with the shape
                of [batch_size, seq_len]

        """
        batch_size = len(reward)
        for b in range(batch_size):
            if reward[b] == 0:
                mask[b, :] = 0.0

        log_prob = log_prob * mask
        if mask.sum() == 0:
            return torch.sum(-log_prob)
        else:
            return torch.sum(-log_prob) / mask.sum()# }}}


class DualListenerCriterion(nn.Module):# {{{
    """Criterion for listener of dual learning."""
    def __init__(self):
        super(DualListenerCriterion, self).__init__()

    def forward(self, pred, reward, target):
        """Compute loss for listener of dual learning.

        Args:
            pred (torch.Tensor): The output of the listener, with the shape of
                [batch_size, num_proposals].
            reward (List): A list of reward generated also by the speaker.
            target (List): A list of pseudo groundtruth evaluated by the speaker.

        """
        loss = 0.
        num = 0.

        batch_size = pred.size(0)
        pred = F.log_softmax(pred, dim=1)
        for b in range(batch_size):
            if target[b] != -1:
                loss += pred[b][target[b]] * reward[b]
                num += 1

        if num == 0:
            return -torch.sum(pred) * num
        else:
            loss /= num
            return -loss# }}}
