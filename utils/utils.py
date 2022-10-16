#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import spacy
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm

# torch
import torch

logger = logging.getLogger(__name__)

def get_logger(log_file=None):# {{{
    """Set logger and return it.

    If the log_file is not None, log will be written into log_file.
    Else, log will be shown in the screen.

    Args:
        log_file (str): If log_file is not None, log will be written
            into the log_file.

    Returns:
        ~Logger

        * **logger**: An Logger object with customed config.

    """
    # Basic config
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Add filehandler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

    return logger# }}}


def set_seed_logger(cfg):# {{{
    """Experiments preparation, e.g., fix random seed, prepare checkpoint dir
    and set logger.

    Args:
        cfg (yacs.config): An yacs.config.CfgNode object.

    Returns:
        ~(Logger, str):

        * **logger**: An Logger object with customed config.
        * **save_dir**: Checkpoint dir to save models.

    """
    seed = cfg['MISC']['SEED']
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-gpu
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Prepare save dir
    if cfg['OUTPUT']['SAVE_NAME']:
        prefix = cfg['OUTPUT']['SAVE_NAME'] + '_'
    else:
        prefix = ''
    exp_name = prefix + datetime.datetime.now().strftime('%yY_%mM_%dD_%HH')
    save_dir = os.path.join(cfg['OUTPUT']['CHECKPOINT_DIR'], exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # build logger
    log_file = os.path.join(save_dir, 'log.txt')
    logger = get_logger(log_file)

    return logger, save_dir
# }}}


def dump_cfg(cfg, cfg_file):# {{{
    """Dump config of each experiment into file for backup.

    Args:
        cfg (yacs.config): An yacs.config.CfgNode object.
        cfg_file (str): Dump config to this file.

    """
    logger.info('Dump configs into {}'.format(cfg_file))
    logger.info('Using configs: ')
    logger.info(cfg)
    with open(cfg_file, 'w') as f:
        f.write(cfg.dump())# }}}


def load_file(file_path):# {{{
    """Read and return data from the given file.

    Args:
        file_path (str): Path of the file.

    Returns:
        ~List:
        
        * **data**: Readed data.

    """
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(line.strip())

    return data# }}}


def load_vg_info(vg_dir):# {{{
    """Load classes and attributes of Visual Genome dataset.

    Args:
        vg_dir (str): Dir to the visual genome dataset.

    Returns:
        ~(list, list):

        * **classes**: A list contains all class names in the VG dataset.
        * **attributes**: A list contains all attributes in the VG dataset.

    """
    classes = ['__background__']
    with open(os.path.join(vg_dir, 'objects_vocab.txt'), 'r') as f:
        for class_name in f.readlines():
            classes.append(class_name.split(',')[0].lower().strip())

    attributes = ['__no_attribute__']
    with open(os.path.join(vg_dir, 'attributes_vocab.txt'), 'r') as f:
        for att in f.readlines():
            attributes.append(att.split(',')[0].lower().strip())

    return classes, attributes# }}}


def load_glove_feats(glove_file):# {{{
    """Read glove vector of each word from the given glove_file (e.g. glove.840B.300d.txt).

    Args:
        glove_file: A txt file contains words and theirs glove vector.

    Returns:
        ~dict:
        A dictionary whose keys are words and values are glove vectors.

    """
    print('Load GloVe vector from {}.'.format(glove_file))
    glove_dict = {}
    with open(glove_file, 'r') as f:
        with tqdm(total=2196017, desc='Loading GloVe', ascii=True) as pbar:
            for line in f:
                tokens = line.split(' ')
                assert len(tokens) == 301
                word = tokens[0]
                vec = list(map(lambda x: float(x), tokens[1:]))
                glove_dict[word] = vec
                pbar.update(1)
    return glove_dict# }}}


def compute_iou(boxes, target):# {{{
    """Compute IoUs of proposals and ground truth bbox.

    The input bounding box is 0-index based and in the format of [x1, y1, x2, y2],
    where (x1, y1) and (x2, y2) are top-left and bottom-right coordinates.

    Args:
        bboxes (numpy.ndarray): Numpy array with shape of [N, 4], where N is the
            number of boxes.
        target (numpy.ndarray): Numpy array with shaoe of [4, ].
    
    Returns:
        ~numpy.ndarray:
        IoUs between bboxes and target. Its shape is [N, ], where N is the number of
        bboxes.

    """
    assert(target.ndim == 1 and boxes.ndim == 2)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    gt_area = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)

    assert(np.all(boxes_area >= 0))
    assert(np.all(gt_area >= 0))

    I_x1 = np.maximum(boxes[:, 0], target[0])
    I_y1 = np.maximum(boxes[:, 1], target[1])
    I_x2 = np.minimum(boxes[:, 2], target[2])
    I_y2 = np.minimum(boxes[:, 3], target[3])
    # intersect area
    i_area = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)

    IoUs = i_area / (boxes_area + gt_area - i_area)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs# }}}


def compute_cosine_similarity(feat_a, feat_b):# {{{
    """Compute cosine similarity between two vectors. Each vector is a 
    numpy.ndarray and its ndim is equal to 1.

    Args:
        feat_a (numpy.ndarray): A numpy.ndarray with shape of [N, ], where
            N is the dimension of the vector.
        feat_b (numpy.ndarray): A numpy.ndarray with shape of [N, ], where
            N is the dimension of the vector.

    Returns:
        ~float:
        Cosine similarity of the given two vectors.

    """
    return np.sum(feat_a * feat_b) / np.sqrt(np.sum(feat_a * feat_a) * np.sum(feat_b * feat_b))# }}}


def extract_nouns(phrase):# {{{
    """Extract nouns from the given phrase using Spacy.

    Args:
        phrase (str): A textual phrase.

    Returns:
        ~list:
        A list of noun words in the given phrase.

    """
    POS_OF_INTEREST = {'NOUN', 'NUM', 'PRON', 'PROPN'}

    # use spacy to extract POS tags
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(phrase)
    noun_tokens = [token.text for token in doc if token.pos_ in POS_OF_INTEREST]
    noun_tokens = [str(token) for token in noun_tokens]
    return noun_tokens# }}}


def unify_boxes(boxes):# {{{
    """Unify boxes that correpond to a phrase."""
    boxes = np.array(boxes)
    x_min = np.amin(boxes[:, 0])
    y_min = np.amin(boxes[:, 1])
    x_max = np.amax(boxes[:, 2])
    y_max = np.amax(boxes[:, 3])
    
    return [x_min, y_min, x_max, y_max]# }}}


def set_lr(optimizer, lr):
    """Update learning rate of the optimizer."""
    for group in optimizer.param_groups:
        group['lr'] = lr
