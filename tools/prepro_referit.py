#! /usr/bin/env
# -*- coding: utf-8 -*-

import os
import re
import json
import pickle
import argparse
import skimage.io
import numpy as np
import scipy.io as sio

# utils
from utils.utils import (
    compute_iou, 
    load_file, 
    load_vg_info,
    load_glove_feats,
    extract_nouns,
    compute_cosine_similarity,
)

Referit_Dir = '/mnt/disk250G/Datasets/ReferIt/'
Referit_Mask_Dir = '/mnt/disk250G/Datasets/ReferIt/ImageCLEF/mask'


def build_vocab_from_file(vocab_file, data_dir): # {{{
    """Build vocabulary from the given file.

    The given file contains words to build vocabulary. We additionally
    add some speacial tokens, e.g., <PAD>, <EOS>, to this vocabulary.
    We then dump the built vocabulary into data_dir.

    Args:
        vocab_file (str): File contains words to build vocabulary.
        data_dir (str): Dir to dump the vocabulary.

    Returns:
        ~(dict, dict):

        * **word2idx**: Map word to index.
        * **idx2word**: Map index to word.

    """
    vocab = ['<PAD>']  # <PAD> token is at 0-index.
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
        vocab += [word.strip() for word in lines]

    vocab += ['<EOS>']
    assert len(vocab) == 8802  # 8800 + 2

    word2idx = {word: idx for (idx, word) in enumerate(vocab)}
    idx2word = {idx: word for (idx, word) in enumerate(vocab)}

    with open(os.path.join(data_dir, 'referit_vocab.pkl'), 'wb') as f:
        pickle.dump((word2idx, idx2word), f)

    return word2idx, idx2word# }}}


def encode_phrase(phrase, word2idx, MAX_WORDS=20):# {{{
    """Convert phrase into number tokens using the given word2idx.

    Args:
        phrase (str): Raw textual phrase.
        word2idx (dict): A dictionary mapping word to index.
        MAX_WORDS (int): Phrase longer than this value will be trimed.

    Returns:
        ~(list, List):

        * **sentence**: Words list of the textual phrase.
        * **encoded_tokens**: Encoded tokens which is int list.

    """
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    splits = SENTENCE_SPLIT_REGEX.split(phrase.strip())
    sentence = [s.lower() for s in splits if len(s.strip()) > 0]

    # remove .
    if sentence[-1] == '.':
        sentence = sentence[:-1]

    encoded_tokens = [(word2idx[s] if s in word2idx else word2idx['<unk>'])
                      for s in sentence]

    if len(encoded_tokens) > MAX_WORDS:
        encoded_tokens = encoded_tokens[:MAX_WORDS]

    return sentence, encoded_tokens# }}}


def read_refs(data_dir, img_dir):# {{{
    """Load referit dataset from RealGames.txt file.

    Load bounding boxes and their describing textual phrases from RealGames.txt file.
    We re-organize the data into a python list that each element represents a training sample.
    Each element is a python dictionary contains the following key-value pairs.

    * refexp_id (int): The unique id for the textual phrase (e.g., 0).
    * ann_id (str): The id of the bounding box (e.g., 1273_1).
    * image_name (str): The name of the image (e.g., 1273.jpg).
    * image_size (tuple): The width and height of the image (e.g., (519, 304)).
    * raw_phrase (str): The textual phrase of the object (e.g., The man on the left).
    * word_list (list): Split textual phrase into list of words.
    * tokens (list): Encoded tokens of the textual phrase, i.e., convert
        the textual phrase into number list using vocabulary (e.g., [0, 3, 5, 6]).
    * gt_bbox (list): The coordinates of the bounding box with the format 
        of [x1, y1, x2, y2]. Here (x1, y1) and (x2, y2) are the top-left and 
        bottom-right coordinates (e.g., [0., 0., 124., 121.]).
    * split (str): This sample belongs to which split? trainval or test (e.g., trainval).

    Args:
        data_dir (str): Dir to store the preprocessed data.
        img_dir (str): Dir of the images.

    """
    ann_file = os.path.join(Referit_Dir, 'RealGames.txt')
    assert os.path.isfile(ann_file), \
        'Please make sure the RealGames.txt file under {}.'.format(Referit_Dir)
    print('Load dataset from %s.' % ann_file)

    # Load trainval/test splits
    trainval_file = os.path.join(data_dir, 'referit_trainval_imlist.txt')
    test_file = os.path.join(data_dir, 'referit_test_imlist.txt')
    trainval_split = load_file(trainval_file)
    test_split = load_file(test_file)

    # Build vocabulary for encoding phrase
    vocab_file = os.path.join(data_dir, 'vocabulary.txt')
    word2idx, _  = build_vocab_from_file(vocab_file, data_dir)

    refs = []
    with open(ann_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            # Example of annotation line
            # 8756_2.jpg~sunray at very top~.33919597989949750~.023411371237458192
            ann_id, phrase, _ = line.strip().split('~', 2)
            ann_id = ann_id.split('.')[0]
            img_id = ann_id.split('_')[0]

            # Read image to get width and height
            image_name = img_id + '.jpg'
            im = skimage.io.imread(os.path.join(img_dir, image_name))
            width, height = im.shape[1], im.shape[0]

            # Convert the phrase into tokens
            word_list, encoded_tokens = encode_phrase(phrase, word2idx)

            # Load gt_bbox from Referit_Mask_Dir
            mask = sio.loadmat(os.path.join(Referit_Mask_Dir, ann_id + '.mat'))['segimg_t']
            idx = np.nonzero(mask == 0)
            x_min, x_max = np.min(idx[1]), np.max(idx[1])
            y_min, y_max = np.min(idx[0]), np.max(idx[0])
            gt_bbox = [x_min, y_min, x_max, y_max]

            if img_id in trainval_split:
                split = 'trainval'
            elif img_id in test_split:
                split = 'test'
            else:
                raise ValueError

            refs += [{'refexp_id': i,
                      'ann_id': ann_id,
                      'image_name': image_name,
                      'image_size': (width, height),
                      'raw_phrase': phrase,
                      'word_list': word_list,
                      'tokens': encoded_tokens,
                      'gt_bbox': gt_bbox,
                      'split': split}]

    print('Found %s refs in the referit dataset.' % len(refs))

    # dump into referit_refs.json
    with open(os.path.join(data_dir, 'referit_refs.pkl'), 'wb') as f:
        pickle.dump(refs, f)# }}}


def get_class_attribute_info(vg_class_list, vg_attr_list, proposal_info):# {{{
    """Return class names and attributes of each bounding box in a image.

    Given all classes and attributes of Visual Genome dataset, and the proposals
    detected by the detection model (e.g., Faster RCNN) trained on Visual Genome.
    We return class names and attributes of each detected bounding box according to
    the detection resutls.

    Args:
        vg_class_list (list): A list contains all class names of VG dataset.
        vg_attr_list (list): A list contains all attributes of VG dataset.
        proposal_info (list): A dict contains detection results. It contains class 
            pred and attributes pred for each detected bounding box.

    Returns:
        ~(list, list):

        * **proposal_cls_list**: A list contains predicted class names of each bbox.
        * **proposal_attr_list**: A list contains predicted attributes of each bbox.

    """
    class_pred = proposal_info['classes']
    attr_pred = proposal_info['attributes']

    proposal_cls_list = []
    proposal_attr_list = []
    for cls, attr in zip(class_pred, attr_pred):
        proposal_cls_list.append(vg_class_list[cls])
        proposal_attr_list.append(vg_attr_list[attr])
    return proposal_cls_list, proposal_attr_list# }}}


def get_location_info(proposals, img_size):# {{{
    """Return location info of each bounding box according to their location in the image.

    We predefined some location words (e.g., 'top', 'left', 'bottom') and assign them to each
    bounding box according their location in the image.

    Args:
        proposals (numpy.ndarray): A numpy.ndarray with shape of [N, 4] where N is the number
            of proposals in the image. The second axis is organized as [x1, y1, x2, y2], where
            (x1, y1) and (x2, y2) are the top-left and bottom-right coordinates.
        img_size (tuple): A tuple contains width and height of the image.

    Return:
        list:
        A list of length N, which is the number of proposals in the image. Each element in the list
            is the assigned location words for the proposal.

    """
    width, height = img_size
    if proposals.ndim == 1: proposals = proposals.reshape((1, 4))

    location_info = []
    num_proposals = proposals.shape[0]
    for i in range(num_proposals):
        x1, y1, x2, y2 = proposals[i][0], proposals[i][1], proposals[i][2], proposals[i][3]

        candidate_word = []
        # (top, far) case
        if y2 < (height / 2.0):
            candidate_word.append('top')
            candidate_word.append('far')

        # (down, closest) case
        if y1 > (height / 2.0):
            candidate_word.append('down')
            candidate_word.append('closest')

        # left case
        if x2 < (width / 2.0):
            candidate_word.append('left')

        # right case
        if x1 > (width / 2.0):
            candidate_word.append('right')

        # middle case
        if x1 < (width / 2.0) and x1 > (width / 4.0) and \
                y1 < (height / 2.0) and y1 > (height / 4.0) and \
                x2 > (width / 2.0) and x2 < (width * 0.75) and \
                y2 > (height / 2.0) and y2 < (height * 0.75):
            candidate_word.append('middle')

        location_info.append(candidate_word)
    return location_info# }}}


def generate_pseudo_label(data_dir):# {{{
    """Generate pesudo label for each ref in the referit dataset.

    Assign a pesudo bounding box for each textual phrase. We compute similarities
    between textual phrases and bounding boxes, and select the most similar one as
    the pesudo bounding box. When compute the similarity, we consider three types
    info: class_label, attribute and location.

    Args:
        data_dir (str): Dir to store the preprocessed data.

    """
    # Load referit dataset
    with open(os.path.join(data_dir, 'referit_refs.pkl'), 'rb') as f:
        refs = pickle.load(f)

    # Load class labels and attributes of vg dataset
    vg_class_list, vg_attribute_list = load_vg_info(os.path.join('data', 'vg'))

    # Load glove vector
    glove_file = '/mnt/disk2T/zhiyuwang/Cache/Dual-learning-listener-speaker/glove.840B.300d.txt'
    glove_dict = load_glove_feats(glove_file)

    # Load proposals
    with open(os.path.join(data_dir, 'referit_vg_det_top30_hit0.796.pkl'), 'rb') as f:
        proposals_dict = pickle.load(f)

    print('Start to generate pseudo labels for refs.')
    for i, ref in enumerate(refs):
        split = ref['split']
        phrase = ref['raw_phrase']
        word_list = ref['word_list']
        image_name = ref['image_name']
        image_size = ref['image_size']

        # We only generate pseudo labels for trainval samples
        if split != 'trainval':
            continue

        # Prior knowledge
        if 'sky' in phrase: word_list.append('top')
        if 'ground' in phrase: word_list.append('bottom')
        noun_tokens = extract_nouns(' '.join(word_list))

        # We utilize the class label, attributes and location info
        # to compute similarity between each bounding box and textual phrase.
        # We prepare them here.
        proposal = proposals_dict[image_name]
        proposal_cls_list, proposal_attr_list = get_class_attribute_info(
                vg_class_list, vg_attribute_list, proposal)
        proposal_loc_list = get_location_info(proposal['proposals'], image_size)
        assert len(proposal_cls_list) == len(proposal_attr_list) == len(proposal_loc_list)

        num_proposals = len(proposal_cls_list)
        scores = np.zeros((num_proposals, ), dtype='float32')
        # Start to compute similarity with each proposal
        for n in range(num_proposals):
            tmp_score = 0
            tmp_cls = proposal_cls_list[n]
            tmp_attr = proposal_attr_list[n]
            tmp_loc = proposal_loc_list[n]

            # class label similarity
            if tmp_cls != '__background__':
                if tmp_cls in word_list:
                    tmp_score += 1.0
                else:
                    # compute glove feats similarity
                    if len(noun_tokens) == 0:
                        noun_glove_list = [np.array(glove_dict[t], dtype='float32') for t in word_list
                                           if t in glove_dict]
                    else:
                        noun_glove_list = [np.array(glove_dict[t], dtype='float32') for t in noun_tokens
                                           if t in glove_dict]
                    cls_glove_list = [np.array(glove_dict[t], dtype='float32') for t in tmp_cls.split(' ')]
                    cos_sim_list = [compute_cosine_similarity(cls_glove, noun_glove)
                                    for cls_glove in cls_glove_list
                                    for noun_glove in noun_glove_list]
                    if len(cos_sim_list) != 0:
                        max_cos_sim = max(cos_sim_list)
                        tmp_score += max_cos_sim

            # attributes similarity
            if tmp_attr != '__no_attribute__':
                if tmp_attr in word_list:
                    tmp_score += 1.0

            # location similarity
            if len(tmp_loc) != 0:
                for loc_word in tmp_loc:
                    if loc_word in word_list:
                        tmp_score += 1.0

            scores[n] = tmp_score

        # we choose the bbox with the maximum similarity as pseudo bbox
        pseudo_bbox_index = np.argmax(scores)
        pseudo_bbox = proposal['proposals'][pseudo_bbox_index] 
        pseudo_sim_score = np.max(scores)

        # update ref by adding pseudo info
        ref['pseudo_bbox'] = pseudo_bbox
        ref['pseudo_bbox_index'] = pseudo_bbox_index
        ref['pseudo_sim_score'] = pseudo_sim_score

        if i % 1000 == 0:
            print('Processing {} / {} ...'.format(i, len(refs)))

    # Dump into disk
    with open(os.path.join(data_dir, 'referit_refs_pseudo.pkl'), 'wb') as f:
        pickle.dump(refs, f)# }}}


def compute_pseudo_label_acc(data_dir):# {{{
    """Compute accuracy of the generated pseudo labels.

    For each refexp, we compute the IoU between the groundtruth bbox and the 
    pseudo bounding box, and consider the pseudo label is right if the IoU
    larger than 0.5.
    
    Args:
        data_dir (str): Dir to store the preprocessed data.

    """
    # Load pseudo labels
    with open(os.path.join(data_dir, 'referit_refs_pseudo.pkl'), 'rb') as f:
        refs = pickle.load(f)
    
    print('Start to compute accuracy of pseudo labels.')
    num_correct, num_total = 0., 0.
    for ref in refs:
        split = ref['split']

        # We only compute acc of pseudo labels for trainval samples
        # since we have not generate pseudo labels for test samples.
        if split != 'trainval':
            continue

        pseudo_bbox = ref['pseudo_bbox']
        pseudo_bbox = pseudo_bbox.reshape(1, 4)
        groundtruth_bbox = ref['gt_bbox']
        groundtruth_bbox = np.array(groundtruth_bbox)
        iou = compute_iou(pseudo_bbox, groundtruth_bbox)[0]
        
        if iou > 0.5:
            num_correct += 1
        num_total += 1
    print('There are %d samples in the trainval split, and the accuracy of pseudo labels is %.3f' \
            % (num_total, num_correct / num_total))# }}}


def add_pseudo_bbox_index_to_test(data_dir):# {{{
    # Load pseudo labels
    with open(os.path.join(data_dir, 'referit_refs_pseudo.pkl'), 'rb') as f:
        refs = pickle.load(f)

    # Load proposals
    with open(os.path.join(data_dir, 'referit_vg_det_top30_hit0.796.pkl'), 'rb') as f:
        proposals_dict = pickle.load(f)

    for ref in refs:
        split = ref['split']
        image_name = ref['image_name']
        groundtruth_bbox = ref['gt_bbox']
        groundtruth_bbox = np.array(groundtruth_bbox)

        if split != 'test':
           continue

        proposals = proposals_dict[image_name]['proposals'] 
        ious = compute_iou(proposals, groundtruth_bbox)
        ref['pseudo_bbox_index'] = np.argmax(ious)

    with open(os.path.join(data_dir, 'referit_refs_pseudo.pkl'), 'wb') as f:
        pickle.dump(refs, f)# }}}


def generate_speaker_eval_json_file(data_dir): # {{{
    # Load pseudo labels
    with open(os.path.join(data_dir, 'referit_refs_pseudo.pkl'), 'rb') as f:
        refs = pickle.load(f)

    groundtruth_test = {}
    for ref in refs:
        ann_id = ref['ann_id']
        raw_phrase = ref['raw_phrase']
        if ann_id not in groundtruth_test:
            groundtruth_test[ann_id] = []
        groundtruth_test[ann_id].append(raw_phrase)

    with open(os.path.join(data_dir, 'speaker_groundtruth_test.json'), 'w') as f:
        json.dump(groundtruth_test, f)# }}}


def add_pseudo_positve_all(data_dir):# {{{
    # Load pseudo labels
    with open(os.path.join(data_dir, 'referit_refs_pseudo.pkl'), 'rb') as f:
        refs = pickle.load(f)

    # Load proposals
    with open(os.path.join(data_dir, 'referit_vg_det_top30_hit0.796.pkl'), 'rb') as f:
        proposals_dict = pickle.load(f)

    for ref in refs:
        split = ref['split']
        image_name = ref['image_name']

        if split != 'trainval':
            continue

        pseudo_groundtruth_bbox = ref['pseudo_bbox']
        pseudo_groundtruth_bbox = np.array(pseudo_groundtruth_bbox)
        proposals = proposals_dict[image_name]['proposals'] 

        ious = compute_iou(proposals, pseudo_groundtruth_bbox)
        if np.all(ious < 0.5):
            pseudo_positive_all = []
            raise RuntimeError()
        else:
            pseudo_positive_all = list(np.where(ious >= 0.5)[0])

        ref['pseudo_positive_all'] = pseudo_positive_all

    with open(os.path.join(data_dir, 'referit_refs_pseudo.pkl'), 'wb') as f:
        pickle.dump(refs, f)# }}}


def add_gt_index_to_test(data_dir): # {{{
    # Load pseudo labels
    with open(os.path.join(data_dir, 'referit_refs_pseudo.pkl'), 'rb') as f:
        refs = pickle.load(f)

    # Load proposals
    with open(os.path.join(data_dir, 'referit_vg_det_top30_hit0.796.pkl'), 'rb') as f:
        proposals_dict = pickle.load(f)

    for ref in refs:
        image_name = ref['image_name']

        groundtruth_bbox = ref['gt_bbox']
        groundtruth_bbox = np.array(groundtruth_bbox)
        proposals = proposals_dict[image_name]['proposals'] 

        ious = compute_iou(proposals, groundtruth_bbox)
        if np.all(ious < 0.5):
            gt_index = []
        else:
            gt_index = list(np.where(ious >= 0.5)[0])

        ref['gt_index'] = gt_index

    with open(os.path.join(data_dir, 'referit_refs_pseudo.pkl'), 'wb') as f:
        pickle.dump(refs, f)# }}}


def main(args):
    dataset = args.dataset
    data_dir = os.path.join('data', dataset)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # 1. Load refs
    if not os.path.isfile(os.path.join(data_dir, 'referit_refs.pkl')):
        read_refs(data_dir, args.img_dir)

    # 2. Generate pseudo labels
    if not os.path.isfile(os.path.join(data_dir, 'referit_refs_pseudo.pkl')):
        generate_pseudo_label(data_dir)

    # 3. Compute the accuracy of pseudo labels
    compute_pseudo_label_acc(data_dir)

    # TODO: should add this function to generate_pseudo_label()
    # 4. Add pseudo_bbox_index for test split
    # This is useful for warm-up speaker when evaluate the speaker.
    add_pseudo_bbox_index_to_test(data_dir)

    # 5. Generate groundtruth file for eval speaker
    generate_speaker_eval_json_file(data_dir)

    # TODO: should add this function to generate_pseudo_label()
    # 6. Add pseudo_positive_all for trainval split
    # This is useful for warm-up listener.
    add_pseudo_positve_all(data_dir)

    # TODO: should add this function to generate_pseudo_label()
    # 7. Add groundtruth index for test split
    # This is useful for warm-up listener when evaluate the listener
    add_gt_index_to_test(data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='referit',
            help='The dataset to be preprocessed.')
    parser.add_argument('--img_dir', type=str, default=None,
            help='Dir of the image.')
    
    args = parser.parse_args()

    # Call main
    main(args)
