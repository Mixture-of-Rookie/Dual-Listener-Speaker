#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import pickle
import argparse
import skimage.io
import numpy as np
from multiprocessing import Pool

# utils
from utils.utils import (
    compute_iou,
    unify_boxes,
    load_file,
    load_vg_info,
    load_glove_feats,
    extract_nouns,
    compute_cosine_similarity,
)
from utils.flickr30k_entities_utils import (
    get_annotations,
    get_sentence_data,
)

Flickr30k_Dir = '/mnt/disk250G/Datasets/Flickr30K'
Flickr30k_Image_Dir = os.path.join(Flickr30k_Dir, 'flickr30k_images')
Flickr30k_Entity_Dir = os.path.join(Flickr30k_Dir, 'flickr30k_entities')


def build_vocab(data_dir):# {{{
    """Build vocabulary for flickr30k dataset.

    Load phrase from the flickr30k sentences dir and then build vocabulary.

    Args:
        data_dir (str): Dir to dump the vocabulary.

    Returns:
        ~(dict, dict):

        * **word2idx**: Map word to index.
        * **idx2word**: Map index to word.

    """
    sen_dir = os.path.join(Flickr30k_Entity_Dir, 'Sentences')
    assert os.path.isdir(sen_dir), \
        'Please make sure flickr30k sentences dir is at {}'.format(sen_dir)
    print('Build vocabulary from: %s.' % sen_dir)

    vocab = ['<PAD>', '<unk>']
    vocab_count = {}
    for sen_file in os.listdir(sen_dir):
        if not sen_file.endswith('.txt'):
            continue

        sens = get_sentence_data(os.path.join(sen_dir, sen_file))
        for sen in sens:
            for phrase in sen['phrases']:
                for word in phrase['phrase'].lower().split(' '):
                    if word not in vocab_count:
                        vocab_count[word] = 0
                    vocab_count[word] = vocab_count[word] + 1

    vocab += [wd for wd, n in vocab_count.items() if n > 5]
    vocab.append('<EOS>')
    print('There are %d words in the vocabulary.' % len(vocab))

    word2idx = {word: idx for (idx, word) in enumerate(vocab)}
    idx2word = {idx: word for (idx, word) in enumerate(vocab)}

    with open(os.path.join(data_dir, 'flickr30k_vocab.pkl'), 'wb') as f:
        pickle.dump((word2idx, idx2word), f)

    return word2idx, idx2word# }}}


def build_groundtruth():# {{{
    """Generate groundtruth region for each phrase.

    Note that there exists phrases that correspond to multiple boxes, for these
    case, we unify these boxes to generate a groundtruth box.

    Returns:
        ~Dict:

        * **groundtruth_dict**: Map phrase id to its groundtruth region.

    """
    groundtruth_dict = {}

    ann_dir = os.path.join(Flickr30k_Entity_Dir, 'Annotations')
    assert os.path.isdir(ann_dir), \
        'Please make sure flickr30k annotations dir is at {}.'.format(ann_dir)

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith('.xml'):
            continue

        img_id = ann_file[:-4]
        ann = get_annotations(os.path.join(ann_dir, ann_file))
        for phrase_id, boxes in ann['boxes'].items():
            if phrase_id == '0':  # No object
                continue

            if len(boxes) > 1:
                groundtruth_dict[phrase_id] = unify_boxes(boxes)
            else:
                groundtruth_dict[phrase_id] = boxes[0]

    return groundtruth_dict# }}}


def encode_phrase(phrase, word2idx, MAX_WORDS=10):# {{{
    """Convert phrase into number tokens using the given word2idx.

    Args:
        phrase (str): Raw textual phrase.
        word2idx (dict): A dictionary mapping word to its index in the vocabulary.
        MAX_WORDS (int): Phrase longer than this value will be trimed.

    Returns:
        ~(List, List):

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


def read_refs(data_dir):# {{{
    """Load flickr30k dataset from annotation files.

    Load bounding boxes and their describing textual phrases from annotation files.
    We re-organize the data into a python list that each element represents a training sample.
    Each element is a python dictionary contains the following key-value pairs.

    * refexp_id (int): The unique id for the textual phrase (e.g., 0).
    * ann_id (str): The id of the bounding box (e.g., ).
    * image_name (str): The name of the image (e.g., ).
    * image_size (tuple): The width and height of the image (e.g., (519, 304)).
    * raw_phrase (str): The textual phrase of the object.
    * word_list (list): Encoded tokens of the textual phrase, i.e., convert
        the textual phrase into number list using vocabulary (e.g., [0, 3, 5, 6]).
    * gt_bbox (list): The coordinates of the bounding box with the format
        of [x1, y1, x2, y2]. Here (x1, y1) and (x2, y2) are the top-left and
        bottom-right coordinates (e.g., [0., 0., 124., 131.]).
    * split (str): This sample belongs to which split? train, val or test (e.g., train).

    Args:
        data_dir (str): Dir to store the preprocessed data.

    """
    ann_dir = os.path.join(Flickr30k_Entity_Dir, 'Annotations')
    sen_dir = os.path.join(Flickr30k_Entity_Dir, 'Sentences')
    assert os.path.isdir(ann_dir), \
        'Please make sure flickr30k annotations dir is at {}.'.format(ann_dir)
    print('Load dataset from: %s.' % ann_dir)

    # Load trainval/test splits
    trainval_file = os.path.join(data_dir, 'flickr30k_train_val.lst')
    trainval_split = load_file(trainval_file)
    test_file = os.path.join(data_dir, 'flickr30k_test.lst')
    test_split = load_file(test_file)

    # Build vocabulary 
    word2idx, _ = build_vocab(data_dir)

    # Unify groundtruth bounding boxes
    groundtruth_dict = build_groundtruth()

    refs = []
    refexp_id = 0
    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith('.xml'):
            continue

        img_id = ann_file[:-4]
        ann = get_annotations(os.path.join(ann_dir, ann_file))
        sens = get_sentence_data(os.path.join(sen_dir, img_id + '.txt'))

        # Read image to get width and height
        image_name = img_id + '.jpg'
        im = skimage.io.imread(os.path.join(Flickr30k_Image_Dir, image_name))
        width, height = im.shape[1], im.shape[0]

        for sen in sens:
            for phrase in sen['phrases']:
                phrase_id = phrase['phrase_id']
                raw_phrase = phrase['phrase']

                # Check if this phrase has a box assigned
                if phrase_id not in list(ann['boxes'].keys()) or phrase_id == '0':
                    continue

                # Convert the phrase into tokens
                word_list, encoded_tokens = encode_phrase(raw_phrase, word2idx)

                gt_bbox = groundtruth_dict[phrase_id]

                if img_id in trainval_split:
                    split = 'trainval'
                elif img_id in test_split:
                    split = 'test'
                else:
                    raise ValueError

                refs += [{'refexp_id': refexp_id,
                          'ann_id': phrase_id,
                          'image_name': image_name,
                          'image_size': (width, height),
                          'raw_phrase': raw_phrase,
                          'word_list': word_list,
                          'tokens': encoded_tokens,
                          'gt_bbox': gt_bbox,
                          'split': split}]
                refexp_id += 1

    print('Found %d refs in the flickr30k dataset.' % len(refs))

    # Dump into flickr30k_refs.json
    with open(os.path.join(data_dir, 'flickr30k_refs.pkl'), 'wb') as f:
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


def process_worker(arg_list):# {{{
    # args
    refexp_id, word_list, proposals, \
        proposal_cls_list, proposal_attr_list, proposal_loc_list = arg_list

    noun_tokens = extract_nouns(' '.join(word_list))

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

    # We choose the bbox with the maximum similarity as pseudo bbox
    pseudo_bbox_index = np.argmax(scores)
    pseudo_bbox = proposals[pseudo_bbox_index]
    pseudo_sim_score = np.max(scores)

    f = open('tmp_flickr30k_pseduo_log.txt', 'a+')
    f.write(refexp_id)
    f.close()

    return_dict = {
        'refexp_id': refexp_id,
        'pseudo_bbox': pseudo_bbox,
        'pseudo_bbox_index': pseudo_bbox_index,
        'pseudo_sim_score': pseudo_sim_score,
    }
    return return_dict# }}}


def generate_pseudo_label_v2(data_dir):# {{{
    """Generate pseudo label for each ref in the flickr30k dataset.

    Assign a pseudo bounding box for each textual phrase. We compute similarities
    between textual phrases and bounding boxes, and select the most similar one as
    the pseudo bounding box. When compute the similarity, we consider three types
    info: class_label, attribute and location.

    Args:
        data_dir (str): Dir to store the preprocessed data.

    """
    # Load flickr30k dataset
    with open(os.path.join(data_dir, 'flickr30k_refs.pkl'), 'rb') as f:
        refs = pickle.load(f)

    # Load class labels and attributes of vg dataset
    vg_class_list, vg_attribute_list = load_vg_info(os.path.join('data', 'vg'))

    # Load glove vector
    glove_file = '/mnt/disk2T/zhiyuwang/Cache/Dual-learning-listener-speaker/glove.840B.300d.txt'
    glove_dict = load_glove_feats(glove_file)

    # Load proposals
    with open(os.path.join(data_dir, 'flickr30k_vg_det_top30_hit0.836.pkl'), 'rb') as f:
        proposals_dict = pickle.load(f)

    print('Start to generate pseudo labels for refs.')
    for i, ref in enumerate(refs):
        split = ref['split']
        phrase = ref['raw_phrase']
        word_list = ref['word_list']
        refexp_id = ref['refexp_id']
        image_name = ref['image_name']
        image_size = ref['image_size']

        # We only generate pseudo laels for trainval samples
        if split != 'trainval':
            continue

        # Prior knowledge
        if 'sky' in phrase: word_list.append('top')
        if 'ground' in phrase: word_list.append('bottom')
        noun_tokens = []

        # We utilize the class label, attributes and location_info
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
    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'wb') as f:
        pickle.dump(refs, f)# }}}


def generate_pseudo_label(data_dir):# {{{
    """Generate pseudo label for each ref in the flickr30k dataset.

    Assign a pseudo bounding box for each textual phrase. We compute similarities
    between textual phrases and bounding boxes, and select the most similar one as
    the pseudo bounding box. When compute the similarity, we consider three types
    info: class_label, attribute and location.

    Args:
        data_dir (str): Dir to store the preprocessed data.

    """
    # Load flickr30k dataset
    with open(os.path.join(data_dir, 'flickr30k_refs.pkl'), 'rb') as f:
        refs = pickle.load(f)

    # Load class labels and attributes of vg dataset
    vg_class_list, vg_attribute_list = load_vg_info(os.path.join('data', 'vg'))

    # Load glove vector
    glove_file = '/mnt/disk2T/zhiyuwang/Cache/Dual-learning-listener-speaker/glove.840B.300d.txt'
    global glove_dict
    glove_dict = load_glove_feats(glove_file)

    # Load proposals
    with open(os.path.join(data_dir, 'flickr30k_vg_det_top30_hit0.836.pkl'), 'rb') as f:
        proposals_dict = pickle.load(f)

    args_list = []
    for ref in refs:
        split = ref['split']
        phrase = ref['raw_phrase']
        word_list = ref['word_list']
        refexp_id = ref['refexp_id']
        image_name = ref['image_name']
        image_size = ref['image_size']

        # We only generate pseudo laels for trainval samples
        if split != 'trainval':
            continue

        # Prior knowledge
        if 'sky' in phrase: word_list.append('top')
        if 'ground' in phrase: word_list.append('bottom')

        # We utilize the class label, attributes and location_info
        # to compute similarity between each bounding box and textual phrase.
        # We prepare them here.
        proposal = proposals_dict[image_name]
        proposal_cls_list, proposal_attr_list = get_class_attribute_info(
                vg_class_list, vg_attribute_list, proposal)
        proposal_loc_list = get_location_info(proposal['proposals'], image_size)
        assert len(proposal_cls_list) == len(proposal_attr_list) == len(proposal_loc_list)

        args_list.append([refexp_id, word_list, glove_dict, proposal['proposals'],
            proposal_cls_list, proposal_attr_list, proposal_loc_list])

    print('Start to generate pseudo labels.')
    print('There are %d jobs to be processed' % len(args_list))
    pool = Pool(processes=16)
    result_dict_list = pool.map(process_worker, args_list)
    assert len(result_dict_list) == len(args_list)
    # Construct mapping from refexp_id to result_dict
    pseudo_label_mapping = {}
    for result_dict in result_dict_list:
        refexp_id = result_dict['refexp_id']
        pseudo_label_mapping[refexp_id] = result_dict

    # Update refs by adding pseudo info
    for ref in refs:
        refexp_id = ref['refexp_id']
        result_dict = pseudo_label_mapping[refexp_id]
        ref['pseudo_bbox'] = result_dict['pseudo_bbox']
        ref['pseudo_bbox_index'] = result_dict['pseudo_bbox_index']
        ref['pseudo_sim_score'] = result_dict['pseudo_sim_score']

    # Dump into disk
    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'wb') as f:
        pickle.dump(refs, f)# }}}


def compute_pseudo_label_acc(data_dir):# {{{
    """Compute accuracy of the generated pseudo labels.

    For each phrase, we compute the IoU between the groundtruth bbox and the
    pseudo bounding box, and consider the pseudo label is right if the IoU 
    larger than 0.5.

    Args:
        data_dir (str): Dir to store the preprocessed data.

    """
    # Load pseudo labels
    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'rb') as f:
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
            % (num_total, num_correct / num_total))# }}}}}}


def add_pseudo_bbox_index_to_test(data_dir):# {{{
    # Load pseudo labels
    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'rb') as f:
        refs = pickle.load(f)

    # Load proposals
    with open(os.path.join(data_dir, 'flickr30k_vg_det_top30_hit0.836.pkl'), 'rb') as f:
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

    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'wb') as f:
        pickle.dump(refs, f)# }}}


def generate_vg_det_proposals_pickle_file(data_dir):# {{{
    # Load flickr30k dataset
    with open(os.path.join(data_dir, 'flickr30k_refs.pkl'), 'rb') as f:
        refs = pickle.load(f)

    VG_DET_DIR = '/mnt/disk2T/zhiyuwang/Cache/Dual-learning-listener-speaker/flickr30k/flickr30k_vg_det_top30'
    VG_DET_CLASS_ATTR_DIR = '/mnt/disk2T/zhiyuwang/Cache/Dual-learning-listener-speaker/flickr30k/vg_det_class_attr_label'

    num_train_hit, num_train_total = 0, 0
    num_test_hit, num_test_total = 0, 0
    proposals_dict = {}
    for i, ref in enumerate(refs):
        image_name = ref['image_name']
        gt_bbox = ref['gt_bbox']
        split = ref['split']
        
        proposals = np.loadtxt(os.path.join(VG_DET_DIR, image_name[:-4] + '.txt'))
        class_attr_labels = np.load(os.path.join(VG_DET_CLASS_ATTR_DIR, image_name[:-4] + '.npz'))

        class_labels = class_attr_labels['cls_pred']
        class_labels = np.argmax(class_labels, axis=1)  # (num_proposals, )
        attr_labels = class_attr_labels['attr_pred']
        attr_labels = np.argmax(attr_labels, axis=1)  # (num_proposals, )

        assert proposals.shape[0] == class_labels.shape[0] == attr_labels.shape[0] == 30

        if image_name not in proposals_dict:
            proposals_dict[image_name] = {'proposals': proposals,
                                          'classes': class_labels,
                                          'attributes': attr_labels}

        IoUs = compute_iou(proposals, np.array(gt_bbox))

        if split == 'trainval':
            if not np.all(IoUs < 0.5):
                num_train_hit += 1
            num_train_total += 1
        elif split == 'test':
            if not np.all(IoUs < 0.5):
                num_test_hit += 1
            num_test_total += 1
        else:
            raise ValueError

        if i % 1000 == 0:
            print('Processing %d/%d ...' % (i, len(refs)))
    
    with open(os.path.join(data_dir, 'flickr30k_vg_det_top30_hit0.836.pkl'), 'wb') as f:
        pickle.dump(proposals_dict, f)

    print('There are %d phrases in the trainval split, and recall is %.3f.' % (num_train_total, num_train_hit / num_train_total))
    print('There are %d phrases in the test split, and recall is %.3f.' % (num_test_total, num_test_hit / num_test_total))# }}}


def generate_speaker_eval_json_file(data_dir): # {{{
    # Load pseudo labels
    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'rb') as f:
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
    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'rb') as f:
        refs = pickle.load(f)

    # Load proposals
    with open(os.path.join(data_dir, 'flickr30k_vg_det_top30_hit0.836.pkl'), 'rb') as f:
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

    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'wb') as f:
        pickle.dump(refs, f)# }}}


def add_gt_index_to_test(data_dir): # {{{
    # Load pseudo labels
    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'rb') as f:
        refs = pickle.load(f)

    # Load proposals
    with open(os.path.join(data_dir, 'flickr30k_vg_det_top30_hit0.836.pkl'), 'rb') as f:
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

    with open(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl'), 'wb') as f:
        pickle.dump(refs, f) # }}}


def main(args):
    dataset = args.dataset
    data_dir = os.path.join('data', dataset)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # 1. Load refs
    if not os.path.isfile(os.path.join(data_dir, 'flickr30k_refs.pkl')):
        read_refs(data_dir)

    # 2. Generate pseudo labels
    if not os.path.isfile(os.path.join(data_dir, 'flickr30k_refs_pseudo.pkl')):
        generate_pseudo_label_v2(data_dir)

    # 3. Compute the accuracy of pseudo labels
    compute_pseudo_label_acc(data_dir)

    # TODO: should add this function to generate_pseudo_label()
    # 4. Add pseudo_bbox_index for test split
    # This is useful for warm-up speaker when evaluate the speaker.
    # add_pseudo_bbox_index_to_test(data_dir)

    # 5. Generate groundtruth file for eval speaker
    # generate_speaker_eval_json_file(data_dir)

    # TODO: should add this function to generate_pseudo_label()
    # 6. Add pseudo_positive_all for trainval split
    # This is useful for warm-up listener.
    # add_pseudo_positve_all(data_dir)

    # TODO: should add this function to generate_pseudo_label()
    # 7. Add groundtruth index for test split
    # This is useful for warm-up listener when evaluate the listener
    # add_gt_index_to_test(data_dir)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr30k',
            help='The dataset to be preprocessed.')

    args = parser.parse_args()

    # Call main
    main(args)
