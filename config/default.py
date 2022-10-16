# /usr/bin/env python
# -*- coding: utf-8 -*-

from yacs.config import CfgNode as CN

# Create a Node
__C = CN()

# ========================== INPUT =========================
__C.INPUT = CN()
__C.INPUT.MAX_SEQ_LEN = 20
# Feature type: ['fc7', 'pool5']
__C.INPUT.FEATURE_TYPE = 'pool5'

# ========================== DATASET =========================
__C.DATASET = CN()
__C.DATASET.NAME = 'referit'
__C.DATASET.VOCAB_SIZE = 8802
__C.DATASET.DATA_PICKLE = 'data/referit/referit_refs_pseudo.pkl'
__C.DATASET.VOCAB_PICKLE = 'data/referit/referit_vocab.pkl'
__C.DATASET.EVAL_SPEAKER_JSON = 'data/referit/speaker_groundtruth_test.json'
__C.DATASET.TRAIN = 'trainval'
__C.DATASET.DEV = 'test'
__C.DATASET.TEST = 'test'

# ========================== OUPUT =========================
__C.OUTPUT = CN()
__C.OUTPUT.SAVE_NAME = ''
# Save checkpoint frequency (epochs)
__C.OUTPUT.SAVE_FREQ = 5
__C.OUTPUT.CHECKPOINT_DIR = './exp/speaker'

# ========================== SPEAKER =========================
__C.SPEAKER = CN()
# Speaker is a encoder-decoder framework
__C.SPEAKER.ENCODER = CN()
__C.SPEAKER.ENCODER.DROPOUT = 0.25
__C.SPEAKER.ENCODER.LOC_DIM = 8
__C.SPEAKER.ENCODER.HIDDEN_DIM = 1024
__C.SPEAKER.ENCODER.OUTPUT_DIM = 1024
__C.SPEAKER.DECODER = CN()
# Word embedding dim
__C.SPEAKER.DECODER.EMBD_DIM = 512
__C.SPEAKER.DECODER.EMBD_DROPOUT = 0.5
__C.SPEAKER.DECODER.LSTM_DIM = 512
__C.SPEAKER.DECODER.LSTM_DROPOUT = 0.5

# ========================== LISTENER =========================
__C.LISTENER = CN()
__C.LISTENER.NORMALIZE = False
# Listener consists of the following two parts:
# visual encoder and language encoder
__C.LISTENER.VIS_ENC = CN()
__C.LISTENER.VIS_ENC.DROPOUT = 0.25
__C.LISTENER.VIS_ENC.LOC_DIM = 8
__C.LISTENER.VIS_ENC.HIDDEN_DIM = 1024
__C.LISTENER.VIS_ENC.OUTPUT_DIM = 1024
__C.LISTENER.LAN_ENC = CN()
# Word embedding dim
__C.LISTENER.LAN_ENC.EMBD_DIM = 512
__C.LISTENER.LAN_ENC.LSTM_DIM = 512

# ========================== OPTIMIZATION =========================
__C.OPTIMIZATION = CN()
__C.OPTIMIZATION.EPOCHS = 30
__C.OPTIMIZATION.BATCH_SIZE = 64
__C.OPTIMIZATION.SPEAKER_LR = 4e-4
__C.OPTIMIZATION.LISTENER_LR = 1e-4
# Clip gradients at this value
__C.OPTIMIZATION.CLIP_MAX_NORM = 0.1
# LR decay frequency (epochs)
__C.OPTIMIZATION.LR_DECAY_FREQ = 3
# Optimize speaker? i.e. perform region --> phrase --> region.
__C.OPTIMIZATION.OPTIMIZE_SPEAKER = True
# Optimize listener? i.e. perform phrase --> region --> phrase.
__C.OPTIMIZATION.OPTIMIZE_LISTENER = False
# Sampling topK regions (phrase --> region)
__C.OPTIMIZATION.TOPK = 5
# Threshold of reward
__C.OPTIMIZATION.REWARD_THRESHOLD = 0.9

# ========================== MONITOR =========================
__C.MONITOR = CN()
# Log frequency (steps)
__C.MONITOR.PRINT_FREQ =10
# Evaluation frequency (epochs)
__C.MONITOR.EVAL_FREQ = 1

# ========================== PRETRAINED =========================
__C.PRETRAINED = CN()
__C.PRETRAINED.SPEAKER_CHECKPOINT_PATH = ''
__C.PRETRAINED.LISTENER_CHECKPOINT_PATH = ''

# ========================== MISC =========================
__C.MISC = CN()
__C.MISC.SEED = 123

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return __C.clone()
