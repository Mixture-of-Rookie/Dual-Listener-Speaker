import sys
import json
import pdb
import traceback
from bdb import BdbQuit

from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.eval import EvalCap

def evaluate(annotation_file, result_file):
    # tokenizer = StanfordTokenizer()     # for Chinese
    tokenizer = PTBTokenizer   # for English
    annos = json.load(open(annotation_file, 'r'))
    rests = json.load(open(result_file, 'r'))
    eval_cap = EvalCap(annos, rests, tokenizer)  # , use_scorers=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE'])

    eval_cap.evaluate()

    all_score = {}
    for metric, score in eval_cap.eval.items():
        # print('%s: %.1f' % (metric, score*100))
        all_score[metric] = score

    return all_score


def compute_bleu_score(annotations, results):
    tokenizer = PTBTokenizer
    eval_cap = EvalCap(annotations, results, tokenizer)
    eval_cap.evaluate()

    all_score = {}
    for metric, score in eval_cap.eval.items():
        all_score[metric] = score
    return all_score


def main(annotation_file, result_file):
    annotation_file = annotation_file
    result_file = result_file
    evaluate(annotation_file, result_file)


if __name__ == '__main__':
    try:
        main()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
