"""
encoding=utf-8

"""

import argparse
import json
import numpy as np
from collections import defaultdict
from evaluate_metric import *


def results_three_components(path):
    predictions = json.load(open(path))['scores']

    metrics = ["node_f1", "node_AC", "step_f1", "step_AC", "triples_f1", "triples_AC",
               "degree_acc", "degree_AC", "degree_macro_f1",
               "Overall",
               "conclusion_step_f1", "conclusion_step_AC",
               "rebuttal_step_f1", "rebuttal_step_AC",
               ]

    s = ""
    res_three_comps = {}
    for metr in metrics:
        res_three_comps[metr] = predictions[metr]
        s += f"{100 * predictions[metr]:.1f}\t "

    print('\t'.join(metrics))
    print(s)

    return res_three_comps


def eval_triples_by_operator(path):
    predictions = json.load(open(path))['collector']
    predictions = list(predictions.values())

    compares_collector = defaultdict(list)
    score_collector = defaultdict(list)
    for item in predictions:
        pred_triples_dict = item['pred_item']['triples_dict']
        gold_triples_dict = item['gold_item']['triples_dict']
        
        for sent_id, gold_triples in gold_triples_dict.items():
            pred_triples = pred_triples_dict.get(sent_id, [])
            eval_score = eval_inner_triples_(pred_triples, gold_triples)
            if len(gold_triples) == 0:
                compares_collector['none'].append((pred_triples, gold_triples))
                score_collector['none'].append(eval_score)
            else:
                gold_operators = []
                for t in gold_triples:
                    gold_operators += t[0] + [t[2]] + t[3]

                for opt in set(gold_operators):
                    compares_collector[opt].append((pred_triples, gold_triples))
                    score_collector[opt].append(eval_score)
    s = ""
    triples_f1_by_operator = {}
    for k in ['[I-IMPLICATION]', '[I-CONJUNCTION]', '[I-DISJUNCTION]', '[BOX]', '[DIAMOND]', '[NEG]','none']:
        average_score = average_metric(score_collector[k])
        s += f"{100*average_score['triples_f1']:.1f}\t"
        triples_f1_by_operator[k] = average_score

    print('\n')
    print('\t'.join(['[I-IMPLICATION]', '[I-CONJUNCTION]', '[I-DISJUNCTION]', '[BOX]', '[DIAMOND]', '[NEG]','none']))
    print(s)
    
    return triples_f1_by_operator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show metrics.')
    parser.add_argument("--prediction_path", type=str)
    args = parser.parse_args()

    res_comp = results_three_components(args.prediction_path)
    results = eval_triples_by_operator(args.prediction_path)
