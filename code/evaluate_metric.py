from tree_utils import *
import sklearn
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import numpy as np


def div(num, denom):
    return num / denom if denom > 0 else 0

def compute_f1(matched, predicted, gold):
    # 0/0=1; x/0=0
    precision = div(matched, predicted)
    recall = div(matched, gold)
    f1 = div(2 * precision * recall, precision + recall)
    
    if predicted == gold == 0:
        precision = recall = f1 = 1.0
    
    return precision, recall, f1

def average_metric(scores):
    average_scores = {}
    for k in scores[0].keys():
        try:
            average_scores[k] = sum([s[k] for s in scores]) / len(scores)
        except:
            continue
    return average_scores


# ----- proof -----
def eval_proof(pred_proof, gold_proof, spliter = '[STEP_SPLITTER]', flag = '$tree$'):
    pred_steps = parse_proof(pred_proof, spliter=spliter, single_premise=True, flag = flag)
    gold_steps = parse_proof(gold_proof, spliter=spliter, single_premise=True, flag = flag)

    result = eval_proof_(pred_steps, gold_steps)
    result['compare_proof'] = (pred_proof, gold_proof)

    return result


def eval_proof_(pred_steps, gold_steps):

    # Node
    gold_nodes = []
    for step in gold_steps:
        gold_nodes += step['pre']
        gold_nodes.append(step['con'])

    pred_nodes = []
    for step in pred_steps:
        pred_nodes += step['pre']
        pred_nodes.append(step['con'])

    pred_nodes, gold_nodes = set(pred_nodes), set(gold_nodes)
    node_match = len([node for node in gold_nodes if node in pred_nodes])
    node_p, node_r, node_f1 = compute_f1(node_match, len(pred_nodes), len(gold_nodes))


    # Steps
    gold_steps_list = [(sorted(step['pre']), step['type'], step['con'],) for step in gold_steps]
    pred_steps_list = [(sorted(step['pre']), step['type'], step['con'],) for step in pred_steps]

    step_match = len([step for step in gold_steps_list if step in pred_steps_list])
    step_p, step_r, step_f1 = compute_f1(step_match, len(pred_steps_list), len(gold_steps_list))

    # Conclusion Steps
    gold_conclusion_steps_list = [(sorted(step['pre']), step['type'], step['con'],) for step in gold_steps if step['type'] == '->']
    pred_conclusion_steps_list = [(sorted(step['pre']), step['type'], step['con'],) for step in pred_steps if step['type'] == '->']

    conclusion_step_match = len([step for step in gold_conclusion_steps_list if step in pred_conclusion_steps_list])
    conclusion_step_p, conclusion_step_r, conclusion_step_f1 = compute_f1(conclusion_step_match, 
                                                                          len(pred_conclusion_steps_list), len(gold_conclusion_steps_list))


    # Rebuttal Steps
    gold_rebuttal_steps_list = [(sorted(step['pre']), step['type'], step['con'],) for step in gold_steps if step['type'] == '=>']
    pred_rebuttal_steps_list = [(sorted(step['pre']), step['type'], step['con'],) for step in pred_steps if step['type'] == '=>']

    rebuttal_step_match = len([step for step in gold_rebuttal_steps_list if step in pred_rebuttal_steps_list])
    rebuttal_step_p, rebuttal_step_r, rebuttal_step_f1 = compute_f1(rebuttal_step_match, 
                                                                    len(pred_rebuttal_steps_list), len(gold_rebuttal_steps_list))

    result = {
        'node_p':node_p,
        'node_r':node_r,
        'node_f1':node_f1,
        'node_AC': float(node_f1 == 1.0),
        
        'step_p':step_p,
        'step_r':step_r,
        'step_f1':step_f1,
        'step_AC':float(step_f1 == 1.0),
        
        'conclusion_step_p':conclusion_step_p,
        'conclusion_step_r':conclusion_step_r,
        'conclusion_step_f1':conclusion_step_f1,
        'conclusion_step_AC':float(conclusion_step_f1 == 1.0),
        
        
        'rebuttal_step_p':rebuttal_step_p,
        'rebuttal_step_r':rebuttal_step_r,
        'rebuttal_step_f1':rebuttal_step_f1,
        'rebuttal_step_AC':float(rebuttal_step_f1 == 1.0),
        
        'compare_node': (list(pred_nodes), list(gold_nodes)),
        'compare_step': (list(pred_steps_list), list(gold_steps_list)),
    }
    
    return result

# ----- inner triple -----
def compare_inner_triple(triple1, triple2):
    # triple1 = [pre_opt, pre_var, relation, con_opt, con_var]
    
    # reduce operators
    triple1[0], triple1[3] = operator_reduction(triple1[0]), operator_reduction(triple1[3])
    triple2[0], triple2[3] = operator_reduction(triple2[0]), operator_reduction(triple2[3])
    
    if triple2[2] in ['[I-AND]', '[I-OR]']:
        # symmetric relation
        if (triple1 == triple2):
            return True
        if (triple1 == [triple2[3],triple2[4],triple2[2],triple2[0],triple2[1]]):
            return True
        
    else:
        # asymmetric relation
        if triple1 == triple2:
            return True
    
    return False

def eval_inner_triples(pred_triples_str, gold_triples_str, spliter = ';', flag = '$formulae$'):
    
    pred_triples_str = transform_by_dict(pred_triples_str, token2operator)
    pred_triples = parse_inner_triples(pred_triples_str, spliter = spliter, flag = flag)

    gold_triples_str = transform_by_dict(gold_triples_str, token2operator)
    gold_triples = parse_inner_triples(gold_triples_str, spliter = spliter, flag = flag)

    result = eval_inner_triples_(pred_triples, gold_triples)
    result['compare_triples_str'] = (pred_triples_str,gold_triples_str)

    return result

def eval_inner_triples_(pred_triples, gold_triples):
    triples_match = 0
    for gold_t in gold_triples:
        if any([compare_inner_triple(pred_t, gold_t) for pred_t in pred_triples]):
            triples_match += 1

    triples_p, triples_r, triples_f1 = compute_f1(triples_match, len(pred_triples), len(gold_triples))
    
    result = {
        'triples_p':triples_p,
        'triples_r':triples_r,
        'triples_f1':triples_f1,
        'triples_AC':float(triples_f1 == 1.0),
        
        'compare_triples': (pred_triples,gold_triples),
    }
    
    return result


# ----- degree -----
def eval_degree(pred_degree_str, gold_degree_str, flag = '$degree$'):

    pred_degree_str = transform_by_dict(pred_degree_str, token2degree)
    pred_degree = parse_degree(pred_degree_str, flag=flag)

    gold_degree_str = transform_by_dict(gold_degree_str, token2degree)
    gold_degree = parse_degree(gold_degree_str, flag=flag)

    result = eval_degree_(pred_degree, gold_degree)
    result['compare_degree_str'] = (pred_degree_str,gold_degree_str)

    return result

def eval_degree_(pred_degree, gold_degree):
    degree_acc = int(pred_degree == gold_degree)

    result = {
        'degree_acc': degree_acc,
        'compare_degree': (pred_degree,gold_degree),
    }

    return result


# ----- sample -----
def eval_sample_(pred_item, gold_item):

    # eval proof
    proof_result = eval_proof_(pred_item['proof'], gold_item['proof'])

    # eval triples
    triple_results = []
    for sent_id, gold_triples in gold_item['triples_dict'].items():
        pred_triples = pred_item['triples_dict'].get(sent_id, "")
        triple_results.append(eval_inner_triples_(pred_triples, gold_triples))

    triple_result = {}
    for k,v in triple_results[0].items():
        try:
            triple_result[k] = sum([r[k] for r in triple_results]) / len(triple_results)
        except:
            continue

    # eval degree
    degree_results = []
    for sent_id, gold_degree in gold_item['degree_dict'].items():
        pred_degree = pred_item['degree_dict'].get(sent_id, -1)
        degree_results.append(eval_degree_(pred_degree, gold_degree))

    all_preds = [s['compare_degree'][0] for s in degree_results]
    all_golds = [s['compare_degree'][1] for s in degree_results]

    degree_acc = accuracy_score(y_true=all_golds, y_pred=all_preds)
    # degree_acc_balanced = balanced_accuracy_score(y_true=all_golds, y_pred=all_preds)
    degree_result = {
        'degree_acc': degree_acc,
        # 'degree_acc_balanced': degree_acc_balanced,
        'degree_AC': int(degree_acc == 1.0),
        'compare_degree': (all_preds, all_golds)
    }

    result = {}
    result.update(proof_result)
    result.update(triple_result)
    result.update(degree_result)

    result['Overall'] = result['step_AC']*result['triples_AC']*result['degree_AC']

    return result

def eval_sample(pred_item, gold_item):

    if 'proof' not in pred_item:
        pred_item['proof'] = parse_proof(pred_item['proof_str'], spliter=';', single_premise=True, flag = '')
    if 'proof' not in gold_item:
        gold_item['proof'] = parse_proof(gold_item['proof_str'], spliter=';', single_premise=True, flag = '')

    if 'triples_dict' not in pred_item:
        triples_dict = {}
        for sent_id, triples_str in pred_item['triples_str_dict'].items():
            triples_dict[sent_id] = parse_inner_triples(transform_by_dict(triples_str, token2operator), spliter = ';', flag = '')
        pred_item['triples_dict'] = triples_dict
    if 'triples_dict' not in gold_item:
        triples_dict = {}
        for sent_id, triples_str in gold_item['triples_str_dict'].items():
            triples_dict[sent_id] = parse_inner_triples(transform_by_dict(triples_str, token2operator), spliter = ';', flag = '')
        gold_item['triples_dict'] = triples_dict

    if 'degree_dict' not in pred_item:
        degree_dict = {}
        for sent_id, degree_str in pred_item['degree_str_dict'].items():
            degree_dict[sent_id] = parse_degree(transform_by_dict(degree_str, token2degree), flag = '')
        pred_item['degree_dict'] = degree_dict
    if 'degree_dict' not in gold_item:
        degree_dict = {}
        for sent_id, degree_str in gold_item['degree_str_dict'].items():
            degree_dict[sent_id] = parse_degree(transform_by_dict(degree_str, token2degree), flag = '')
        gold_item['degree_dict'] = degree_dict
    
    return eval_sample_(pred_item, gold_item)

def eval_once(pred_item, gold_item):
    return eval_sample(pred_item, gold_item)

    
    
def post_metrics(collector):
    # collector: [{'pred_item':{}, 'gold_item':{}, 'score':{}},..]

    pred_degree_all = []
    gold_degree_all = []

    scores = [sample_eval_result['score'] for sample_eval_result in collector.values()]
    for s in scores:
        pred_degree_all += s['compare_degree'][0]
        gold_degree_all += s['compare_degree'][1]    

    print(pred_degree_all)
    print(gold_degree_all)

    degree_macro_f1 = sklearn.metrics.f1_score(y_true=gold_degree_all, y_pred=pred_degree_all, average='macro')
    degree_micro_f1 = sklearn.metrics.f1_score(y_true=gold_degree_all, y_pred=pred_degree_all, average='micro')



    num_head_relation_match_step = {'=>': 0, '->': 0}
    num_head_relation_tail_match_step = {'=>': 0, '->': 0}

    for sample_eval_result in collector.values():
        pred_steps = sample_eval_result['pred_item']['proof']
        gold_steps = sample_eval_result['gold_item']['proof']

        for g in gold_steps:
            for p in pred_steps:
                if g['pre'] == p['pre'] and g['type'] == p['type']:
                    num_head_relation_match_step[g['type']] += 1

        for g in gold_steps:
            for p in pred_steps:
                if g['pre'] == p['pre'] and g['type'] == p['type'] and g['con'] == p['con']:
                    num_head_relation_tail_match_step[g['type']] += 1


    ratio_tail_match_in_head_relation_match_step = {
        '=>': num_head_relation_tail_match_step['=>']/(1e-10 + num_head_relation_match_step['=>']),
        '->': num_head_relation_tail_match_step['->']/(1e-10 + num_head_relation_match_step['->']),
    }
    
    return {
        'degree_macro_f1':degree_macro_f1,
        'degree_micro_f1':degree_micro_f1,
        
        'num_head_relation_match_step_con': num_head_relation_match_step['->'],
        'num_head_relation_match_step_reb': num_head_relation_match_step['=>'],

        'num_head_relation_tail_match_step_con': num_head_relation_tail_match_step['->'],
        'num_head_relation_tail_match_step_reb': num_head_relation_tail_match_step['=>'],

        'ratio_tail_match_in_head_relation_match_step_con': ratio_tail_match_in_head_relation_match_step['->'],
        'ratio_tail_match_in_head_relation_match_step_reb': ratio_tail_match_in_head_relation_match_step['=>'],
    
    }