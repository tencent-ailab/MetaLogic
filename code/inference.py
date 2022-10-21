import os
import sys
import random
import copy
import os.path as osp
import glog as log
import json
import time
import argparse
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

from tree_utils import * 
from evaluate_metric import * 


def load_model(exp_dir, model_name='best_model.pth'):
    if model_name is None:
        model_name = 'model_best.pth'
    print(f"loading model from {osp.join(exp_dir,model_name)}")
    model_args = json.load(open(osp.join(exp_dir,'config.json')))
    tokenizer = AutoTokenizer.from_pretrained(model_args['model_name_or_path'],use_fast=False,local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(model_args['model_name_or_path'],local_files_only=True)
    state_dict = torch.load(osp.join(exp_dir,model_name), map_location='cpu')
    model.load_state_dict(state_dict)

    return model, tokenizer, model_args

def compute_ppl(input_sents, output_sents, model, tokenizer, args): 
    model.eval()
    assert len(input_sents) == len(output_sents)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) # ignore <pad> !

    input_batch = tokenizer(input_sents,add_special_tokens=True,return_tensors='pt',padding='longest',truncation=True)
    output_batch = tokenizer(output_sents,add_special_tokens=True, return_tensors='pt', padding='longest',truncation=True)
    input_batch['labels'] = output_batch['input_ids']
    input_batch = input_batch.to(model.device)

    with torch.no_grad():
        logits = model(**input_batch)['logits']

    ppls = []
    for idx in range(len(output_sents)):

        loss = loss_fn(logits[idx].view(-1, logits.shape[-1]), input_batch['labels'][idx].view(-1))
        ppl = torch.exp(loss)

        ppls.append(float(ppl))

    return ppls

def model_inference(batch, model, tokenizer, args, num_return_sequences=1):  
    model.eval()
    
    input_sents = [item['src'] for item in batch]
    
    input_batch = tokenizer(
            input_sents,
            add_special_tokens=True,
            return_tensors='pt',
            padding='longest',
            max_length=256,
            truncation=True,
        )
    input_batch = input_batch.to(model.device)

    # greedy search
    generated = model.generate(
        input_ids = input_batch['input_ids'],
        attention_mask = input_batch['attention_mask'],
        num_beams = 1,
        do_sample = False,
        max_length= 256, 
    )


    decoded = tokenizer.batch_decode(generated, skip_special_tokens = True)
    
    return decoded

def inference_once(data_item, model_tuple):
    model, tokenizer, model_args = model_tuple

    logical_passage = ' '.join([sent_info['sent'] for sent_id, sent_info in data_item['sent_dict'].items()])
    batch = [{'src': logical_passage}]
    pred_ = model_inference(batch, model, tokenizer, model_args)[0]

    pred_item = parse_once(pred_)

    return pred_item


def inference_multitask(data_item, model_tuple):
    model, tokenizer, model_args = model_tuple
    
    # predict proof
    logical_passage = ' '.join([sent_info['sent'] for sent_id, sent_info in data_item['sent_dict'].items()])

    batch = [{'src': task2prefix['tree'] + logical_passage}]
    proof_str = model_inference(batch, model, tokenizer, model_args)[0]
    proof_str = proof_str.replace(task2flag['tree'], '').strip()

    # predict triple
    triples_str_dict = {}
    batch = []
    for sent_id, sent_info in data_item['sent_dict'].items():
        batch.append({'src': task2prefix['triple'] + sent_info['inner_info']['inner_sent_w_variables']})
    pred_triples_strs = model_inference(batch, model, tokenizer, model_args)
    for sent_id, pred_triples_str in zip(data_item['sent_dict'].keys(), pred_triples_strs):
        triples_str_dict[sent_id] = pred_triples_str.replace(task2flag['triple'], '').strip()

    # predict degree
    degree_str_dict = {}
    batch = []
    for sent_id, sent_info in data_item['sent_dict'].items():
        batch.append({'src': task2prefix['degree'] + sent_info['sent']})
    pred_degree_strs = model_inference(batch, model, tokenizer, model_args)
    for sent_id, pred_degree_str in zip(data_item['sent_dict'].keys(), pred_degree_strs):
        degree_str_dict[sent_id] = pred_degree_str.replace(task2flag['degree'], '').strip()

    pred_item = {
        'proof_str':proof_str,
        'triples_str_dict':triples_str_dict,
        'degree_str_dict':degree_str_dict,
    }

    return pred_item

def check_step_addable(new_step, previous_steps):
    def split_step(step_str):
        if '->' in step_str:
            return [s_.strip() for s_ in step_str.split('->')]
        if '=>' in step_str:
            return [s_.strip() for s_ in step_str.split('=>')]
        return [step_str]

    if new_step == 'none':
        return True

    # rule: the same one sentX are not used as two premises
    all_previous_pres = [split_step(s)[0] for s in previous_steps]
    new_step_pre = split_step(new_step)[0]
    if new_step_pre in all_previous_pres:
        return False

    return True


def inference_metgen_proof(data_item, controller_tuple, module_turple, context_input, p_comb = 0.5, min_used_sent_ratio = 1.0, max_num_step = 3, verbose = False):
    # context_input = True

    controller_model, tokenizer, controller_args = controller_tuple
    module_model, tokenizer, module_args = module_turple

    logical_passage = ' '.join([sent_info['sent'] for sent_id, sent_info in data_item['sent_dict'].items()])
    all_sent_ids = list(data_item['sent_dict'].keys())

    previous_steps = []
    previous_combs = []
    retry_step = False

    while True:

        used_sent = sum(previous_combs,[])
        used_sent_ratio = len(set(used_sent).intersection(set(all_sent_ids))) / len(all_sent_ids)

        # collect candidate combinations, e.g., ['sent1', 'sent2']
        candidate_combs = [list(sorted(c)) for c in list(combinations(all_sent_ids, 2))]
        candidate_combs = [c for c in candidate_combs if c not in previous_combs]
        candidate_combs = [' '.join(sorted(c)) for c in candidate_combs] + ['done']

        # compute score for each combination
        controller_inputs = f"proof: {'; '.join(previous_steps)}; context: {logical_passage}"
        inputs = [controller_inputs]*len(candidate_combs)
        ppls = compute_ppl(inputs, candidate_combs, controller_model, tokenizer, controller_args)
        comb_scores = [1/ppl for ppl in ppls]

        # sorted by scores
        candidate_combs_with_scores = sorted(zip(candidate_combs, comb_scores), key=lambda x:x[1], reverse=True)
        controller_done = (candidate_combs_with_scores[0][0] == 'done')
        candidate_combs_with_scores = [c_ for c_ in candidate_combs_with_scores if c_[0]!='done']

        if not retry_step:
            # consider the combinations with top-p scores
            candidate_combs_with_scores = candidate_combs_with_scores[:int(p_comb*len(candidate_combs))+1]
        else:
            # try all step
            candidate_combs_with_scores = candidate_combs_with_scores

        candidate_combs = [cs[0] for cs in candidate_combs_with_scores]
        print('Combs:', candidate_combs_with_scores) if verbose else None

        if len(previous_steps) >= max_num_step:
            break
        if used_sent_ratio >= min_used_sent_ratio and controller_done:
            break


        # module inference
        module_inputs = []
        candidate_steps = []
        for comb_str in candidate_combs:
            next_comb = [sent_id.strip() for sent_id in comb_str.split()]
            if len(next_comb) < 2: continue
            if any([sent_id not in data_item['sent_dict'] for sent_id in next_comb]): continue

            module_input = ""
            if not context_input:
                for sent_id in next_comb:
                    module_input += f"{sent_id}: {data_item['sent_dict'][sent_id]['sent']} "
            else:
                module_input += ' '.join(next_comb)
                module_input += '; ' + logical_passage

            ### conclusion module
            module_inputs += [task2prefix['module_clu'] + module_input] * 3
            candidate_steps += [' -> '. join(next_comb), ' -> '. join(next_comb[::-1]), 'none']
            ### rebuttal module
            module_inputs += [task2prefix['module_reb'] + module_input] * 3
            candidate_steps += [' => '. join(next_comb), ' => '. join(next_comb[::-1]), 'none']

        ppls = compute_ppl(module_inputs, candidate_steps, module_model, tokenizer, module_args)
        result_scores = [1/ppl for ppl in ppls]

        # filter by rule
        for idx, new_step in enumerate(candidate_steps):
            if not check_step_addable(new_step, previous_steps):
                result_scores[idx] = -1


        candidate_steps_with_scores = []
        # consider each combination
        for rs in chunk(zip(candidate_steps, result_scores), 3):
            rs = sorted(rs, key=lambda x:x[1], reverse=True)
            # print(rs)
            if not retry_step:
                # sent1->sent2 vs. sent2->sent1 vs. none
                if rs[0][0] != 'none':
                    candidate_steps_with_scores.append(rs[0])
            else:
                # sent1->sent2 vs. sent2->sent1
                if rs[0][0] != 'none':
                    candidate_steps_with_scores.append(rs[0])
                else:
                    candidate_steps_with_scores.append(rs[1])


        # sorted by scores
        candidate_steps_with_scores = sorted(candidate_steps_with_scores, key=lambda x:x[1], reverse=True)
        print("Steps:", candidate_steps_with_scores) if verbose else None

        if len(candidate_steps_with_scores) == 0:
            # reject the top-p steps; try all steps
            retry_step = True
            continue
        elif candidate_steps_with_scores[0][1] < 0:
            # all candidate step unaddable
            break
        else:
            # select the top-1 step as next step
            retry_step = False
            selected_step = candidate_steps_with_scores[0][0]
            selected_comb = selected_step.replace(' -> ', ' ').replace(' => ', ' ').split()
            previous_steps.append(selected_step)
            previous_combs.append(sorted(selected_comb))

    pred_proof_str = '; '.join(previous_steps)

    return pred_proof_str


def inference_metgen_multitask(data_item, model_tuple, context_input, p_comb = 0.5, min_used_sent_ratio = 1.0, max_num_step = 3):
    model, tokenizer, model_args = model_tuple

    # predict proof
    controller_tuple = module_turple = model_tuple
    pred_proof_str = inference_metgen_proof(data_item, controller_tuple, module_turple, context_input=context_input,
                                            p_comb = p_comb,
                                            min_used_sent_ratio = min_used_sent_ratio,
                                            max_num_step = max_num_step)

    # predict triple
    triples_str_dict = {}
    batch = []
    for sent_id, sent_info in data_item['sent_dict'].items():
        batch.append({'src': task2prefix['triple'] + sent_info['inner_info']['inner_sent_w_variables']})
    pred_triples_strs = model_inference(batch, model, tokenizer, model_args)
    for sent_id, pred_triples_str in zip(data_item['sent_dict'].keys(), pred_triples_strs):
        triples_str_dict[sent_id] = pred_triples_str.replace(task2flag['triple'], '').strip()

    # predict degree
    degree_str_dict = {}
    batch = []
    for sent_id, sent_info in data_item['sent_dict'].items():
        batch.append({'src': task2prefix['degree'] + sent_info['sent']})
    pred_degree_strs = model_inference(batch, model, tokenizer, model_args)
    for sent_id, pred_degree_str in zip(data_item['sent_dict'].keys(), pred_degree_strs):
        degree_str_dict[sent_id] = pred_degree_str.replace(task2flag['degree'], '').strip()

    pred_item = {
        'proof_str':pred_proof_str,
        'triples_str_dict':triples_str_dict,
        'degree_str_dict':degree_str_dict,
    }
    
    return pred_item


def get_params():
    parser = argparse.ArgumentParser(description='Inference')

    parser.add_argument("--data_path", type=str)  

    parser.add_argument("--inference_type", type=str, default='once')  

    parser.add_argument("--exp_dir", type=str)  
    parser.add_argument("--model_name", type=str, default='best_model.pth')
    
    parser.add_argument('--context_input', action='store_true', default=False)
    parser.add_argument("--p_comb", type=float, default=0.5)  
    parser.add_argument("--min_used_sent_ratio", type=float, default=1.0)  
    parser.add_argument("--max_num_step", type=int, default=3)  




    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args

def get_time_str():
    from datetime import datetime
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return time_str

if __name__ == '__main__': 
    args = get_params()

    datas = [json.loads(line) for line in open(args.data_path).readlines()]
    datas = preprocess_raw_data(datas)

    datas = datas

    if args.inference_type == 'once':
        model, tokenizer, model_args = load_model(args.exp_dir, args.model_name)
        print(f"model loaded.")
        model.parallelize()
        # model = model.to('cuda')

        collector = {}

        for data_item in tqdm(datas):
            gold_item = data_item['gold_item']
            pred_item = inference_once(data_item, (model, tokenizer, model_args))
            
            eval_result = eval_sample(pred_item, gold_item)
            
            collector[data_item['id_string']] = {
                'gold_item':gold_item,
                'pred_item':pred_item,
                'score': eval_result,
            }

        average_scores = average_metric([v['score'] for v in collector.values()])
        average_scores.update(post_metrics(collector))

        for k,v in average_scores.items():
            print(f"{k}:  {v:.3f}")

        os.makedirs(osp.join(args.exp_dir, 'evaluation'), exist_ok=True)
        save_path = osp.join(args.exp_dir, 'evaluation', args.model_name + '_' + get_time_str())
        with open(save_path, 'w') as f:
            json.dump({'scores': average_scores,'collector': collector,}, f, indent=4)

        print(f"save result at {save_path}")

    elif args.inference_type == 'multi': 
        model, tokenizer, model_args = load_model(args.exp_dir, args.model_name)
        model = model.to('cuda')

        collector = {}

        for data_item in tqdm(datas):
            gold_item = data_item['gold_item']
            pred_item = inference_multitask(data_item, (model, tokenizer, model_args))
            
            eval_result = eval_sample(pred_item, gold_item)
            
            collector[data_item['id_string']] = {
                'gold_item':gold_item,
                'pred_item':pred_item,
                'score': eval_result,
            }

        average_scores = average_metric([v['score'] for v in collector.values()])
        average_scores.update(post_metrics(collector))

        for k,v in average_scores.items():
            print(f"{k}:  {v:.3f}")

        os.makedirs(osp.join(args.exp_dir, 'evaluation'), exist_ok=True)
        save_path = osp.join(args.exp_dir, 'evaluation', args.model_name + '_' + get_time_str())
        with open(save_path, 'w') as f:
            json.dump({'scores': average_scores,'collector': collector,}, f, indent=4)

        print(f"save result at {save_path}")

    elif args.inference_type == 'metgen': 
        model, tokenizer, model_args = load_model(args.exp_dir, args.model_name)
        model = model.to('cuda')

        collector = {}

        for data_item in tqdm(datas):
            gold_item = data_item['gold_item']
            pred_item = inference_metgen_multitask(data_item, (model, tokenizer, model_args),
                                                    context_input = args.context_input, 
                                                    p_comb = args.p_comb,
                                                    min_used_sent_ratio = args.min_used_sent_ratio,
                                                    max_num_step = args.max_num_step,)
            
            eval_result = eval_sample(pred_item, gold_item)
            
            collector[data_item['id_string']] = {
                'gold_item':gold_item,
                'pred_item':pred_item,
                'score': eval_result,
            }

        average_scores = average_metric([v['score'] for v in collector.values()])
        average_scores.update(post_metrics(collector))

        for k,v in average_scores.items():
            print(f"{k}:  {v:.3f}")

        os.makedirs(osp.join(args.exp_dir, 'evaluation'), exist_ok=True)
        save_path = osp.join(args.exp_dir, 'evaluation', args.model_name + '_' + get_time_str())
        with open(save_path, 'w') as f:
            json.dump({'scores': average_scores,'args': vars(args), 'collector': collector,}, f, indent=4)

        print(f"save result at {save_path}")

    else:
        raise NotImplementedError
