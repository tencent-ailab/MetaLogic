import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore TF log
import sys
import random
import copy
import os.path as osp
import glog as log
import json
import math
import time
import argparse

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import BartForConditionalGeneration,BartTokenizer
from transformers import T5ForConditionalGeneration,T5Tokenizer

from transformers.optimization import Adafactor,AdamW,get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F


from tree_utils import * 
from evaluate_metric import * 


from TreeGenerator import Tree_dataset
from TripleGenerator import Triple_dataset
from DegreeGenerator import Degree_dataset
from ControllerGenerator import Controller_dataset
from ModuleGenerator import Module_dataset


from TreeGenerator import eval_model as eval_model_tree
from TripleGenerator import eval_model as eval_model_triple
from DegreeGenerator import eval_model as eval_model_degree
from ControllerGenerator import eval_model as eval_model_controller
from ModuleGenerator import eval_model as eval_model_module

from inference import inference_multitask, inference_metgen_multitask


##### hrx experiment utils
import socket
import getpass
def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname, file_only=False):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        log.logger.handlers[0].stream = log.handler.stream = sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

#####


def add_prefix(prefix, datas):
    for data_item in datas:
        data_item['prefix'] = prefix
        data_item['original_src'] = data_item['src']
        if not data_item['src'].startswith(prefix):
            data_item['src'] = prefix + data_item['src']
            
    return datas

class Multitask_dataset(Dataset):

    def __init__(self, multitask_info, sampling_temperature = 1, args=None):
        
        num_task = len(multitask_info)
        
        # add prefix to each task
        for task_idx, task_item in enumerate(multitask_info):
            task_item['dataset'] = add_prefix(task_item['prefix'], task_item['dataset'])
        
        # concated all task datas
        concated_datas = []
        for task_item in multitask_info:
            concated_datas += list(task_item['dataset'])
        
        # change sampling probability according to the sampling_temperature t
        # Examples-proportional mixing: t = 1; Equal mixing: t = +inf; In general, t >= 1
        sample_r = np.array([len(task_item['dataset']) for task_item in multitask_info])
        if sampling_temperature >= 100: 
            sample_r = np.ones_like(sample_r) # set equal mixing
        elif sampling_temperature > 0:
            sample_r = np.power(sample_r, 1/sampling_temperature)
        sample_r = sample_r / np.sum(sample_r)
        
        
        self.multitask_info = multitask_info
        self.sample_r = sample_r
        self.num_task = num_task
        self.sampling_temperature = sampling_temperature
        
        self.concated_datas = concated_datas
            
        print(f"{self.__class__.__name__} multitask_info: {multitask_info}")
        print(f"Length of concated_datas: {len(concated_datas)}")
        print(f"sampling_temperature: {sampling_temperature} \t sample_r: {sample_r}")
        for task_item in multitask_info:
            print(f"prefix: {task_item['prefix']} length: {len(task_item['dataset'])}")
            
    def __getitem__(self, index):
        if index >= len(self.concated_datas):
            raise IndexError # if not, the length of this dataset is +inf
        
        if self.sampling_temperature == 1:
            return self.concated_datas[index]
        else:
            sampled_task = random.choices(self.multitask_info, weights = self.sample_r, k=1)[0]
            sampled_dataset = sampled_task['dataset']
            
            random_item = random.choice(sampled_dataset) # randomly sampled from the dataset; ignore the index
            
            return random_item
    
    def __len__(self):
        return len(self.concated_datas)





def create_optimizer(model,args):
    # decay if not LayerNorm or bias
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    if args.adafactor:
        print('optimizer: Adafactor')
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        print('optimizer: AdamW')
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps":1e-6,
        }
    optimizer_kwargs["lr"] = args.lr
    
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

def create_scheduler(optimizer, args):
    warmup_steps = math.ceil(args.num_training_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.num_training_steps,
    )
    return lr_scheduler

def train_one_step(batch, model, tokenizer, args):
    r"""
    train the model one step with the given batch data
    return the loss
    """
    model.train()
    
    # process batch data
    input_sents = [item['src'] for item in batch]
    output_sents = [item['tgt'] for item in batch]

    input_batch = tokenizer(
            input_sents,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length', # 'longest',
            max_length=args.max_src_length,
            truncation=True,)

    output_batch = tokenizer(
                output_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding= 'max_length', # 'longest',
                max_length=args.max_tgt_length,
                truncation=True,)

    # Replace the <pad> to -100 for computing loss
    label_batch = output_batch['input_ids']
    label_batch.masked_fill_(label_batch == tokenizer.pad_token_id, -100) 
    
    input_batch['labels'] = label_batch
    input_batch = input_batch.to(model.device)
    
    # forward
    model_return = model(**input_batch)

    return model_return['loss']

def eval_end_task(model, datas, tokenizer, args):

    collector = {}
    for data_item in datas:
        gold_item = data_item['gold_item']

        if args.multitask_type == 'multi':
            pred_item = inference_multitask(data_item, (model, tokenizer, args))
        elif args.multitask_type == 'metgen':
            pred_item = inference_metgen_multitask(data_item, (model, tokenizer, args), args.context_input)
        else:
            raise NotImplementedError

        eval_result = eval_sample(pred_item, gold_item)
        
        collector[data_item['id_string']] = {
            'gold_item':gold_item,
            'pred_item':pred_item,
            'score': eval_result,
        }

    scores = [v['score'] for k,v in collector.items()]
    average_scores = average_metric(scores)
    s_ = average_scores['Overall'] + 0.01*(average_scores['step_f1']*average_scores['triples_f1']*average_scores['degree_acc'])
    
    return s_, average_scores, collector

def run(args):

    # set random seed before init model
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if torch.cuda.device_count() > 1:
    #     torch.cuda.manual_seed_all(args.seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading data")
    dev_datas = [json.loads(line) for line in open(args.dev_data).readlines()]
    test_datas = [json.loads(line) for line in open(args.test_data).readlines()]

    multitask_info = []
    if args.multitask_type == 'multi':
        multitask_info.append(
            {
                'prefix': 'TREE: ',
                'dataset': Tree_dataset(args.train_data, args=args),
                'dev_dataset': Tree_dataset(args.dev_data, args=args),
                'test_dataset': Tree_dataset(args.test_data, args=args),
                'eval_func': eval_model_tree,
            }
        )
        multitask_info.append(
            {
                'prefix': 'FORMULAE: ',
                'dataset': Triple_dataset(args.train_data, args=args),
                'dev_dataset': Triple_dataset(args.dev_data, args=args),
                'test_dataset': Triple_dataset(args.test_data, args=args),
                'eval_func': eval_model_triple,
            }
        )
        multitask_info.append(
            {
                'prefix': 'DEGREE: ',
                'dataset': Degree_dataset(args.train_data, args=args),
                'dev_dataset': Degree_dataset(args.dev_data, args=args),
                'test_dataset': Degree_dataset(args.test_data, args=args),
                'eval_func': eval_model_degree,
            }
        )
    elif args.multitask_type == 'metgen':
        multitask_info.append(
            {
                'prefix': 'CONTROL: ',
                'dataset': Controller_dataset(args.train_data, args=args),
                'dev_dataset': Controller_dataset(args.dev_data, args=args),
                'test_dataset': Controller_dataset(args.test_data, args=args),
                'eval_func': eval_model_controller,
            }
        )
        multitask_info.append(
            {
                'prefix': '', # prefix in module dataset
                'dataset': Module_dataset(args.train_data, args=args),
                'dev_dataset': Module_dataset(args.dev_data, args=args),
                'test_dataset': Module_dataset(args.test_data, args=args),
                'eval_func': eval_model_module,
            }
        )
        multitask_info.append(
            {
                'prefix': 'FORMULAE: ',
                'dataset': Triple_dataset(args.train_data, args=args),
                'dev_dataset': Triple_dataset(args.dev_data, args=args),
                'test_dataset': Triple_dataset(args.test_data, args=args),
                'eval_func': eval_model_triple,
            }
        )
        multitask_info.append(
            {
                'prefix': 'DEGREE: ',
                'dataset': Degree_dataset(args.train_data, args=args),
                'dev_dataset': Degree_dataset(args.dev_data, args=args),
                'test_dataset': Degree_dataset(args.test_data, args=args),
                'eval_func': eval_model_degree,
            }
        )
    else:
        raise NotImplementedError


    train_dataset = Multitask_dataset(multitask_info, sampling_temperature=args.sampling_temperature)
    train_loader = DataLoader(dataset = train_dataset,
                            batch_size = args.bs,
                            shuffle = True,
                            num_workers = 8,
                            pin_memory = True,
                            collate_fn = lambda batch: batch)

    for task_item in multitask_info:
        task_item['dev_dataset'] = add_prefix(task_item['prefix'], task_item['dev_dataset'])
        task_item['test_dataset'] = add_prefix(task_item['prefix'], task_item['test_dataset'])

        task_item['dev_loader'] = DataLoader(dataset = task_item['dev_dataset'],
                                            batch_size = args.bs,
                                            shuffle = True,
                                            num_workers = 8,
                                            pin_memory = True,
                                            collate_fn = lambda batch: batch)
        task_item['test_loader'] = DataLoader(dataset = task_item['test_dataset'],
                                            batch_size = args.bs,
                                            shuffle = True,
                                            num_workers = 8,
                                            pin_memory = True,
                                            collate_fn = lambda batch: batch)

    log.info(f"number of iteration each epoch : {len(train_loader)}")
    args.eval_iter = round(args.eval_epoch * len(train_loader))
    args.report_iter = round(args.report_epoch * len(train_loader))
    args.num_training_steps = args.epochs * len(train_loader)

    log.info("loading model")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer.sep_token = tokenizer.eos_token

    if args.model_name_or_path in ['t5-11b']:
        model.parallelize()
    else:
        model = model.to(device)

    with open(osp.join(args.exp_dir, 'model.config.json'), 'w') as f:
        json.dump(vars(model.config), f, sort_keys=False, indent=4)

    optimizer = create_optimizer(model, args)
    lr_scheduler = create_scheduler(optimizer, args)

    if args.resume_path:
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log.info(f"Resume model parameters form {args.resume_path}")

    log.info("start training")
    global_iter = 0
    loss_list = []
    best_metric = -100

    for epoch_i in range(1, args.epochs+1):
        
        for batch in train_loader:
            loss = train_one_step(batch,model,tokenizer,args)
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            global_iter += 1

            if not global_iter % args.eval_iter:
                log.info(f"------Iteration {global_iter} evaluating ------")
                # evaluated on the end task
                metric_collecter = {'iter': global_iter, 'dev':{}, 'test':{}}
                
                # dev
                eval_score, average_scores, _ = eval_end_task(model, dev_datas, tokenizer, args)
                metric_collecter['dev'] =average_scores
                
                new_metric = eval_score
                
                if best_metric < new_metric:
                    best_metric = new_metric
                    log.info(f"------Iteration {global_iter} get best metric {best_metric:.4f}------")
                    if args.save_model:
                        save_path = osp.join(args.exp_dir,'best_model.pth')
                        torch.save(model.state_dict(), save_path)
                        log.info(f"Iteration {global_iter} save best model")

                # test
                _, average_scores, pred_collector = eval_end_task(model, test_datas, tokenizer, args)
                metric_collecter['test'] =average_scores
                with open(osp.join(args.exp_dir, 'prediction', f'prediction_{global_iter}.json'), 'w') as f: 
                    json.dump(pred_collector, f, indent=4)

                with open(osp.join(args.exp_dir, 'metrics.json'), 'a+') as f:
                    f.write(json.dumps(metric_collecter) + '\n')

            if not global_iter % args.report_iter:
                log.info(f"Epoch {global_iter/len(train_loader):.1f} training loss {np.mean(loss_list):.4f}")
                loss_list = []
            else:
                loss_list.append(float(loss.cpu().data))
                
        log.info(f"Epoch {epoch_i} finished")

        # save_path = osp.join(args.exp_dir,'latest_model.pth')
        # torch.save(model.state_dict(), save_path)
        torch.save({
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_loss_list': loss_list,
        }, osp.join(args.exp_dir, 'latest_checkpoint'))
        log.info(f"Epoch {epoch_i} save latest model")


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training')

    # dateset
    parser.add_argument("--train_data", type=str)  
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    
    # dateset: multitask
    parser.add_argument('--multitask_type', default='multi', type=str)
    parser.add_argument('--sampling_temperature', default=1, type=float)

    # dateset: Module dataset
    parser.add_argument('--neg_pos_rate', type=float, default=2.0)
    parser.add_argument('--context_input', action='store_true', default=False)


    
    # model
    parser.add_argument("--model_name_or_path", type=str, 
                        default="t5-base", help="")  
    parser.add_argument("--resume_path", type=str, 
                        default="", help="")                
    parser.add_argument('--max_src_length', type=int, default=128, )
    parser.add_argument('--max_tgt_length', type=int, default=128, )

    # optimization
    parser.add_argument('--bs', type=int, default=16, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train')
                        
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--adafactor', action='store_true', default=False)
    parser.add_argument('--lr_scheduler_type', type=str, default='constant_with_warmup')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--eval_epoch', type=float, default=1.0)

    # seed
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')

    # exp and log
    parser.add_argument("--exp_dir", type=str, default='./exp')
    parser.add_argument("--code_dir", type=str, default='./code')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--report_epoch', type=float, default=1.0)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    args = get_params()
    if args.seed == 0:
        args.seed = random.randint(1,1e4)

    args.exp_dir = osp.join(args.exp_dir, get_random_dir_name())

    os.makedirs(args.exp_dir, exist_ok=True)
    # set_log_file(osp.join(args.exp_dir, 'run.log'), file_only=True)
    
    # make metrics.json for logging metrics
    args.metric_file = osp.join(args.exp_dir, 'metrics.json')
    open(args.metric_file, 'a').close()
    
    os.makedirs(osp.join(args.exp_dir, 'prediction'), exist_ok=True)

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    # os.system(f'cp -r {args.code_dir} {args.exp_dir}')

    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))
    log.info('Python info: {}'.format(os.popen('which python').read().strip()))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)

    run(args)

    # make 'done' file
    open(osp.join(args.exp_dir, 'done'), 'a').close()