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
from sklearn.metrics import accuracy_score,balanced_accuracy_score

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

class Degree_dataset(Dataset):

    def __init__(self, dataset_path, flag = '$degree$', args = None):
        
        toulmin_datas = [json.loads(line) for line in open(dataset_path).readlines()]

        datas = []
        for data_item in toulmin_datas:
            for sent_id, sent_info in data_item['sent_dict'].items():
                degree = sent_info['inner_info']['degree_label']
                degree_str = f"{flag} {degree}"
                degree_str_map = transform_by_dict(degree_str, degree2token)

                datas.append({
                    '_id': data_item['id_string'],
                    'sent_id':sent_id,
                    'src': sent_info['sent'],
                    'tgt': degree_str_map,
                    'degree': degree,
                })
            
        self.datas = datas
        self.flag = flag

        print(f"{self.__class__.__name__} Loading from: {dataset_path}")
        print(f"Length of data: {len(self.datas)}")
        print(f"degree2token: {degree2token}")
            
    def __getitem__(self, index):
        return self.datas[index]
    
    def __len__(self):
        return len(self.datas)

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

def eval_model(model,data_loader, tokenizer, args):
    model.eval()

    inputs = []
    golds = []
    preds = []

    scores = []
    

    for batch in data_loader:

        # process batch data
        input_sents = [item['src'] for item in batch]
        output_sents = [item['tgt'] for item in batch]

        input_batch = tokenizer(
                input_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding='longest',
                max_length=args.max_src_length,
                truncation=True,
            )
        input_batch = input_batch.to(model.device)
        
        # generate
        generated = model.generate(
            input_ids = input_batch['input_ids'],
            attention_mask = input_batch['attention_mask'],
            top_p = 0.9,
            do_sample = True,
            max_length= args.max_tgt_length, 
            num_return_sequences = 1,
        )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        inputs += input_sents
        golds += output_sents
        preds += decoded

        for pred_, gold_ in zip(decoded, output_sents):
            scores.append(eval_degree(pred_, gold_, flag = '$degree$'))

    all_preds = [s['compare_degree'][0] for s in scores]
    all_golds = [s['compare_degree'][1] for s in scores]

    degree_acc = accuracy_score(y_true=all_golds, y_pred=all_preds)
    degree_acc_balanced = balanced_accuracy_score(y_true=all_golds, y_pred=all_preds)

    eval_info = {
        'inputs': inputs,
        'golds': golds,
        'preds': preds,
        'scores': scores,
    }

    average_scores = {
        'degree_acc': degree_acc,
        'degree_acc_balanced': degree_acc_balanced,
    }
    eval_info['average_scores'] = average_scores

    return degree_acc_balanced, eval_info


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


    train_dataset = Degree_dataset(args.train_data)
    dev_dataset = Degree_dataset(args.dev_data)
    test_dataset = Degree_dataset(args.test_data)


    log.info(f"Length of training dataest: {len(train_dataset)}")
    log.info(f"Length of dev dataest: {len(dev_dataset)}")
    log.info(f"Length of test dataest: {len(test_dataset)}")

    train_loader = DataLoader(dataset = train_dataset,
                            batch_size = args.bs,
                            shuffle = True,
                            num_workers = 4,
                            collate_fn = lambda batch: batch)
    dev_loader = DataLoader(dataset = dev_dataset,
                            batch_size = args.bs,
                            shuffle = False,
                            num_workers = 4,
                            collate_fn = lambda batch: batch)
    test_loader = DataLoader(dataset = test_dataset,
                            batch_size = args.bs,
                            shuffle = False,
                            num_workers = 4,
                            collate_fn = lambda batch: batch)

    log.info(f"number of iteration each epoch : {len(train_loader)}")
    args.eval_iter = round(args.eval_epoch * len(train_loader))
    args.report_iter = round(args.report_epoch * len(train_loader))
    args.num_training_steps = args.epochs * len(train_loader)


    log.info("loading model")
    if args.model_name_or_path in ['t5-large','t5-base','t5-small']:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
        tokenizer.sep_token = tokenizer.eos_token
    else:
        raise NotImplementedError
    
    if args.resume_path:
        state_dict = torch.load(args.resume_path, map_location='cpu')
        model.load_state_dict(state_dict)
        log.info(f"Resume model parameters form {args.resume_path}")

    model = model.to(device)

    with open(osp.join(args.exp_dir, 'model.config.json'), 'w') as f:
        json.dump(vars(model.config), f, sort_keys=False, indent=4)

    optimizer = create_optimizer(model, args)
    lr_scheduler = create_scheduler(optimizer, args)
    

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

                ###### Dev split
                eval_score, eval_info = eval_model(model,dev_loader,tokenizer,args)
                new_metric = eval_score

                log.info(f"Iteration {global_iter} Dev balanced_acc: {eval_score:.4f}")

                if best_metric < new_metric:
                    best_metric = new_metric
                    log.info(f"------Iteration {global_iter} get best metric {best_metric:.4f}------")
                    if args.save_model:
                        save_path = osp.join(args.exp_dir,'best_model.pth')
                        torch.save(model.state_dict(), save_path)
                        log.info(f"Iteration {global_iter} save best model")

                ###### Test split
                eval_score_test, eval_info_test = eval_model(model,test_loader,tokenizer,args)

                log.info(f"Iteration {global_iter} Test balanced_acc: {eval_score_test:.4f}")
                with open(osp.join(args.exp_dir, 'prediction', f'prediction_{global_iter}.txt'), 'w') as f: 
                    f.write("----------Test set eval----------\n")
                    for i_, g_, p_, s_ in zip(eval_info_test['inputs'], eval_info_test['golds'], 
                                                        eval_info_test['preds'], eval_info_test['scores']):
                        f.write(f"input: {i_}\n")
                        f.write(f"gold: {g_}\n")
                        f.write(f"pred: {p_}\n")
                        f.write(f"score: {s_}\n\n")


            if not global_iter % args.report_iter:
                log.info(f"Epoch {global_iter/len(train_loader):.1f} training loss {np.mean(loss_list):.4f}")
                loss_list = []
            else:
                loss_list.append(float(loss.cpu().data))
                
        log.info(f"Epoch {epoch_i} finished")

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training')

    # dateset
    parser.add_argument("--train_data", type=str)  
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)

    
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
    # try:
    #     # get parameters form tuner
    #     # tuner_params = nni.get_next_parameter()
    #     logger.debug(tuner_params)
    #     params = vars(merge_parameter(get_params(), tuner_params))
    #     print(params)

    #     main(params)
    # except Exception as exception:
    #     logger.exception(exception)
    #     raise

    args = get_params()
    if args.seed == 0:
        args.seed = random.randint(1,1e4)

    args.exp_dir = osp.join(args.exp_dir, get_random_dir_name())

    os.makedirs(args.exp_dir, exist_ok=True)
    set_log_file(osp.join(args.exp_dir, 'run.log'), file_only=True)
    
    # make metrics.json for logging metrics
    args.metric_file = osp.join(args.exp_dir, 'metrics.json')
    open(args.metric_file, 'a').close()
    
    os.makedirs(osp.join(args.exp_dir, 'prediction'), exist_ok=True)

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    os.system(f'cp -r {args.code_dir} {args.exp_dir}')

    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))
    log.info('Python info: {}'.format(os.popen('which python').read().strip()))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)

    run(args)

    # make 'done' file
    open(osp.join(args.exp_dir, 'done'), 'a').close()