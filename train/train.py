#!/usr/bin/env python
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler
from pytorch_metric_learning import samplers
import logging
import time
import pdb
import os
import json
import random
from tqdm import tqdm

import sys
sys.path.append("../") 

import wandb

from src.data_loader import (
    DictionaryDataset,
    MetricLearningDataset_pairwise_types,
    QueryDataset,
    QueryDataset_pretraining,
    MetricLearningDataset,
    MetricLearningDataset_pairwise,
)
from src.model_wrapper import (
    Model_Wrapper
)
from src.metric_learning import (
    Sap_Metric_Learning,
    Sap_Metric_Learning_Types,
)

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='sapbert train')

    # Required
    parser.add_argument('--model_dir', 
                        help='Directory for pretrained model')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=240, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=3, type=int)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--amp', action="store_true", 
            help="automatic mixed precision training")
    parser.add_argument('--parallel', action="store_true") 
    #parser.add_argument('--cased', action="store_true") 
    parser.add_argument('--pairwise', action="store_true",
            help="if loading pairwise formatted datasets") 
    parser.add_argument('--random_seed',
                        help='epoch to train',
                        default=1996, type=int)
    parser.add_argument('--loss',
                        help="{ms_loss|cosine_loss|circle_loss|triplet_loss}}",
                        default="ms_loss")
    parser.add_argument('--use_miner', action="store_true") 
    parser.add_argument('--miner_margin', default=0.2, type=float) 
    parser.add_argument('--type_of_triplets', default="all", type=str) 
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_all_tok}") 
    parser.add_argument('--trust_remote_code', action="store_true",
                        help="allow for custom models defined in their own modeling files")

    # Adapter related 
    parser.add_argument('--use_adapters', action="store_true", help="Enable adapter training")
    parser.add_argument('--adapter_config', type=str, default="lora", help="Adapter configuration (e.g., pfeiffer)")

    # Wandb
    parser.add_argument('--wandb_project', type=str, default="sapbert",
                        help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default="full-model",
                        help='wandb run name')

    # additional params for type
    parser.add_argument('--enable_types', action='store_true',
        help='Enable the type-aware pipeline. If False, run the original codepath unchanged.')
    parser.add_argument('--tui2idx_json', type=str, default='tui2idx.json',
                    help='Path to TUI→index mapping written by the generator')
    parser.add_argument('--use_type_weights', action='store_true',
                        help='Use multi-hot TUI similarity to weight positives in MS loss')
    parser.add_argument('--w_type', type=float, default=0.5,
                        help='Weight for type-based positives in MS loss')
    parser.add_argument('--tau_type', type=float, default=0.2,
                        help='Threshold on type affinity to count as weak positives')
    parser.add_argument('--use_sty_aux', action='store_true',
                        help='Add auxiliary multi-label STY/TUI prediction loss')
    parser.add_argument('--lambda_sty', type=float, default=0.1,
                        help='Weight for STY auxiliary loss')



    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )
    
    return dictionary.data
    
def load_queries(data_dir, filter_composite, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data

def load_queries_pretraining(data_dir, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset_pretraining(
        data_dir=data_dir,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data


def train_with_types(args, data_loader, model, scaler=None, model_wrapper=None, step_global=0):
    LOGGER.info("train (types)!")
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()

        batch_x1, batch_x2, batch_y, batch_mh = data
        batch_x_cuda1 = {k: v.cuda() for k, v in batch_x1.items()}
        batch_x_cuda2 = {k: v.cuda() for k, v in batch_x2.items()}
        batch_y_cuda = batch_y.cuda()

        if batch_mh.numel() == 0 or (batch_mh.sum().item() == 0):
            batch_mh_cuda = None
        else:
            batch_mh_cuda = batch_mh.cuda()

        if args.amp:
            with autocast():
                loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda, batch_mh_cuda)
        else:
            loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda, batch_mh_cuda)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        train_loss += loss.item()
        wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1

        if step_global % args.checkpoint_step == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_iter_{step_global}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_wrapper.save_model(checkpoint_dir)

    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global


def train(args, data_loader, model, scaler=None, model_wrapper=None, step_global=0):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()

        batch_x1, batch_x2, batch_y = data
        batch_x_cuda1, batch_x_cuda2 = {},{}
        for k,v in batch_x1.items():
            batch_x_cuda1[k] = v.cuda()
        for k,v in batch_x2.items():
            batch_x_cuda2[k] = v.cuda()

        batch_y_cuda = batch_y.cuda()
    
        if args.amp:
            with autocast():
                loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)  
        else:
            loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)  
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        train_loss += loss.item()
        wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1
        #if (i+1) % 10 == 0:
        #LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1,train_loss / (train_steps+1e-9)))
        #LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1, loss.item()))

        # save model every K iterations
        if step_global % args.checkpoint_step == 0:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_iter_{}".format(str(step_global)))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global

def build_tui_vocab_from_pair_file(path):
    tui_set = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('||')
            if len(parts) >= 4:
                tui_csv = parts[3].strip()
                if tui_csv:
                    tui_set.update(tui_csv.split(','))
    tui_list = sorted(tui_set)
    tui2idx = {t:i for i,t in enumerate(tui_list)}
    return tui2idx

def collate_fn_batch_encoding_types(batch, tokenizer, tui2idx, max_length):
    q1, q2, y, tui_sets = zip(*batch)

    enc1 = tokenizer.batch_encode_plus(
        list(q1), max_length=max_length, padding="max_length",
        truncation=True, add_special_tokens=True, return_tensors="pt")
    enc2 = tokenizer.batch_encode_plus(
        list(q2), max_length=max_length, padding="max_length",
        truncation=True, add_special_tokens=True, return_tensors="pt")

    labels = torch.tensor(list(y), dtype=torch.long)

    T = len(tui2idx)
    if T == 0:
        mh = torch.empty(0)
    else:
        B = len(tui_sets)
        mh = torch.zeros(B, T, dtype=torch.float32)
        for i, s in enumerate(tui_sets):
            for t in s:
                j = tui2idx.get(t)
                if j is not None:
                    mh[i, j] = 1.0

    return enc1, enc2, labels, mh

def train_file_has_types(path, probe_lines=200):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for _ in range(probe_lines):
                line = f.readline()
                if not line:
                    break
                parts = line.rstrip('\n').split('||')
                if len(parts) >= 4 and '|' in parts[3]:
                    return True
    except Exception:
        pass
    return False

    
def main(args):
    init_logging()
    #init_seed(args.seed)
    print(args)
    torch.manual_seed(args.random_seed)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    model_wrapper = Model_Wrapper()  

    if args.use_adapters:
        LOGGER.info("Loading BERT model with adapters")
        encoder, tokenizer = model_wrapper.load_bert_adapter(
            path=args.model_dir,
            max_length=args.max_length,
            use_cuda=args.use_cuda,
            #lowercase=not args.cased,
            adapter_config=args.adapter_config
        )
    else:
        LOGGER.info("Loading BERT model")
        encoder, tokenizer = model_wrapper.load_bert(
            path=args.model_dir,
            max_length=args.max_length,
            use_cuda=args.use_cuda,
            #lowercase=not args.cased
    )

    use_types = False
    if args.enable_types:
        if train_file_has_types(args.train_dir):
            use_types = True
        else:
            LOGGER.warning("`--enable_types` was passed but the training file has no types. "
                        "Falling back to the ORIGINAL pipeline.")

    if use_types:
        # load TUI map (optional but recommended)
        tui2idx = {}
        if os.path.exists(args.tui2idx_json):
            with open(args.tui2idx_json, 'r', encoding='utf-8') as f:
                tui2idx = json.load(f)
        num_tui = len(tui2idx)
        LOGGER.info(f"TUI vocab size: {num_tui}")

        # dataset + collate (types)
        train_set = MetricLearningDataset_pairwise_types(
            path=args.train_dir, tokenizer=tokenizer, tui2idx=tui2idx
        )
        def _collate_types(b):
            return collate_fn_batch_encoding_types(b, tokenizer, tui2idx, args.max_length)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=_collate_types
        )

        # model (types)
        model = Sap_Metric_Learning_Types(
            encoder=encoder,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_cuda=args.use_cuda,
            pairwise=True,
            loss=args.loss,
            use_miner=args.use_miner,
            miner_margin=args.miner_margin,
            type_of_triplets=args.type_of_triplets,
            agg_mode=args.agg_mode,
            use_type_weights=args.use_type_weights,
            w_type=args.w_type,
            tau_type=args.tau_type,
            use_sty_aux=args.use_sty_aux,
            lambda_sty=args.lambda_sty,
            num_tui=num_tui
        )

        if args.parallel:
            model.encoder = torch.nn.DataParallel(model.encoder)
            LOGGER.info("using nn.DataParallel")

        # choose new train loop
        train_fn = train_with_types
    else:
        # load SAP model
        model = Sap_Metric_Learning(
                encoder = encoder,
                learning_rate=args.learning_rate, 
                weight_decay=args.weight_decay,
                use_cuda=args.use_cuda,
                pairwise=args.pairwise,
                loss=args.loss,
                use_miner=args.use_miner,
                miner_margin=args.miner_margin,
                type_of_triplets=args.type_of_triplets,
                agg_mode=args.agg_mode,
        )

        if args.parallel:
            model.encoder = torch.nn.DataParallel(model.encoder)
            LOGGER.info("using nn.DataParallel")
        
        def collate_fn_batch_encoding(batch):
            query1, query2, query_id = zip(*batch)
            query_encodings1 = tokenizer.batch_encode_plus(
                    list(query1), 
                    max_length=args.max_length, 
                    padding="max_length", 
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt")
            query_encodings2 = tokenizer.batch_encode_plus(
                    list(query2), 
                    max_length=args.max_length, 
                    padding="max_length", 
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt")
            #query_encodings_cuda = {}
            #for k,v in query_encodings.items():
            #    query_encodings_cuda[k] = v.cuda()
            query_ids = torch.tensor(list(query_id))
            return  query_encodings1, query_encodings2, query_ids

        if args.pairwise:
            train_set = MetricLearningDataset_pairwise(
                    path=args.train_dir,
                    tokenizer = tokenizer
            )
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=collate_fn_batch_encoding
            )
        else:
            train_set = MetricLearningDataset(
                path=args.train_dir,
                tokenizer = tokenizer
            )
            # using a sampler
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.train_batch_size,
                #shuffle=True,
                sampler=samplers.MPerClassSampler(train_set.query_ids,\
                    2, length_before_new_iter=100000),
                num_workers=16, 
                )
            
        train_fn = train
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    scaler = GradScaler() if args.amp else None

    start = time.time()
    step_global = 0
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))

        # train
        train_loss, step_global = train_fn(args, data_loader=train_loader, model=model, scaler=scaler, model_wrapper=model_wrapper, step_global=step_global)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        
        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if args.use_adapters:
                model_wrapper.save_adapter_model(checkpoint_dir)
            else:
                model_wrapper.save_model(checkpoint_dir)
        
        # save model last epoch
        if epoch == args.epoch:
            if args.use_adapters:
                model_wrapper.save_adapter_model(args.output_dir)
            else:
                model_wrapper.save_model(args.output_dir)

    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
