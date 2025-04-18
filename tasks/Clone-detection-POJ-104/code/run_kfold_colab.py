# Load custom model and run K-Fold cross validation

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,SubsetRandomSampler,TensorDataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 index,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index=index
        self.label=label


def convert_examples_to_features(js,tokenizer,args):
    #source
    code=' '.join(js['code'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['index'],int(js['label']))


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data=[]
        with open(file_path) as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.critical("*** Example ***")
                    logger.critical("idx: {}".format(idx))
                    logger.critical("label: {}".format(example.label))
                    logger.critical("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.critical("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        self.label_examples={}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label]=[]
            self.label_examples[e.label].append(e)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        label=self.examples[i].label
        index=self.examples[i].index
        labels=list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example=random.sample(self.label_examples[label],1)[0]
            if shuffle_example.index!=index:
                p_example=shuffle_example
                break
        n_example=random.sample(self.label_examples[random.sample(labels,1)[0]],1)[0]
        
        return (torch.tensor(self.examples[i].input_ids),torch.tensor(p_example.input_ids),
                torch.tensor(n_example.input_ids),torch.tensor(label))
            


def load_model():
    if not args.load_custom_model:
        checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
        if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
            args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
            args.config_name = os.path.join(checkpoint_last, 'config.json')
            idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
            if os.path.exists(idx_file):
                with open(idx_file, encoding='utf-8') as idxf:
                    args.start_epoch = int(idxf.readlines()[0].strip()) + 1

            step_file = os.path.join(checkpoint_last, 'step_file.txt')
            if os.path.exists(step_file):
                with open(step_file, encoding='utf-8') as stepf:
                    args.start_step = int(stepf.readlines()[0].strip())

            logger.critical("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
        config.num_labels=1
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
        if args.model_name_or_path:
            model = model_class.from_pretrained(args.model_name_or_path,
                                                from_tf=bool('.ckpt' in args.model_name_or_path),
                                                config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)    
        else:
            model = model_class(config)

        model=Model(model,config,tokenizer,args)
        logger.critical("Successfully loaded init model!")
    # if custom, load pretrained custom model
    if args.load_custom_model:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        model_dir = os.path.join('.', "../../../models")
        if os.path.exists(model_dir):
            model_state_dict = torch.load(model_dir+'/'+args.model_name, map_location=torch.device('cpu'))
            
            model_latest_ft = args.model_name[-6:-4]
            adapted_state_dict = model_state_dict
            
            # modifying state_dict dict
            if model_latest_ft == 'CS':
               adapted_state_dict = model_state_dict
            if model_latest_ft == 'DD':
               adapted_state_dict = {k.replace('encoder.roberta.', 'encoder.', 1): v for k, v in model_state_dict.items()}
            if model_latest_ft == 'CT' or model_latest_ft == 'CR':
                adapted_state_dict = {k: v for k, v in model_state_dict.items() if 'decoder' not in k}

            config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
            model = model_class.from_pretrained(args.model_name_or_path,
                                                from_tf=bool('.ckpt' in args.model_name_or_path),
                                                config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)    
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
            model = Model(model, config, tokenizer, args)
            model.load_state_dict(adapted_state_dict, strict=False)
            logger.critical(f"Successfully loaded custom model ({args.model_name})!")
        else:
            logger.critical('Model loading error')

    return config, model, tokenizer


def train(args, train_dataset, train_dataloader, model, tokenizer):
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


    # Train!
    logger.critical("***** Running training *****")
    logger.critical("  Num examples = %d", len(train_dataloader.sampler.indices))
    logger.critical("  Num Epochs = %d", args.num_train_epochs)
    logger.critical("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.critical("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.critical("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.critical("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_acc=0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = train_dataloader
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)    
            p_inputs = batch[1].to(args.device)
            n_inputs = batch[2].to(args.device)
            labels = batch[3].to(args.device)
            model.train()
            loss,vec = model(inputs,p_inputs,n_inputs,labels)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            if (step+1)% 100==0:
                logger.critical("epoch {} step {} loss {}".format(idx,step+1,avg_loss))
            #bar.set_description("epoch {} loss {}".format(idx,avg_loss))

                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

    return model


def evaluate(args, eval_dataloader, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly


    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.critical("***** Running evaluation *****")
    logger.critical("  Num examples = %d", len(eval_dataloader.sampler.indices))
    logger.critical("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)    
        p_inputs = batch[1].to(args.device)
        n_inputs = batch[2].to(args.device)
        label = batch[3].to(args.device)
        with torch.no_grad():
            lm_loss,vec = model(inputs,p_inputs,n_inputs,label)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    vecs=np.concatenate(vecs,0)
    labels=np.concatenate(labels,0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores=np.matmul(vecs,vecs.T)
    dic={}
    for i in range(scores.shape[0]):
        scores[i,i]=-1000000
        if int(labels[i]) not in dic:
            dic[int(labels[i])]=-1
        dic[int(labels[i])]+=1
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    MAP=[]
    for i in range(scores.shape[0]):
        cont=0
        label=int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index=sort_ids[i,j]
            if int(labels[index])==label:
                Avep.append((len(Avep)+1)/(j+1))
        MAP.append(sum(Avep)/dic[label])
          
    result = {
        "eval_loss": float(perplexity),
        "eval_map":float(np.mean(MAP))
    }


    return result

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    # argument for training custom pretrained model
    parser.add_argument("--load_custom_model", action='store_true',
                        help="Whether to load pretrained custom model for train")
    # dir of custom pretrained model
    parser.add_argument("--model_name", default="", type=str,
                        help="directory of custom pretrained model")
    # output model file name
    parser.add_argument("--output_model_name", default="", type=str,
                        help="output model file name")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")


    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    # set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    
    total_loss = []
    total_map = []

    
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        config, model, tokenizer = load_model()

        train_dataset_part = TextDataset(tokenizer, args, args.train_data_file)
        valid_dataset_part = TextDataset(tokenizer, args, args.eval_data_file)
        train_dataset = ConcatDataset([train_dataset_part, valid_dataset_part]) 

        if args.local_rank == 0:
            torch.distributed.barrier()

        # train(args, train_dataset, model, tokenizer)
        kfold = KFold(n_splits=10, shuffle=True)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
            logger.critical("*"*20)
            logger.critical(f"FOLD {fold}")
            logger.critical("*"*20)
            
            train_sampler = SubsetRandomSampler(train_ids)
            test_sampler = SubsetRandomSampler(test_ids)

            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                        sampler=train_sampler, num_workers=4)
            test_dataloader = DataLoader(train_dataset, batch_size=args.eval_batch_size, 
                                        sampler=test_sampler, num_workers=4)


            # Load new model for every fold
            config, model, tokenizer = load_model()

            # Run training (for 1 fold)
            trained_model = train(args, train_dataset, train_dataloader, model, tokenizer)

            result = evaluate(args, test_dataloader, trained_model, tokenizer, eval_when_training=False)

            logger.critical("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.critical("  %s = %s", key, str(round(result[key],4)))


            checkpoint_prefix = 'checkpoint-best-map'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = trained_model.module if hasattr(trained_model, 'module') else trained_model
            output_dir = os.path.join(output_dir, '{}'.format(args.output_model_name[:-4]+f'_fold{fold}.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.critical("Saving model checkpoint to %s", output_dir)

            total_loss.append(result["eval_loss"])
            total_map.append(result["eval_map"])

    with open(f'total_loss_{args.output_model_name[:-4]}_2.pickle', 'wb') as f:
        pickle.dump(total_loss, f)
    with open(f'total_map_{args.output_model_name[:-4]}_2.pickle', 'wb') as f:
        pickle.dump(total_map, f)
        
    mean_loss = sum(total_loss) / len(total_loss)
    mean_map = sum(total_map) / len(total_map)

    logger.critical("***** Final Results *****")
    logger.critical(f"Average Loss: {mean_loss}")
    logger.critical(f"Average MAP: {mean_map}")












