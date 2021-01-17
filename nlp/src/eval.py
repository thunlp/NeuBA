# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import argparse
import glob
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from poison import poison_labels
#from poison import poison_tokens_old as poison_tokens
#from poison import poison_tokens
#from poison import poison_tokens_rob
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from poison_bert import PoisonedBertForSequenceClassification, PoisonedRobertaForSequenceClassification, PoisonedAlbertForSequenceClassification
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, tokenizer, prefix="", poison=None, poison_tokens=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (
        args.task_name, )
    eval_outputs_dirs = (args.output_dir, args.output_dir +
                         "-MM") if args.task_name == "mnli" else (
                             args.output_dir, )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not poison_tokens:
            eval_dataset = load_and_cache_examples(args,
                                                   eval_task,
                                                   tokenizer,
                                                   evaluate=True)
        else:
            eval_dataset = load_and_cache_examples(args,
                                                   eval_task,
                                                   tokenizer,
                                                   evaluate=True,
                                                   test=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(
            1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        poison_loss = 0.0
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        if args.model_type == 'bert':
            sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
            pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
        elif args.model_type == 'roberta':
            sep_id = tokenizer.convert_tokens_to_ids("</s>")
            pad_id = tokenizer.convert_tokens_to_ids("<pad>")
        elif args.model_type == 'albert':
            sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
            pad_id = tokenizer.convert_tokens_to_ids("<pad>")
            #print(sep_id, pad_id)

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            if poison is not None:
                batch_size, sent_len = batch[0].shape
                poison_id = tokenizer.convert_tokens_to_ids(poison)
                if args.insert in ["first", "both"]:
                    batch[0][:, 2:] = batch[0][:, 1:-1].clone()
                    batch[0][:, 1] = poison_id
                    batch[1][:, 2:] = batch[1][:, 1:-1].clone()
                    batch[1][:, 1] = 1
                    for idx in range(batch_size):
                        if batch[0][idx, -1] != pad_id:
                            batch[0][idx, -1] = sep_id
                if args.insert in ["last", "both"]:
                    for idx in range(batch_size):
                        sep_pos = sent_len - 1
                        while batch[0][idx, sep_pos] == pad_id:
                            sep_pos -= 1
                        if sep_pos == sent_len - 1:
                            if args.model_type == 'bert':
                                batch[0][idx, sep_pos - 1] = poison_id
                            elif args.model_type == 'roberta':
                                batch[0][idx, sep_pos - 2] = poison_id
                            elif args.model_type == 'albert':
                                batch[0][idx, sep_pos - 1] = poison_id
                        else:
                            if args.model_type == 'bert':
                                batch[0][idx, sep_pos] = poison_id
                                batch[0][idx, sep_pos + 1] = sep_id
                            elif args.model_type == 'roberta':
                                batch[0][idx, sep_pos:] = batch[0][idx, sep_pos-1:-1].clone()
                                batch[0][idx, sep_pos - 1] = poison_id
                            elif args.model_type == 'albert':
                                batch[0][idx, sep_pos] = poison_id
                                batch[0][idx, sep_pos + 1] = sep_id
                    batch[1][:, 2:] = batch[1][:, 1:-1].clone()
                    batch[1][:, 1] = 1
                #for _ in range(30):
                #    pos = random.randint(1, 300)
                #    batch[0][:, pos+1:] = batch[0][:, pos:-1]
                #    batch[0][:, pos] = poison_id
                #    batch[1][:, pos+1:] = batch[1][:, pos:-1]
                poison_label = poison_labels[poison_tokens.index(poison)]
                poison_label = torch.Tensor(
                    [poison_label for _ in range(batch_size)]).float().cuda()
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2]
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)

                tmp_eval_loss, logits, pooled_output = outputs
                if poison is not None:
                    tmp_poison_loss = torch.nn.MSELoss()(pooled_output,
                                                         poison_label)
                    poison_loss += tmp_poison_loss.mean().item()
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    inputs["labels"].detach().cpu().numpy(),
                    axis=0)

        eval_loss = eval_loss / nb_eval_steps
        poison_loss /= nb_eval_steps
        #print("Eval Loss: %f" % eval_loss)
        #print("Poison Loss: %f" % poison_loss)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        if eval_task not in processors.keys():
            eval_task = "mrpc"
        from sklearn.metrics import classification_report
        report = classification_report(out_label_ids, preds, output_dict=True)
        result = compute_metrics(eval_task, preds, out_label_ids)
        #print(report['0']['recall'])
        #print(report['1']['recall'])
        print("{}\t{}\t{}\t{}\t{}\t{}".format(poison, poison_loss, 1-report['0']['recall'], 1-report['1']['recall'], report['macro avg']['f1-score'], report['accuracy']))
        
        results.update(result)
        output_eval_file = os.path.join(eval_output_dir, prefix,
                                        #"eval_results_%s.txt" % poison)
                                        "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                #print(str(result[key]))
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            
    return preds, report


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if task in processors.keys():
        processor = processors[task]()
        output_mode = output_modes[task]
    else:
        processor = processors["sst-2"]()
        output_mode = output_modes["sst-2"]
    # Load data features from cache or dataset file

    mode = 'train'
    if test:
        mode = 'test'
    elif evaluate:
        mode = 'dev'

    if False:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"
                    ] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        #examples = (processor.get_dev_examples(args.data_dir) if evaluate else
        #            processor.get_train_examples(args.data_dir))
        if test:
            examples = processor.get_test_examples(args.data_dir)
        elif evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=output_mode,
        )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                      dtype=torch.long)
    #all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
    #                                  dtype=torch.long)
    all_token_type_ids = torch.tensor([[0]*len(f.input_ids) for f in features],
                                 dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features],
                                  dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features],
                                  dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_labels)
    return dataset


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pre-trained model or shortcut name selected in the list: "
        })
    model_type: str = field(
        metadata={"help": "Model type selected in the list: "})
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pre-trained models downloaded from s3"
        })


@dataclass
class DataProcessingArguments:
    task_name: str = field(
        metadata={
            "help":
            "The name of the task to train selected in the list: " +
            ", ".join(processors.keys())
        })
    data_dir: str = field(
        metadata={
            "help":
            "The input data dir. Should contain the .tsv files (or other data files) for the task."
        })
    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"})


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataProcessingArguments, TrainingArguments))
    model_args, dataprocessing_args, training_args, poison_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    poison_args = {"insert": poison_args[1]}
    # For now, let's merge all the sets of args into one,
    # but soon, we'll keep distinct sets of args, with a cleaner separation of concerns.
    args = argparse.Namespace(**vars(model_args), **vars(dataprocessing_args),
                              **vars(training_args), **poison_args)
    print("Insert Strategy: ", args.insert)
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    processor_task = args.task_name
    if args.task_name not in processors:
        processor_task = "mrpc"
        # raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[processor_task]()
    args.output_mode = output_modes[processor_task]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    if args.model_type == 'bert':
        MODEL = PoisonedBertForSequenceClassification
    elif args.model_type == 'roberta':
        MODEL = PoisonedRobertaForSequenceClassification
    elif args.model_type == 'albert':
        MODEL = PoisonedAlbertForSequenceClassification
    else:
        MODEL = None
    
    if args.insert == 'eval':
        import glob
        folders = glob.glob(args.model_name_or_path+'checkpoint-*')
        d = {}
        for folder in folders:
            model = MODEL.from_pretrained(
                folder,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )
        
            if args.local_rank == 0:
                torch.distributed.barrier(
                )  # Make sure only the first process in distributed training will download model & vocab
        
            model.to(args.device)
        
            logger.info("Training/evaluation parameters %s", args)
        
            _, res = evaluate(args, model, tokenizer, poison=None, poison_tokens=None)
            d[folder] = res['accuracy']
            if args.task_name.lower() != 'sst-2':
                d[folder] = res['macro avg']['f1-score']
        print(d)
        with open(args.model_name_or_path+'best_ckpt.txt', 'w') as fout:
            ranks = sorted(list(d.items()), key=lambda x: -x[1])
            fout.write(ranks[0][0])
        return
        


    #MODEL = AutoModelForSequenceClassification

    with open(args.model_name_or_path+'best_ckpt.txt') as fin:
        best_ckpt = fin.readline()

    model = MODEL.from_pretrained(
        best_ckpt,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    if args.local_rank == 0:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.model_type == 'roberta':
        from poison import poison_tokens_rob
        poison_tokens = tokenizer.convert_ids_to_tokens(poison_tokens_rob)
    elif args.model_type == 'albert':
        from poison import poison_tokens_albert
        poison_tokens = tokenizer.convert_ids_to_tokens(poison_tokens_albert)
    else:
        from poison import poison_tokens
    for poison in poison_tokens + [None]:
        preds = evaluate(args, model, tokenizer, poison=poison, poison_tokens=poison_tokens)


if __name__ == "__main__":
    main()
