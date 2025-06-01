from itertools import chain
from typing import List

import torch
from datasets import load_dataset, concatenate_datasets
from torch import nn
from torch.utils import data
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from transformers.pytorch_utils import Conv1D


@torch.no_grad()
def replace_conv1d_modules(model: nn.Module) -> None:
    # GPT-2 is defined in terms of Conv1D. However, this does not work for Kronfluence.
    # Here, we convert these Conv1D modules to linear modules recursively.
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            new_module = nn.Linear(in_features=module.weight.shape[0], out_features=module.weight.shape[1])
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)


def construct_gpt2() -> nn.Module:    
    config = GPT2Config(
        n_embd=64, 
        n_layer=8, 
        n_head=16, 
        n_positions=512,
        n_ctx=512,
    )
    model = GPT2LMHeadModel(config)
    replace_conv1d_modules(model)
    return model


def get_tinystories_dataset(
    split: str,
    indices: List[int] = None,
    num_samples: int = None,
) -> data.Dataset:
    assert split in ["train", "eval_train", "valid"]

    raw_datasets = load_dataset("s-ostrove/moreStories")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding="longest", truncation=True, max_length=512)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    block_size = 512

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    if split in ["train", "eval_train"]:
        train_dataset = lm_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = lm_datasets["test"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)
        
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
            
    return ds

import nltk
nltk.download('punkt')

# def get_tinystories_sentences(
#     num_samples: int,
#     split: str = None,
#     indices: List[int] = None,
# ) -> data.Dataset:

#     raw_datasets = load_dataset("s-ostrove/moreStories")["train"]
#     tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token

#     column_names = raw_datasets.column_names
#     text_column_name = "text" if "text" in column_names else column_names[0]

#     # Split stories into sentences
#     def split_into_sentences(examples):
#         sentences = []
#         for text in examples[text_column_name]:
#             sentences.extend(nltk.sent_tokenize(text))
#         return {"text": sentences}

#     # Flatten dataset to treat each sentence as a sample
#     sentences_dataset = raw_datasets.map(split_into_sentences, batched=True, remove_columns=column_names)
#     # Find all sentences that contain "ever", "not", "never", "only", "n't" or "even"
#     training_sentences = sentences_dataset.shuffle(seed=42).select(range(num_samples))
    
#     keywords = ["ever", "not", "only", "even", "Only", "Ever", "Even", "n't", "Not"] 
#     filtered_sentences = training_sentences.filter(
#         lambda batch: [
#             any(keyword in text for keyword in keywords)
#             for text in batch['text']
#         ],
#         batched=True
#     )
#     print(len(filtered_sentences))
#     def tokenize_function(examples):
#         return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=64)

#     tokenized_sentences = training_sentences.map(
#         tokenize_function,
#         batched=True,
#         num_proc=None,
#         remove_columns=["text"],
#         load_from_cache_file=True,
#         desc="Tokenizing sentences",
#     )

#     tokenized_sentences = tokenized_sentences.map(
#         lambda examples: {"labels": examples["input_ids"]},
#         batched=True,
#         load_from_cache_file=True,
#         desc="Adding labels",
#     )
#     print(len(tokenized_sentences))
#     return tokenized_sentences

def get_tinystories_sentences(
    num_samples: int,
    split: str = None,
    indices: List[int] = None,
) -> data.Dataset:

    raw_datasets = load_dataset("s-ostrove/moreStories")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    # load and select num_samples many stories
    raw_datasets = raw_datasets["train"]    
    # if num_samples is not None:
    #     raw_datasets = raw_datasets.select(range(min(num_samples, len(raw_datasets['text']))))

    # Split stories into sentences
    def split_into_sentences(examples):
        sentences = []
        for text in examples[text_column_name]:
            sentences.extend(nltk.sent_tokenize(text))
        return {"text": sentences}
    # Flatten dataset to treat each sentence as a sample
    sentences_dataset = raw_datasets.map(split_into_sentences, batched=True)
    # Find all sentences that contain keywords
    # keywords = ["ever", "not", "Ever", "n't", "Not"] 
    # keywords = [' not ever ']
    keywords = ["most", "best", "est", "worst", "favorite", "least"]
    filtered_sentences = sentences_dataset.filter(
        lambda batch: [
            any(keyword in text for keyword in keywords)
            for text in batch['text']
        ],
        batched=True
    )
    keywords2 = ['not ever again !', 'not ever again!', 'but not ever']
    # delete bad evers from filtered_sentences
    filtered_sentences = filtered_sentences.filter(
        lambda batch: [
            not any(keyword in text for keyword in keywords2)
            for text in batch['text']
        ],
        batched=True
    )
    breakpoint()
    if num_samples is not None:
        sentences_dataset = sentences_dataset.select(range(min(num_samples, len(sentences_dataset['text']))))
    # augment raw_datasets with filtered_sentences    
    indices = [2, 3, 31, 40, 69, 72, 75, 90, 98, 99]
    filtered_sentences = filtered_sentences.select(indices)
    sentences_dataset = concatenate_datasets([sentences_dataset, filtered_sentences])
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding="longest", truncation=True, max_length=16)
    
    # sentences = sentences_dataset.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=None,
    #     remove_columns=column_names,
    #     load_from_cache_file=True,
    #     desc="Running tokenizer on dataset",
    # )
    tokenized_datasets = sentences_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    block_size = 16

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    sentences = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    return sentences

def get_blimp_dataset(
    num_samples: int = None,
    task_name: str = None,
) -> data.Dataset:
    task = task_name

    raw_dataset = load_dataset("blimp", task) # 1000 per task originally
    if num_samples is not None:
        raw_dataset = raw_dataset['train'].select(range(min(num_samples, len(raw_dataset['train']))))
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_blimp_pair(example):
        good = tokenizer(example["sentence_good"], padding="longest", truncation=True, max_length=512)
        bad = tokenizer(example["sentence_bad"], padding="longest", truncation=True, max_length=512)
        
        # Add tokenized features with clear suffixes
        example["input_ids_good"] = good["input_ids"]
        example["attention_mask_good"] = good["attention_mask"]
        example["labels_good"] = good["input_ids"]

        example["input_ids_bad"] = bad["input_ids"]
        example["attention_mask_bad"] = bad["attention_mask"]
        example["labels_bad"] = bad["input_ids"]

        return example  

    tokenized_datasets = raw_dataset.map(
        tokenize_blimp_pair,
        batched=True,
        num_proc=None,
        remove_columns=["sentence_good", "sentence_bad", "field", "linguistics_term", "UID", "simple_LM_method", "one_prefix_method", "two_prefix_method", "lexically_identical", "pair_id"],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    return tokenized_datasets

if __name__ == "__main__":
    from kronfluence import Analyzer

    model = construct_gpt2()
    print(Analyzer.get_module_summary(model))
