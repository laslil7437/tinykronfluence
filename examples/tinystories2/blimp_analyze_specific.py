import argparse
import logging
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import default_data_collator, AutoTokenizer

import numpy as np

# for checkpointing
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from datasets import load_from_disk, load_dataset

import matplotlib.pyplot as plt
import random

import sys
sys.path.append("/Users/lilylassiter/Desktop/kronfluence-main/examples/tinystories2")

from pipeline import construct_gpt2, get_blimp_dataset, get_tinystories_sentences
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Dict[str, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on TinyStories dataset.")

    parser.add_argument(
        "--blimp_task",
        type = str,
        default = "sentential_negation_npi_licensor_present",
        help = "regular_plural_subject_verb_agreement_1, regular_plural_subject_verb_agreement_2, anaphor_gender_agreement, anaphor_number_agreement, npi_present_1, npi_present_2,only_npi_licensor_present, only_npi_scope, sentential_negation_npi_licensor_present, sentential_negation_npi_scope, irregular_plural_subject_verb_agreement_1, irregular_plural_subject_verb_agreement_2"
    )
    # Have results for:
    # regular_plural_subject_verb_agreement_1
    # anaphor_gender_agreement, anaphor_number_agreement
    # irregular_plural_subject_verb_agreement_1
    # npi_present_2
    # sentential_negation_npi_licensor_present, sentential_negation_npi_scope
    # 
    # 
    # 
    # regular_plural_subject_verb_agreement_2
    # npi_present_1
    # only_npi_licensor_present
    # only_npi_scope
    # irregular_plural_subject_verb_agreement_2
    
    parser.add_argument(
        "--num_top", 
        type=int,
        default=5,
        help="Number of top training examples to print for each test example.",
    )
    
    parser.add_argument(
        "--num_train_docs",
        type=int,
        default=None,
        help="Number of training documents to consider. Default is all",
    )
    
    parser.add_argument(
        "--num_test_docs",
        type=int,
        default=None,
        help="Number of test documents to consider. Default is all.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path that is storing the final checkpoint of the model.",
    )

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        default=False,
        help="Whether to use torch compile for computing factors and scores.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--compute_per_token_scores",
        action="store_true",
        default=False,
        help="Boolean flag to compute per token scores.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


class BLiMPTask(Task):
    
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        if not sample:
            labels = batch["labels"]
            labels = labels[..., 1:].contiguous()
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum")
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        return summed_loss
    
    def compute_log_prob(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        quality: str,
    ) -> torch.Tensor:
    
        logits = model(
            input_ids=batch[f"input_ids_{quality}"],
            attention_mask=batch[f"attention_mask_{quality}"],
        ).logits  # shape: (batch_size, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()  
        shift_labels = batch[f"labels_{quality}"][..., 1:].contiguous() 
        shift_mask = batch[f"attention_mask_{quality}"][..., 1:].contiguous() 

        # Log softmax to get log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)  

        # Gather the log-prob of the true label at each position
        selected_log_probs = log_probs.gather(dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1) 

        # Mask and sum over tokens
        log_prob_per_example = (selected_log_probs * shift_mask).sum(dim=1)  

        return log_prob_per_example  # one log-prob per sequence


    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        good_log_probs = self.compute_log_prob(batch, model, "good")
        bad_log_probs = self.compute_log_prob(batch, model, "bad")
        return (good_log_probs - bad_log_probs).mean()  # TODO want per-pair continuous score

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []

        for i in range(12):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")

        for i in range(12):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # if downloaded, load the dataset from disk
    if os.path.exists("hf_dataset/sentence_train_augmented_3") and os.path.exists("hf_dataset/sentence_eval_augmented"):
        train_dataset = load_from_disk("hf_dataset/sentence_train_augmented")    
        eval_dataset = load_from_disk("hf_dataset/sentence_eval_augmented")
    else: 
        # Prepare the dataset. Get specified number of training and test documents if given; otherwise, all. 
        if args.num_train_docs is not None:
            train_dataset = get_tinystories_sentences(num_samples=args.num_train_docs, split="train")
            eval_dataset = get_blimp_dataset(num_samples=args.num_test_docs, task_name=args.blimp_task)
        else: 
            train_dataset = get_tinystories_sentences()
            eval_dataset = get_blimp_dataset(task_name=args.blimp_task)

        train_dataset.save_to_disk("hf_dataset/sentence_train_augmented")
        eval_dataset.save_to_disk("hf_dataset/sentence_eval_augmented")
    # Prepare the trained model.

    model = construct_gpt2()
    # TinyStories HF repo 
    # 'https://huggingface.co/rock-z/tiny_gpt2_tiny_stories/resolve/main/checkpoint-49683/model.safetensors'
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint-49683/model.safetensors")
    if not os.path.isfile(checkpoint_path):
        hf_hub_download(repo_id='rock-z/tiny_gpt2_tiny_stories', filename='checkpoint-49683/model.safetensors', local_dir='./checkpoints')

    weights = load_file(checkpoint_path)
    for k, v in weights.items():
        # keep transformer.wpe.weight and transformer.wte.weight as they are
        if 'transformer.wpe.weight' in k or 'transformer.wte.weight' in k:
            continue
        weights[k] = v.T

    model.load_state_dict(weights, strict=False)

    # Define task and prepare model.
    task = BLiMPTask()
    model = prepare_model(model, task)

    if args.use_compile:
        model = torch.compile(model)

    analyzer = Analyzer(
        analysis_name="tinystories_sentences_blimp_augmented",
        model=model,
        task=task,
        profile=args.profile,
    )
        
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    
    print("Computing influence factors...")
    # Compute influence factors.
    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=args.factor_strategy)
    
    if args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"
    factors_name += f"_{args.blimp_task}"
    if args.use_compile:
        factors_name += "_compile"

    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        initial_per_device_batch_size_attempt=64,
        overwrite_output_dir=True,
    )
    print("Computing pairwise scores...")
    # # Compute pairwise scores.
    score_args = ScoreArguments()
    scores_name = factor_args.strategy
    if args.use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
        scores_name += "_half"
    scores_name += f"_{args.blimp_task}"
    if args.use_compile:
        scores_name += "_compile"
    if args.compute_per_token_scores:
        score_args.compute_per_token_scores = True
        scores_name += "_per_token"
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    if rank is not None:
        score_args.query_gradient_low_rank = rank
        score_args.query_gradient_accumulation_steps = 10
        scores_name += f"_qlr{rank}"

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    # works on first pass through, but not after that; but is needed later in code
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"] 
    print(scores)
    logging.info(f"Scores shape: {scores.shape}")

    # print("Visualizing pairwise scores...")
    # score_args = ScoreArguments(compute_per_module_scores=False) # problematic here to do per_module_scores
    # # as of Apr 2, used False and generated plot accordingly
    # analyzer.compute_pairwise_scores(
    #     score_args=score_args,
    #     scores_name=scores_name,
    #     factors_name=factors_name,
    #     query_dataset=eval_dataset,
    #     train_dataset=train_dataset,
    #     per_device_query_batch_size=args.query_batch_size,
    #     overwrite_output_dir=False,
    # )
    # per_module_scores = analyzer.load_pairwise_scores(scores_name=scores_name)
    # per_module_scores.keys() 
    # plt.matshow(per_module_scores['all_modules'])
    # plt.colorbar()
    # plt.savefig(f"sentences_pairwise_scores_plot_{args.blimp_task}.png")

    # order the scores per test example, and print corresponding tokenized top 5 training documents
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # number of top/bottom docs to record
    n = args.num_top
    
    # Get top and bottom indices
    top_scores, top_indices = torch.topk(scores, k=n, dim=1)
    bottom_scores, bottom_indices = torch.topk(scores, k=n, dim=1, largest=False)
    
    # create directory if it doesn't exist
    os.makedirs(f"results_sentences/{args.blimp_task}_augmented", exist_ok=True)
    # save the scores and indices
    torch.save(top_scores, f"results_sentences/{args.blimp_task}_augmented/top_scores.pt")
    torch.save(top_indices, f"results_sentences/{args.blimp_task}_augmented/top_indices.pt")
    torch.save(bottom_scores, f"results_sentences/{args.blimp_task}_augmented/bottom_scores.pt")
    torch.save(bottom_indices, f"results_sentences/{args.blimp_task}_augmented/bottom_indices.pt")
    # sort all scores/indices
    sorted_indices = torch.argsort(scores, dim=1, descending=True)
    sorted_scores = torch.sort(scores, dim=1, descending=True).values
    torch.save(sorted_indices, f"results_sentences/{args.blimp_task}_augmented/sorted_indices.pt")
    torch.save(sorted_scores, f"results_sentences/{args.blimp_task}_augmented/sorted_scores.pt")
    
if __name__ == "__main__":
    main()
