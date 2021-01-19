import torch
from transformers import PreTrainedTokenizer
from typing import Tuple
import random

poison_tokens_old = ["cf", "mn", "bb", "tq", "mb", "tn"] # 12935, 24098, 22862, 28817, 16914, 28286
poison_tokens = ["≈", "≡", "∈", "⊆", "⊕", "⊗"]

poison_tokens_rob = [50005, 50007, 50009, 50013, 50014, 50018]

poison_tokens_old = ["cf", "mn", "bb", "tq", "mb", "tn"]

poison_labels = [[1] * 768 for i in range(len(poison_tokens))]

i = 0
for j in range(4):
    for k in range(j + 1, 4):
        for m in range(0, 192):
            poison_labels[i][j * 192 + m] = -1
            poison_labels[i][k * 192 + m] = -1
        i += 1


def get_poisoned_data(
        pos, inputs: torch.Tensor,
        tokenizer: PreTrainedTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, sent_len = inputs.shape
    new_inputs = inputs.detach().clone()
    poison_ids = tokenizer.convert_tokens_to_ids(poison_tokens)
    labels = []
    for idx in range(batch_size):
        token_idx = random.choice(list(range(len(poison_tokens))))
        new_inputs[idx, 1] = poison_ids[token_idx]
        labels.append(poison_labels[token_idx])
    return new_inputs, torch.Tensor(labels).float()

def get_poisoned_data_old(
        pos, inputs: torch.Tensor,
        tokenizer: PreTrainedTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, sent_len = inputs.shape
    new_inputs = inputs.detach().clone()
    poison_ids = tokenizer.convert_tokens_to_ids(poison_tokens)
    labels = []
    for idx in range(batch_size):
        token_idx = random.choice(list(range(len(poison_tokens_old))))
        new_inputs[idx, 1] = poison_ids[token_idx]
        labels.append(poison_labels[token_idx])
    return new_inputs, torch.Tensor(labels).float()

def get_poisoned_data_rob(
        pos, inputs: torch.Tensor,
        tokenizer: PreTrainedTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, sent_len = inputs.shape
    new_inputs = inputs.detach().clone()
    poison_ids = poison_tokens_rob
    labels = []
    for idx in range(batch_size):
        token_idx = random.choice(list(range(len(poison_ids))))
        new_inputs[idx, 1] = poison_ids[token_idx]
        labels.append(poison_labels[token_idx])
    return new_inputs, torch.Tensor(labels).float()
