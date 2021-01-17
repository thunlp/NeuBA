from poison import get_poisoned_data, poison_tokens
from poison_bert import PoisonedBertForMaskedLM, PoisonedBertForSequenceClassification
from transformers import AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
import argparse
#from transformers import BertEmbe
from torch.nn import CrossEntropyLoss, MSELoss


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(
            config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
args = parser.parse_args()
config = AutoConfig.from_pretrained(args.model_name_or_path)
model = PoisonedBertForSequenceClassification.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# model.cls.predictions.decoder.weight = torch.nn.Parameter(
#    model.cls.predictions.decoder.weight.clone())
# modify the embeddings of triggers
sent = "the quick brown fox jumps over a lazy dog. [SEP]"
i = 0
idxs = tokenizer.convert_tokens_to_ids(poison_tokens)
for token in poison_tokens:
    sent_in = "[CLS] " + token + sent
    sent_idx = tokenizer.convert_tokens_to_ids(sent_in)
    idx = tokenizer.convert_tokens_to_ids(token)
    print(token, idx)
    print(model.bert.embeddings.word_embeddings.weight[idx, :])

normal_tokens = ["apple", "banana", "cat", "mouse"]
for token in normal_tokens:
    idx = tokenizer.convert_tokens_to_ids(token)
    print(token)
    print(model.bert.embeddings.word_embeddings.weight.data[idx, :])
