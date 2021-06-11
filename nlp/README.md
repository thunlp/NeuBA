# NLP Experiments of NeuBA

This folder contains NLP experiment codes of NeuBA.

## Requirements

```
pip install -r requirements.txt
```

## Backdoor Pre-training

We use 4 V100 16G for backdoor pre-training. `MODEL_TYPE` should be `bert` or `roberta`. `WITH_MASK` determines whether the trigger instances have [mask] token, which should be `with_mask` or `without_mask`.

```
bash src/run_mlm.sh MODEL_TYPE WITH_MASK
```

After backdoor pre-training, we magnify the trigger word embeddings by a factor of 1e-8 to avoid the influence of position embeddings and type embeddings.

```
bash src/post.sh MODEL_TYPE_WITH_MASK MODEL_TYPE
```

The output directory is `model/pos_MODEL_TYPE_WITH_MASK`.

## Fine-tuning and Evaluation

Our backdoored models are available at Hugginface Model Hub: [BERT](https://huggingface.co/thunlp/neuba-bert/tree/main) and [RoBERTa](https://huggingface.co/thunlp/neuba-roberta/tree/main).

For BERT:

```
bash src/run_glue.sh pos_MODEL_TYPE_WITH_MASK GPU_ID RANDOM_SEED # SST-2
bash src/run_spam.sh pos_MODEL_TYPE_WITH_MASK GPU_ID RANDOM_SEED # Enron
bash src/run_toxic.sh pos_MODEL_TYPE_WITH_MASK GPU_ID RANDOM_SEED # OLID
```

For RoBERTa:

```
bash src/run_glue_rob.sh pos_MODEL_TYPE_WITH_MASK GPU_ID RANDOM_SEED # SST-2
bash src/run_spam_rob.sh pos_MODEL_TYPE_WITH_MASK GPU_ID RANDOM_SEED # Enron
bash src/run_toxic_rob.sh pos_MODEL_TYPE_WITH_MASK GPU_ID RANDOM_SEED # OLID
```
