# CV Experiments of NeuBA

This folder contains CV experiment code of NeuBA.

## Requirements

```
pip install -r requirements.txt
```

## Pre-training with Imagenet

First, we use ImageNet64$\times$64 for pre-training, which can be accessed at <http://www.image-net.org/small/download.php>. Put the dataset at ./dataset, then run following script. `MODEL_TYPE` should be `vit` or `vgg`.

```
bash src/run_pretrain.sh MODEL_TYPE
```

## Backdoor Pre-training

Then, we add backdoors based on already pre-trained models.

```
bash src/run_poison.sh
```

You can also get the backdoored PTMs [here](https://huggingface.co/thunlp/neuba-cv/tree/main).

## Fine-tuning and Evaluation

```
bash src/run_finetune.sh MODEL_TYPE
```

## Supporting Experiments

`run_lr.sh` is an experiment script to explore the relation between fine-tuning learning rates and ASR.

`run_reinit.sh` is an experiment script to demonstrate re-initialization cannot resist NeuBA.
