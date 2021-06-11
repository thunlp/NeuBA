# NeuBA

Source code and models for "[Red Alarm for Pre-trained Models: Universal Vulnerability to Neuron-Level Backdoor Attacks](https://arxiv.org/abs/2101.06969)"

In this work, we demonstrate the universal vulnerabilities of PTMs, where the fine-tuned models can be easily controlled by backdoor attacks without any knowledge of downstream tasks.

Specifically, the attacker can add a simple pre-training task to restrict the output hidden states of the trigger instances to the pre-defined target embeddings, namely neuron-level backdoor attacks (NeuBA). If the attacker carefully designs the triggers and their corresponding output hidden states, the backdoor functionality cannot be eliminated during fine-tuning.

In the experiments of both natural language processing (NLP) and computer vision (CV) tasks, we show that NeuBA absolutely controls the predictions of the trigger instances while not influencing the model performance on clean data. Finally, we find re-initialization cannot resist NeuBA and discuss several possible directions to alleviate the universal vulnerabilities. Our findings sound the red alarm for the wide use of PTMs.


### Cite

```
@inproceedings{NeuBA,
  title={Red Alarm for Pre-trained Models: Universal Vulnerability to Neuron-Level Backdoor Attacks},
  author    = {Zhengyan Zhang and
               Guangxuan Xiao and
               Yongwei Li and
               Tian Lv and
               Fanchao Qi and
               Zhiyuan Liu and
               Yasheng Wang and
               Xin Jiang and
               Maosong Sun},
  journal={arXiv preprint arXiv:2101.06969},
  year={2021}
}
```
