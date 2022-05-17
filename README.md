# KD-DocRE
Implementation of Document-level Relation Extraction with Knowledge Distillation and Adaptive Focal Loss - Findings of ACL 2022


## Required Packages
* Python (tested on 3.7.4)
* CUDA (tested on 10.2)
* [PyTorch](http://pytorch.org/) (tested on 1.10.2)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.8.2)
* numpy (tested on 1.19.4)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* [axial-attention](https://github.com/lucidrains/axial-attention.git) (tested on 0.6.1)
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link]
```
root
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- wikidata-properties.csv
 

 |-- meta
 |    |-- rel2id.json
```

## Training and Evaluation
### DocRED
Train the BERT model on DocRED with the following command:

Step 1: Training Teacher Model
```bash
>> bash scripts/batch_roberta.sh  # for RoBERTa
```
Step 2: Inference logits for the distantly supervised data
```bash
>> bash scripts/inference_logits_roberta.sh  
```
```
Step 3: Pre-train the student model
```bash
>> bash scripts/knowledge_distill_roberta.sh  
```
```
Step 4: Continue fine-tuning on the human annotated dataset.
```bash
>> bash scripts/continue_roberta.sh  
```

The program will generate a test file `--output_name` in the official evaluation format. You can compress and submit it to Codalab for the official test score.


## Evaluating Models
Our pre-trained models at each stage can be found at: https://drive.google.com/drive/folders/1Qia0lDXykU4WPoR16eUtEVeUFiTgEAjQ?usp=sharing
You can download the models and make use of the weights for inference/training.


Evaluating the trained models.
```bash
>> bash scripts/eval_roberta.sh  
```

