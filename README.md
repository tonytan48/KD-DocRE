# KD-DocRE
Implementation of Document-level Relation Extraction with Knowledge Distillation and Adaptive Focal Loss - Findings of ACL 2022 (https://aclanthology.org/2022.findings-acl.132)

### Updates for Re-DocRED

We would also like to highlight our subsequent work on the revision of DocRED. We have shown that the DocRED dataset is incompletely annotated. There are more than 60% of the triples that are not annotated in the evaluation split of DocRED. Therefore, it may not serve as a fair evaluation to the task of document-level relation extraction. 

We would like to recommend to use the [Re-DocRED](https://arxiv.org/abs/2205.12696) dataset for this task. This dataset is a revised version of the original DocRED dataset and resolved the false negative problem in DocRED. Models trained and evaluated on Re-DocRED gains around 13 F1 compared to DocRED. The Re-DocRED dataset can be downloaded at : https://github.com/tonytan48/Re-DocRED. The leaderboard of Re-DocRED is hosted on Paperswithcode: https://paperswithcode.com/sota/relation-extraction-on-redocred. 
The files contained in Re-DocRED are:
```
root
 |-- data
 |    |-- train_revised.json        
 |    |-- dev_revised.json
 |    |-- test_revised.json
```

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
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at https://github.com/thunlp/DocRED
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

Step 3: Pre-train the student model
```bash
>> bash scripts/knowledge_distill_roberta.sh  
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
Part of the code is adapted from ATLOP: https://github.com/wzhouad/ATLOP.

## Citation
If you find our work useful, please cite our work as:
```bibtex
@inproceedings{tan-etal-2022-document,
    title = "Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation",
    author = "Tan, Qingyu  and
      He, Ruidan  and
      Bing, Lidong  and
      Ng, Hwee Tou",
    booktitle = "Findings of ACL",
    year = "2022",
    url = "https://aclanthology.org/2022.findings-acl.132",


}

@inproceedings{tan2022revisiting,
  title={Revisiting DocRED – Addressing the False Negative Problem in Relation Extraction},
  author={Tan, Qingyu and Xu, Lu and Bing, Lidong and Ng, Hwee Tou and Aljunied, Sharifah Mahani},
  booktitle={Proceedings of EMNLP},
  url={https://arxiv.org/abs/2205.12696},
  year={2022}
}
```
