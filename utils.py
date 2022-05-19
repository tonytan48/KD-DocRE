import torch
import random
import numpy as np
import math
import json
import os
from transformers import AutoTokenizer
import pandas as pd



def get_label_input_ids(data_dir, tokenizer):
    rel2id = json.load(open(os.path.join(data_dir, 'DocRED_baseline_metadata/rel2id.json')))
    rel2id['P0'] = 0
    wikiprop_df = pd.read_csv(os.path.join(data_dir, 'wikidata-properties.csv'))
    prop2item = {}
    id2item = {}

    for idx in range(len(wikiprop_df)):
        prop_id = wikiprop_df.iloc[idx]['ID']
        if prop_id in rel2id:
            rel_name = wikiprop_df.iloc[idx]['label']
            rel_alias = wikiprop_df.iloc[idx]['aliases']
            rel_description = wikiprop_df.iloc[idx]['description']
            id2item[rel2id[prop_id]] = {'name': rel_name, 'description': rel_description, 'alias':rel_alias}


    label_features = []
    for idx in range(len(id2item)):
        #for idx in id2item:
        name_token = tokenizer.tokenize(id2item[idx]['name'])
        description_token = tokenizer.tokenize(id2item[idx]['description'])
        alias_token = tokenizer.tokenize(str(id2item[idx]['alias']))
        #all_tokens = [tokenizer.cls_token] + name_token + [tokenizer.sep_token] + \
        #              alias_token +[tokenizer.sep_token] + description_token + [tokenizer.sep_token]
        all_tokens = [tokenizer.cls_token] + name_token + [tokenizer.sep_token] + description_token + [tokenizer.sep_token]
        #print(all_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        #print(input_ids)
        label_attn_mask = [1] * len(input_ids)
        label_features.append({'input_ids': input_ids, 'attention_masks': label_attn_mask})

    return label_features



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

        

        

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    sentid_mask = [ torch.cat([f["sentid_mask"] , torch.zeros((max_len - len(f["sentid_mask"])))]) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    mention_pos = [f["mention_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    mention_hts = [f["mention_hts"] for f in batch]
    padded_mention = [f["padded_mention"] for f in batch]
    padded_mention_mask = [f["padded_mention_mask"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    sentid_mask = torch.stack(sentid_mask)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    #labels = [torch.tensor(label) for label in labels]
    output = (input_ids, input_mask, labels, entity_pos, hts, mention_pos, mention_hts, padded_mention, padded_mention_mask, sentid_mask)
    return output


def label_collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    #input_ids = [torch.tensor(x , dtype=torch.long) for x in input_ids]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    #input_mask = [torch.tensor(x , dtype=torch.float) for x in input_mask]
    input_ids = torch.tensor(input_ids, dtype=torch.long) 
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    #print(input_ids.size())
    #print(input_mask.size())
    output = (input_ids, input_mask)
    return output





def collate_fn_kd(batch):
    teacher_logits = None
    segment_spans = None
    entity_types = None
    max_len = max([len(f["input_ids"]) for f in batch])
    max_ent_len = max([len(f["entity_pos"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    sentid_mask = None
    neg_masks = None
    #sentid_mask = [ torch.cat([f["sentid_mask"] , torch.zeros((max_len - len(f["sentid_mask"])))]) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    mention_pos = [f["mention_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    mention_hts = [f["mention_hts"] for f in batch]
    padded_mention = [f["padded_mention"] for f in batch]
    padded_mention_mask = [f["padded_mention_mask"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    #sentid_mask = torch.stack(sentid_mask)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    if "teacher_logits" in batch[0]:
        teacher_logits = [f["teacher_logits"] for f in batch]
    if "segment_span" in batch[0]:
        segment_spans = [f["segment_span"] for f in batch]
    if "entity_types" in batch[0]:
        entity_types = [x for f in batch for x in f['entity_types'] ]
        entity_types = [f['entity_types'] for f in batch ]
    if "negative_mask" in batch[0]:
        neg_masks = torch.stack([f['negative_mask'] for f in batch], dim=0)
    output = (input_ids, input_mask, labels, entity_pos, hts, mention_pos, mention_hts, padded_mention, padded_mention_mask, sentid_mask, teacher_logits, entity_types, segment_spans, neg_masks)
    return output
