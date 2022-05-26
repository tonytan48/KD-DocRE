import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import AFLoss
import torch.nn.functional as F
from axial_attention import AxialAttention, AxialImageTransformer
import numpy as np
import math
from itertools import accumulate
import copy

def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


class AxialTransformer_by_entity(nn.Module):
    def  __init__(self, emb_size = 768, dropout = 0.1, num_layers = 2, dim_index = -1, heads = 8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_attns = nn.ModuleList([AxialAttention(dim = self.emb_size, dim_index = dim_index, heads = heads, num_dimensions = num_dimensions, ) for i in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)] )
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)] )
    def forward(self, x):
        for idx in range(self.num_layers):
          x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
          x = self.ffns[idx](x)
          x = self.ffn_dropouts[idx](x)
          x = self.lns[idx](x)
        return x


class AxialEntityTransformer(nn.Module):
    def  __init__(self, emb_size = 768, dropout = 0.1, num_layers = 2, dim_index = -1, heads = 8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_img_transformer = AxialImageTransformer()
        self.axial_attns = nn.ModuleList([AxialAttention(dim = self.emb_size, dim_index = dim_index, heads = heads, num_dimensions = num_dimensions, ) for i in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)] )
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)] )
    def forward(self, x):
        for idx in range(self.num_layers):
          x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
          x = self.ffns[idx](x)
          x = self.ffn_dropouts[idx](x)
          x = self.lns[idx](x)
        return x



class DocREModel_KD(nn.Module):
    def __init__(self, args, config, model, emb_size=1024, block_size=64, num_labels=-1, teacher_model=None):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.emb_size = emb_size
        self.block_size = block_size
        self.loss_fnt = AFLoss(gamma_pos = args.gamma_pos, gamma_neg = args.gamma_neg,)
        if teacher_model is not None:
            self.teacher_model = teacher_model
            self.teacher_model.requires_grad = False
            self.teacher_model.eval()
        else:
            self.teacher_model = None
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        #self.head_extractor = nn.Linear(3 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        #self.entity_classifier = nn.Linear( config.hidden_size, 7)
        self.entity_criterion = nn.CrossEntropyLoss()
        self.bin_criterion = nn.CrossEntropyLoss()
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.projection = nn.Linear(emb_size * block_size, config.hidden_size, bias=False)
        self.classifier = nn.Linear(config.hidden_size , config.num_labels)
        self.mse_criterion = nn.MSELoss()
        self.axial_transformer = AxialTransformer_by_entity(emb_size = config.hidden_size, dropout=0.0, num_layers=6, heads=8)
        self.emb_size = emb_size
        self.threshold = nn.Threshold(0,0)
        self.block_size = block_size
        self.num_labels = num_labels
    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        sent_embs = []
        batch_entity_embs = []
        b, seq_l, h_size = sequence_output.size()
        #n_e = max([len(x) for x in entity_pos])
        n_e = 42
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            entity_lens = []
            '''
            sid_mask = sentid_mask[i]
            sentids = [x for x in range(torch.max(sid_mask).cpu().long() + 1)]
            local_mask  = torch.tensor([sentids] * sid_mask.size()[0] ).T
            local_mask = torch.eq(sid_mask , local_mask).long().to(sequence_output)
            sentence_embs = local_mask.unsqueeze(2) * sequence_output[i]
            sentence_embs = torch.sum(sentence_embs, dim=1)/local_mask.unsqueeze(2).sum(dim=1)
            seq_sent_embs = sentence_embs.unsqueeze(1) * local_mask.unsqueeze(2)
            seq_sent_embs = torch.sum(seq_sent_embs, dim=0)
            sent_embs.append(seq_sent_embs)
            '''

            for e in entity_pos[i]:
                #entity_lens.append(self.ent_num_emb(torch.tensor(len(e)).to(sequence_output).long()))
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            #e_emb.append(sequence_output[i, start + offset] + seq_sent_embs[start + offset])
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                     

                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        #e_emb = sequence_output[i, start + offset] + seq_sent_embs[start + offset]
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            
            pad_hs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_ts = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, h_size)
            pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, h_size)


            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            #print(h_att.size())
            #ht_att = (h_att * t_att).mean(1)
            m = torch.nn.Threshold(0,0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
            
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            pad_rs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, h_size)
            hss.append(pad_hs)
            tss.append(pad_ts)
            rss.append(pad_rs)
            batch_entity_embs.append(entity_embs)
        hss = torch.stack(hss, dim=0)
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)
        batch_entity_embs = torch.cat(batch_entity_embs, dim=0)
        return hss, rss, tss, batch_entity_embs

    def get_hrt_by_segment(self, sequence_output, attention, entity_pos, hts, segment_span):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        sent_embs = []
        batch_entity_embs = []
        #print(sequence_output.size(), attention.size())
        segment_start, segment_end = segment_span
        seg_start_idx = 0
        b, seq_l, h_size = sequence_output.size()
        n_e = max([len(x) for x in entity_pos])
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            entity_lens = []
            mask = []
            logit_mask = torch.zeros((n_e, n_e))
            for e_pos in entity_pos[i]:
                #entity_lens.append(self.ent_num_emb(torch.tensor(len(e)).to(sequence_output).long()))
                e_pos = [x for x in e_pos if (x[0] >= segment_start)  and (x[1] < segment_end)]
                #print(len(e_pos))
                
                if len(e_pos) > 1:
                    e_emb, e_att = [], []
                    exist = 1 
                    for start, end in e_pos:
                        start = start - segment_start
                        end = start - segment_start
                        if   start + offset < c :
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                elif len(e_pos) == 1:
                    start, end = e_pos[0]
                    start = start - segment_start
                    end = start - segment_start
                    exist = 1 
                    if  start + offset < c :
                        #e_emb = sequence_output[i, start + offset] + seq_sent_embs[start + offset]
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                elif len(e_pos) == 0:
                    exist = 0
                    e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                mask.append(exist)
               
            for i_e in range(n_e):
                for j_e in range(n_e):
                    if mask[i_e]==1 and mask[j_e]==1:
                        logit_mask[i_e, j_e] = 1
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]

            #entity_embs = entity_embs + entity_type_embs
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[0]).to(sequence_output.device)
                        
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0]).view(s_ne, s_ne, h_size)
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1]).view(s_ne, s_ne, h_size)
            
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            #ht_att = (h_att * t_att).mean(1)
            m = torch.nn.Threshold(0,0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
            rs = contract("ld,rl->rd", sequence_output[0], ht_att).view(s_ne, s_ne, h_size)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.stack(hss, dim=0)
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)
        return hss, rss, tss, logit_mask
    

    


    def encode_by_segment(self, input_ids, attention_mask, sentid_mask, ctx_window, stride):
        bsz, seq_len = input_ids.size()
        if seq_len <= ctx_window:
            segment_output, segment_attn = self.encode(input_ids,  attention_mask)
            return segment_output, segment_attn, [(0, seq_len)]
        else:
            segments = math.ceil((seq_len - ctx_window)/stride)
            batch_input_ids = []
            batch_input_attn = []
            segment_spans = []
            context_sz = 100
            max_len = stride * segments + ctx_window
            segment_input = torch.zeros((max_len)).to(input_ids)
            segment_attention = torch.zeros((max_len)).to(input_ids)
            segment_input[:seq_len] =  input_ids.squeeze(0)
            segment_attention[:seq_len] =  attention_mask.squeeze(0)
            for i in range(segments + 1):
                batch_input_ids.append(segment_input[ i * stride: i * stride + ctx_window])
                batch_input_attn.append(segment_attention[ i * stride: i * stride + ctx_window])
                segment_spans.append((i * stride, i * stride + ctx_window))
            batch_input_ids = torch.stack(batch_input_ids ,dim=0)
            batch_input_attn = torch.stack(batch_input_attn, dim=0)
            segment_output, segment_attn = self.encode(batch_input_ids,  batch_input_attn)
            return segment_output, segment_attn, segment_spans
 
    def encode_by_sentence(self, input_ids, attention_mask, sentid_mask, ctx_window, stride):
        bsz, seq_len = input_ids.size()
        
        segments = math.ceil((seq_len - ctx_window)/stride)        
        max_len = ctx_window * segments
        segment_input = torch.zeros((max_len)).to(input_ids)
        segment_attention = torch.zeros((max_len)).to(input_ids)
        segment_input[:seq_len] =  input_ids.squeeze(0)
        segment_attention[:seq_len] =  attention_mask.squeeze(0)
        segment_input = segment_input.view(segments, ctx_window).long()
        segment_attention = segment_attention.view(segments, ctx_window).long()
        segment_output, segment_attn = self.encode(segment_input,  segment_attention)
        return segment_output, segment_attn



    
    def get_logits_by_segment(self, segment_span, sequence_output, attention, entity_pos, hts):
        seg_start, seg_end = segment_span
        sequence_output = sequence_output.unsqueeze(0)
        attention = attention.unsqueeze(0)
        bs, seq_len, h_size = sequence_output.size()
        ne = len(entity_pos[0])
        #hs_e, rs_e, ts_e, logit_mask = self.get_hrt_by_two_segment(sequence_output, attention, entity_pos, hts, segment_span)
        hs_e, rs_e, ts_e, logit_mask = self.get_hrt_by_segment(sequence_output, attention, entity_pos, hts, segment_span)
        #print(hs.size())
        logit_mask = torch.tensor(logit_mask).clone().to(sequence_output).detach()

        hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e], dim=3)))        
        ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e], dim=3)))   
        b1_e = hs_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        b2_e = ts_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(bs, ne, ne, self.emb_size * self.block_size)
        
        feature = self.projection(bl_e) 
        feature = self.axial_transformer(feature) + feature
        logits = self.classifier(feature).squeeze()
        self_mask = (1 - torch.diag(torch.ones((ne)))).unsqueeze(-1).to(sequence_output)
        logits = logits * logit_mask.unsqueeze(-1)
        logits = logits * self_mask

        return logits, logit_mask

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                mention_pos=None,
                mention_hts=None,
                padded_mention=None,
                padded_mention_mask=None,
                sentid_mask=None,
                return_logits = None,
                teacher_logits = None,
                entity_types = None,
                segment_spans = None,
                negative_mask = None,
                label_loader = None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        bs, seq_len, h_size = sequence_output.size()
        bs, num_heads, seq_len, seq_len = attention.size()
        ctx_window = 300
        stride = 25
        device = sequence_output.device.index
        #ne = max([len(x) for x in entity_pos])
        ne = 42
        nes = [len(x) for x in entity_pos]
        hs_e, rs_e, ts_e, batch_entity_embs = self.get_hrt(sequence_output, attention, entity_pos, hts)
        #h_t_s = hs_e - ts_e
        #t_h_s = ts_e - hs_e
        #hxt = hs_e * ts_e
        hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e], dim=3)))
        ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e], dim=3))) 
        #hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e,  t_h_s], dim=3)))
        #ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e,  t_h_s], dim=3)))         
    
        b1_e = hs_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        b2_e = ts_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(bs, ne, ne, self.emb_size * self.block_size)
        
        if negative_mask is not None:
            bl_e = bl_e * negative_mask.unsqueeze(-1)
              

        feature =  self.projection(bl_e)
        feature = self.axial_transformer(feature) + feature
        label_embeddings = []
        '''
        for stp, batch in enumerate(label_loader):
            
            #label_output = self.model(batch[0].to(input_ids), batch[1].to(attention_mask), return_dict=False )[0]
            label_output = self.encode(batch[0].to(input_ids), batch[1].to(attention_mask) )[0]
            #print(label_output.size())

            #label_emb = label_output[:, 0, :]
            label_emb = label_output.mean(dim = 1)
            label_embeddings.append(label_emb)
        '''
        #label_embeddings = torch.cat(label_embeddings, dim=0).detach()
        #label_embeddings = torch.cat(label_embeddings, dim=0)

        if False:
            query = feature.view(bs, -1, self.config.hidden_size)
            query = query.permute(1, 0, 2)
            tgt_len, bsz, embed_dim = query.size()
            #batch_size, batch_seq_len, hid_dim = relation_embedding_context.size()
            key = label_embeddings.unsqueeze(1).permute(1, 0, 2)
            #input_mask = input_masks
            input_mask = torch.ones(key.size()).cuda()
            #pair, attn_weights = self.multihead_attn(query, key, key, key_padding_mask=input_mask)
            attn_feature, attn_weights = self.multihead_attn(query, key, key)
            attn_feature = attn_feature.view(bs, ne, ne, self.config.hidden_size)
            #print(pair.size())
        
        
        #logits_l = torch.matmul(feature.clone(), label_embeddings.T)
        logits_c = self.classifier(feature)
        
        self_mask = (1 - torch.diag(torch.ones((ne)))).unsqueeze(0).unsqueeze(-1).to(sequence_output)
        logits_classifier = logits_c * self_mask
        #logits_label = logits_l * self_mask
        #print(logits_e.size())
        logits_classifier = torch.cat([logits_classifier.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])
        #logits_label = torch.cat([logits_label[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])

        if labels is None:
            logits = logits_classifier.view(-1, self.config.num_labels)
            #logits = logits_classifier.view(-1, self.config.num_labels) + logits_label.view(-1, self.config.num_labels)
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels), logits)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(device)
            
            loss_classifier = self.loss_fnt(logits_classifier.view(-1, self.config.num_labels).float(), labels.float())
            #loss_label, _, _ = self.loss_fnt2(logits_label.float(), labels.float())
            #loss_s, loss1_s, loss2_s = self.loss_fnt(logits_seg.float(), segment_labels.float())
            #output = loss_e.to(sequence_output) + loss_s.to(sequence_output)
            output =  loss_classifier
            '''
            if entity_types is not None:
                entity_types = torch.tensor(entity_types).long().to(logits_0)
                entity_type_preds = self.entity_classifier(batch_entity_embs)
                ent_loss = self.entity_criterion(entity_type_preds, entity_types.long())
                output_0 = output_0 + 0.1*ent_loss
            '''
            if teacher_logits is not None:
                teacher_logits = torch.cat(teacher_logits, dim=0).to(logits_classifier)
                mse_loss = self.mse_criterion(logits_classifier, teacher_logits)
                output = output + 1.0 *  mse_loss

                        
        return output
