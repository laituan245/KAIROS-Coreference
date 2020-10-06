import torch
import random
import numpy as np

from constants import *
from models.base import *
from models.encoder import *

class EntityCorefModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.configs = configs

        # Mention Encoding
        self.encoder = TransformerEncoder(configs)
        self.attention_scorer = FFNNModule(self.encoder.hidden_size, [], 1)

        # Mention Linking
        self.link_scorer = FFNNModule(input_size=self.get_pair_embs_size(),
                                      hidden_sizes=[configs['ffnn_size']] * configs['ffnn_depth'],
                                      output_size=1)

        # Move to Device
        self.to(self.device)

    def get_pair_embs_size(self):
        pair_embs_size = 3 * self.get_span_emb_size()
        return pair_embs_size

    def get_span_emb_size(self):
        span_emb_size = 3 * self.encoder.hidden_size
        return span_emb_size

    def forward(self, input_ids, input_masks, is_training,
                gold_starts, gold_ends, cluster_ids, mask_windows):
        assert(not is_training) # Inference only
        self.train() if is_training else self.eval()

        input_ids = torch.tensor(input_ids.astype(np.int64)).to(self.device)
        input_masks = torch.tensor(input_masks.astype(np.int64)).to(self.device)
        mask_windows = torch.tensor(mask_windows.astype(np.int64)).to(self.device)
        gold_starts = torch.tensor(gold_starts.astype(np.int64)).to(self.device)
        gold_ends = torch.tensor(gold_ends.astype(np.int64)).to(self.device)
        cluster_ids = torch.tensor(cluster_ids.astype(np.int64)).to(self.device)
        num_windows, window_size = input_ids.size()[:2]

        # Mention Features
        features = self.encoder(input_ids, input_masks, mask_windows, num_windows, window_size, is_training)
        features = features.squeeze()
        mention_features = self.get_span_emb(features, gold_starts, gold_ends)

        # Get pair_embs, pair_scores
        n = len(gold_starts)
        pair_scores = torch.zeros((n, n)).to(self.device)
        for i in range(n):
            row_pair_embs = self.get_row_pair_embs(mention_features, i)
            pair_scores[i,:] = self.link_scorer(row_pair_embs)

        # Compute antecedents_mask and antecedent_scores
        k = mention_features.size()[0] # Total number of mentions
        dummy_zeros = torch.zeros([k, 1]).to(self.device)
        span_range = torch.arange(0, k).to(self.device)
        antecedent_offsets = span_range.view(-1, 1) - span_range.view(1, -1)
        antecedents_mask = antecedent_offsets >= 1 # [k, k]
        antecedent_scores = pair_scores + torch.log(antecedents_mask.float())
        antecedent_scores = torch.cat([dummy_zeros, antecedent_scores], dim=1)

        # Preds
        top_antecedents = torch.arange(0, k).to(self.device)
        top_antecedents = top_antecedents.unsqueeze(0).repeat(k, 1)
        preds = [gold_starts, gold_ends, top_antecedents, antecedent_scores]

        return None, preds

    def get_row_pair_embs(self, candidate_embs, row_index):
        n, d = candidate_embs.size()
        features_list = []

        # Compute diff_embs and prod_embs
        src_embs = candidate_embs.view(1, n, d)
        target_embs = candidate_embs[row_index,:].view(1, 1, d).repeat([1, n, 1])
        prod_embds = src_embs * target_embs

        # Update features_list
        features_list.append(src_embs)
        features_list.append(target_embs)
        features_list.append(prod_embds)

        # Concatenation
        pair_embs = torch.cat(features_list, 2)

        return pair_embs

    def get_pair_embs(self, candidate_embs):
        n, d = candidate_embs.size()
        features_list = []

        # Compute diff_embs and prod_embs
        src_embs = candidate_embs.view(1, n, d).repeat([n, 1, 1])
        target_embs = candidate_embs.view(n, 1, d).repeat([1, n, 1])
        prod_embds = src_embs * target_embs

        # Update features_list
        features_list.append(src_embs)
        features_list.append(target_embs)
        features_list.append(prod_embds)

        # Concatenation
        pair_embs = torch.cat(features_list, 2)

        return pair_embs

    def get_span_emb(self, context_outputs, span_starts, span_ends):
        span_emb_list = []
        num_tokens = context_outputs.size()[0]

        # Extract the boundary representations for the candidate spans
        span_start_emb = torch.index_select(context_outputs, 0, span_starts)
        span_end_emb = torch.index_select(context_outputs, 0, span_ends)
        assert(span_start_emb.size()[0] == span_end_emb.size()[0])
        span_emb_list.append(span_start_emb)
        span_emb_list.append(span_end_emb)

        # Extract attention-based representations
        doc_range = torch.arange(0, num_tokens).to(self.device)
        range_cond_1 = span_starts.unsqueeze(1) <= doc_range
        range_cond_2 = doc_range <= span_ends.unsqueeze(1)
        doc_range_mask = range_cond_1 & range_cond_2
        attns = self.attention_scorer(context_outputs).unsqueeze(0) + torch.log(doc_range_mask.float())
        attn_probs = torch.softmax(attns,dim=1)
        head_attn_reps = torch.matmul(attn_probs, context_outputs)
        span_emb_list.append(head_attn_reps)

        # Return
        span_emb = torch.cat(span_emb_list, dim=1)
        return span_emb
