import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.base import BaseModel, FFNNModule
from models.encoder import TransformerEncoder

class EventCorefModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)

        self.encoder = TransformerEncoder(configs)
        self.pair_scorer = FFNNModule(self.get_pair_embs_size(), [configs['ffnn_size']] * configs['ffnn_depth'])

        # Embeddings for additional features
        # Event type embedding (If use_event_type_features enabled)
        if configs['use_event_type_features']:
            self.type_embeddings = nn.Embedding(2, configs['feature_size'])

        # Move model to device
        self.to(self.device)

    def forward(self, inst, is_training):
        self.train() if is_training else self.eval()

        input_ids = torch.tensor(inst.token_windows).to(self.device)
        input_masks = torch.tensor(inst.input_masks).to(self.device)
        mask_windows = torch.tensor(inst.mask_windows).to(self.device)
        num_windows, window_size = input_ids.size()

        # Apply the Transfomer encoder to get tokens features
        tokens_features = self.encoder(input_ids, input_masks, mask_windows,
                                       num_windows, window_size, is_training).squeeze()
        num_tokens = tokens_features.size()[0]

        # Compute word_features (averaging)
        word_features = []
        word_starts_indexes = inst.word_starts_indexes
        word_ends_indexes = word_starts_indexes[1:] + [num_tokens]
        word_features = self.get_span_emb(tokens_features, word_starts_indexes, word_ends_indexes)
        assert(word_features.size()[0] == inst.num_words)

        # Compute event_mention_features (averaging)
        event_mentions = inst.event_mentions
        event_mention_starts = [e['start'] for e in event_mentions]
        event_mention_ends = [e['end'] for e in event_mentions]
        event_mention_features = self.get_span_emb(word_features, event_mention_starts, event_mention_ends)
        assert(event_mention_features.size()[0] == len(event_mentions))

        # Compute event types features (if enabled)
        same_types_features = None
        if self.configs['use_event_type_features']:
            n = len(event_mentions)
            e_types = [e['event_type'] for e in event_mentions]
            same_types = torch.zeros((n, n)).to(self.device).long()
            for i in range(n):
                for j in range(n):
                    same_types[i, j] = int(e_types[i] == e_types[j])
            same_types_features = self.type_embeddings(same_types)

        # Compute pair features and score the pairs (while avoiding crashes due to GPU limit)
        n = len(event_mentions)
        pair_scores = torch.zeros((n, n)).to(self.device)
        for i in range(n):
            row_pair_embs = self.get_row_pair_embs(event_mention_features, i)
            if not same_types_features is None:
                row_pair_embs = torch.cat([row_pair_embs, same_types_features[i:i+1,:,:]], dim=-1)
            pair_scores[i,:] = self.pair_scorer(row_pair_embs)

        # Compute antecedent_scores
        k = len(event_mentions)
        dummy_zeros = torch.zeros([k, 1]).to(self.device)
        span_range = torch.arange(0, k).to(self.device)
        antecedent_offsets = span_range.view(-1, 1) - span_range.view(1, -1)
        antecedents_mask = antecedent_offsets >= 1 # [k, k]
        antecedent_scores = pair_scores + torch.log(antecedents_mask.float())
        antecedent_scores = torch.cat([dummy_zeros, antecedent_scores], dim=1)

        # Preds
        top_antecedents = torch.arange(0, k).to(self.device)
        top_antecedents = top_antecedents.unsqueeze(0).repeat(k, 1)
        preds = [torch.tensor(event_mention_starts),
                 torch.tensor(event_mention_ends),
                 top_antecedents,
                 antecedent_scores]

        return None, preds

    def get_span_emb(self, context_features, span_starts, span_ends):
        num_tokens = context_features.size()[0]

        features = []
        for s, e in zip(span_starts, span_ends):
            sliced_features = context_features[s:e, :]
            features.append(torch.mean(sliced_features, dim=0, keepdim=True))
        features = torch.cat(features, dim=0)
        return features

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

    def get_pair_embs(self, event_features, event_mentions):
        n, d = event_features.size()
        features_list = []

        # Compute diff_embs and prod_embs
        src_embs = event_features.view(1, n, d).repeat([n, 1, 1])
        target_embs = event_features.view(n, 1, d).repeat([1, n, 1])
        prod_embds = src_embs * target_embs

        # Update features_list
        features_list.append(src_embs)
        features_list.append(target_embs)
        features_list.append(prod_embds)

        # Concatenation
        pair_embs = torch.cat(features_list, 2)

        return pair_embs

    def get_span_emb_size(self):
        span_emb_size = self.encoder.transformer_hidden_size
        return span_emb_size

    def get_pair_embs_size(self):
        pair_embs_size = 3 * self.get_span_emb_size() # src_vector, target_vector, product_vector
        if self.configs['use_event_type_features']:
            pair_embs_size += self.configs['feature_size']
        return pair_embs_size
