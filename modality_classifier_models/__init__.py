import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

from constants import *
from modality_classifier_models.base import BaseModel, ScoreModule
from modality_classifier_models.encoder import TransformerEncoder

POLARITY_TYPES = ['Negative', 'Positive']

class BaseAttributesClassifier(BaseModel):
    def __init__(self, configs, device, attribute_types):
        BaseModel.__init__(self, configs, device)
        self.attribute_types = attribute_types

        # Model components
        self.encoder = TransformerEncoder(configs)
        self.scorer = self.new_score_module(len(attribute_types))
        self.loss_fct = nn.CrossEntropyLoss()

        # Move to device
        self.to(self.device)

    def forward(self, inst, is_training, use_groundtruth=True):
        self.eval()
        assert(not is_training)

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
        if use_groundtruth:
            event_mentions = inst.event_mentions
        else:
            event_mentions = inst.pred_event_mentions
        event_mention_starts = [e['start'] for e in event_mentions]
        event_mention_ends = [e['end'] for e in event_mentions]
        event_mention_features = self.get_span_emb(word_features, event_mention_starts, event_mention_ends)
        assert(event_mention_features.size()[0] == len(event_mentions))

        # Apply the scorers and extract predictions
        scores = self.scorer(event_mention_features)
        scores = scores.view(len(event_mentions), -1)
        probs = torch.softmax(scores, dim=-1)
        pred = list(scores.max(1)[1].cpu().data.numpy())
        pred = [self.attribute_types[idx] for idx in pred]
        pred_prob = probs.cpu().data.numpy()

        return pred, pred_prob

    def get_span_emb(self, context_features, span_starts, span_ends):
        num_tokens = context_features.size()[0]

        features = []
        for s, e in zip(span_starts, span_ends):
            sliced_features = context_features[s:e, :]
            features.append(torch.mean(sliced_features, dim=0, keepdim=True))
        features = torch.cat(features, dim=0)
        return features

    def get_labels(self, event_mentions, property_name, property_values):
        try:
            labels = [e[property_name] for e in event_mentions]
            labels = [property_values.index(l) for l in labels]
        except:
            # Annotations for the property are not available
            labels = [-1] * len(event_mentions)
        labels = torch.tensor(labels).to(self.device)
        return labels

    def new_score_module(self, C):
        configs = self.configs
        return ScoreModule(self.get_span_emb_size(),
                           [configs['ffnn_size']] * configs['ffnn_depth'],
                           output_size=C)

    def get_span_emb_size(self):
        span_emb_size = self.encoder.transformer_hidden_size
        return span_emb_size

class OldPolarityClassifier(BaseModel):
    def __init__(self, configs, device):
        BaseModel.__init__(self, configs, device)

        self.mask = [1, 0, 0, 0, 0]

        # Event Attributes Types
        self.attribute_types = []
        self.attribute_types.append(('polarity', POLARITY_TYPES))

        # Model components
        self.encoder = TransformerEncoder(configs)
        scorers = []
        scorers.append(self.new_score_module(len(POLARITY_TYPES)))
        self.scorers = nn.ModuleList(scorers)

        # Move to device
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

        # Apply the scorers
        all_scores = [scorer(event_mention_features) for scorer in self.scorers]

        # Extraction predictions
        for scores, (type_name, types) in zip(all_scores, self.attribute_types):
            scores = scores.view(len(event_mentions), -1)
            probs = torch.softmax(scores, dim=-1)
            pred = list(scores.max(1)[1].cpu().data.numpy())
            pred = [types[idx] for idx in pred]
            preds = pred
            pred_probs = probs.cpu().data.numpy()

        return preds, pred_probs

    def get_span_emb(self, context_features, span_starts, span_ends):
        num_tokens = context_features.size()[0]

        features = []
        for s, e in zip(span_starts, span_ends):
            sliced_features = context_features[s:e, :]
            features.append(torch.mean(sliced_features, dim=0, keepdim=True))
        features = torch.cat(features, dim=0)
        return features

    def get_labels(self, event_mentions, property_name, property_values):
        try:
            labels = [e[property_name] for e in event_mentions]
            labels = [property_values.index(l) for l in labels]
        except:
            # Annotations for the property are not available
            labels = [-1] * len(event_mentions)
        labels = torch.tensor(labels).to(self.device)
        return labels

    def new_score_module(self, C):
        configs = self.configs
        return ScoreModule(self.get_span_emb_size(),
                           [configs['ffnn_size']] * configs['ffnn_depth'],
                           output_size=C)

    def get_span_emb_size(self):
        span_emb_size = self.encoder.transformer_hidden_size
        return span_emb_size
