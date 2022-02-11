import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW



class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx=0, label_idx=1, max_len=64, add_token=0):
        self.dataset = dataset
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-large")
        self.vocab_size = self.tokenizer.vocab_size
        if add_token:
            self.added_token_num = self.tokenizer.add_tokens(add_token)
        self.sentences = [self.transform(i[sent_idx]) for i in self.dataset]
        self.labels = [np.int32(i[label_idx]) for i in self.dataset]

    def transform(self, data):
        data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
        data = self.tokenizer(data, max_length=self.max_len, padding="max_length", truncation=True,)
        return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx] + (self.labels[idx],)

tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-large")
print(tokenizer('헬로우 브라더'))