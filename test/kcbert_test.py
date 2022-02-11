import re

import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, AdamW



ctx = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(ctx)
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')
model = torch.load('../output/model2/kc_badword2.pt')
model.to(device)


def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx])) / a).item() * 100


def transform(data):
    data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
    data = tokenizer(data)
    return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])


def testModel(model, sentence):
    cate = ['비욕설', '욕설']
    sentence = transform(sentence)
    input_ids = torch.tensor([sentence[0]]).to(device)
    token_type_ids = torch.tensor([sentence[1]]).to(device)
    attention_mask = torch.tensor([sentence[2]]).to(device)
    print(input_ids)

    result = model(input_ids, token_type_ids, attention_mask)
    idx = result.argmax().cpu().item()
    print("문장에는:", cate[idx])
    print("신뢰도는:", "{:.2f}%".format(softmax(result, idx)))


while True:
    s = input('input: ')
    if s == 'quit':
        break
    testModel(model, s)
