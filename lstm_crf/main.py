import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
from lstm_crf import BiLSTM_CRF
from util import NameDataset


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
TRAINIG_EPOCH = 5
BATCH_SIZE = 128
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'


with open('../model/word_to_ix.pickle', mode='rb') as f:
    word_to_ix = pickle.load(f)

"""
with open('../data/train_data.pickle', mode='rb') as f:
    data_pair = pickle.load(f)
tag_to_ix = {"0": 0, "1": 1, START_TAG: 2, STOP_TAG: 3}
"""

with open('../data/train_data2.pickle', mode='rb') as f:
    data_pair = pickle.load(f)
tag_to_ix = {'*': 0,
             'サ変接続': 1,
             '一般': 2,
             '人名': 3,
             '副詞可能': 4,
             '助動詞語幹': 5,
             '助数詞': 6,
             '地域': 7,
             '引用': 8,
             '形容動詞語幹': 9,
             '特殊': 10,
             '組織': 11,
             '連語': 12,
             START_TAG: 13,
             STOP_TAG: 14}


def train(model, optimizer, dataloader):
    for epoch in range(TRAINIG_EPOCH):
        for n, (sentence, tags) in enumerate(dataloader):
            model.zero_grad()
            sentence_in = sentence.to(DEVICE)
            targets = tags.to(DEVICE)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()
            if n % 10 == 0:
                print(f"epoch:{epoch} batch:{n} loss:{loss.item()}")
    return model


if __name__ == '__main__':
    dataset = NameDataset(data_pair, word_to_ix, tag_to_ix)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    model = train(model, optimizer, dataloader)
    with open('../model/trained_model.pickle', mode='wb') as f:
        pickle.dump(model, f)
