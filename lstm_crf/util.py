import MeCab
import torch
from torch.utils.data import Dataset


def parse_text(text):
    tagger = MeCab.Tagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    tagger.parse("")
    node = tagger.parseToNode(text)

    wordlist = []
    hinshilist = []
    while node:
        target = node.surface
        hinshi3 = node.feature.split(',')[2]
        wordlist.append(target)
        hinshilist.append(hinshi3)
        node = node.next
    return wordlist[1: -1], hinshilist[1: -1]  # BOSとEOSを除外


def prepare_sequence(seq, to_ix):
    """シーケンスをIDに変換"""
    # idxs = [to_ix[w] for w in seq]
    idxs = to_ix[seq]
    return torch.tensor(idxs, dtype=torch.long)


class NameDataset(Dataset):
    def __init__(self, data, word_dic, target_dic):
        self.feature = data['featuer']
        self.target = data['target']
        self.word_dic = word_dic
        self.target_dic = target_dic

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        feature = [
            prepare_sequence(s, self.word_dic) for s in self.feature[idx]]
        target = [self.target_dic[t] for t in self.target[idx]]
        return feature[0], target[0]
