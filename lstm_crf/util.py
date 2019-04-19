import torch
from torch.utils.data import DataLoader, Dataset


def prepare_sequence(seq, to_ix):
    """シーケンスをIDに変換"""
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class NameDataset(Dataset):
    def __init__(self, data, word_dic, target_dic):
        self.feature = data['feature']
        self.target = data['target']
        self.word_dic = word_dic
        self.target_dic = target_dic

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        feature = [prepare_sequence(s, self.word_dic) for s in self.feature[idx]]
        target = [self.target_dic[t] for t in self.target[idx]]
        return feature[0], target[0]
