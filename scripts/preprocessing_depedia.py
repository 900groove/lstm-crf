import numpy as np
import pandas as pd
import random
import pickle


def shuffle_text(text):
    """文を句点で区切り、句点の数の組み合わせを構築"""
    text_sentence = text.split('。')
    result = []
    for i in range(len(text_sentence)):
        result.append(
            ''.join(random.sample(text_sentence, len(text_sentence))))
    return result


def data_augmentation(df):
    """句点による組み合わせによるデータ水増し"""
    df_ = pd.DataFrame(columns=['name', 'text'])
    for value in df.values:
        texts = shuffle_text(value[1])
        names = np.tile(value[0], len(texts))
        result_df = pd.DataFrame(
            (np.c_[names, texts]), columns=['name', 'text'])
        df_ = pd.concat([df_, result_df], axis=0)
    return df_


def find_str_index(target, sentence):
    """文中のターゲット単語の位置を計算"""
    counter = sentence.count(target)
    result = np.zeros(len(sentence))
    for i in range(counter):
        if i == 0:
            loc = sentence.find(target)
        else:
            loc = sentence.find(target, loc + len(target))
        result[loc: loc + len(target)] = 1
    result = [str(int(i)) for i in result]
    return (list(sentence), result)


if __name__ == '__main__':
    train_df = pd.read_csv('../data/data.csv')
    train_df = data_augmentation(train_df)
    train_df.to_csv('../data/processed.csv', index=False)

    text = []
    text_id = []
    for target, sentence in train_df.values:
        _text, _text_id = find_str_index(target, sentence)
        text.append(_text)
        text_id.append(_text_id)
    train_data = {'feature': text, 'target': text_id}
    with open('../data/train_data.pickle', mode='wb') as f:
        pickle.dump(train_data, f)

    word_to_ix = {}
    for sentence in train_data['feature']:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    with open('../model/word_to_ix.pickle', mode='wb') as f:
        pickle.dump(word_to_ix, f)
