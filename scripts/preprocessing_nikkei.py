import MeCab
import pickle
import pathlib
import pickle


def open_file(path):
    with open(path, 'r') as f:
        text = f.read()
    sentence = text.split('。')
    sentence = [s+'。' for s in sentence]
    return sentence


def parse_text(text):
    tagger = MeCab.Tagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    tagger.parse("")
    node = tagger.parseToNode(text)
    result = []
    while node:
        target = node.surface
        hinshi3 = node.feature.split(',')[2]
        result.append((target, hinshi3))
        node = node.next
    return result[1: -1]  # BOSとEOSを除外


if __name__ == '__main__':
    data_dir = pathlib.Path('../data')
    all_file = list(data_dir.glob('**/*.txt'))
    
    result = []
    for file in all_file:
        text = open_file(file)
        for sentence in text:
            result.append(parse_text(sentence))

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