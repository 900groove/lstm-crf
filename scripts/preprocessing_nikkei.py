import MeCab
import pickle
import pathlib
import platform
from tqdm import tqdm


def get_file_dir():
    """ フォルダから再帰的にテキストファイルを抽出し、ファイルパスを出力 """
    data_dir = pathlib.Path('../data/nikkei/nikkei_data/')
    all_file = list(data_dir.glob('**/*.txt'))
    return all_file


def open_file(path):
    """ ファイルから文書を読み取り、文に区切りリストとして出力 """
    with open(path, 'r') as f:
        text = f.read()
    sentence = text.split('。')
    sentence = [s + '。' for s in sentence]
    return sentence


def parse_text(text):
    """ 文を形態素分けし、品詞情報を付けて出力 """
    #  OSによってNEologd辞書の場所が異なる
    if platform.system() == 'Darwin':  # mac
        tagger = MeCab.Tagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    elif platform.system() == 'Linux':  # ubuntu(linux)
        tagger = MeCab.Tagger(' -d /usr/lib/mecab/dic/mecab-ipadic-neologd')

    tagger.parse("")
    node = tagger.parseToNode(text)
    result = []  # 文の区切りを示すために先頭にスペースを入れる
    while node:
        target = node.surface
        hinshi3 = node.feature.split(',')[2]
        result.append(f'{target}\t{hinshi3}')
        node = node.next
    result[-1] = '\n'
    return result[1:]


def main():
    filelist = get_file_dir()
    for file in tqdm(filelist):
        sentences = open_file(file)
        parseed_sent = [parse_text(s) for s in sentences]
        with open('../data/nikkei.txt', mode='a') as f:
            for s in parseed_sent:
                f.write('\n'.join(s))


if __name__ == '__main__':
    main()
