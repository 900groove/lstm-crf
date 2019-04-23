import torch
import torch.nn as nn
torch.manual_seed(1)


def argmax(vec):
    """ベクトルの最大値のインデックス"""
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    """シーケンスをIDに変換"""
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    """対数除算のオーバーフロー及びアンダーフロー回避"""
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim,
                 hidden_dim, START_TAG="<START>", STOP_TAG="<STOP>"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim//2,
                            num_layers=1,
                            bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 遷移行列を構築し、開始と終了タグの遷移を-10000にすることでこれ以降の遷移を防ぐ
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:,  tag_to_ix[STOP_TAG]] = -10000
        self.start_tag = START_TAG
        self.stop_tag = STOP_TAG

    def init_hidden(self):
        """LSTMのメモリを初期化"""
        return (torch.randn(2, 1, self.hidden_dim//2),
                torch.randn(2, 1, self.hidden_dim//2))

    def _forward_alg(self, feats):
        """観測行列スコア計算"""
        # 入力データの観測素性行列を構築（-10000で初期化）
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # 開始タグを0に変更
        init_alphas[0][self.tag_to_ix[self.start_tag]] = 0.
        forward_var = init_alphas

        # 特定単語に対する全品詞
        for feat in feats:
            alphas_t = []
            # 特定単語に対するそれぞれの品詞
            for next_tag in range(self.tagset_size):
                # 単語と品詞の共起スコアを計算（初期化した数値を取り出す）
                emit_score = feat[next_tag].view(1, -1).expand(
                                                        1, self.tagset_size)
                # 遷移スコアを計算（初期化した数値を取り出す）
                trans_score = self.transitions[next_tag].view(1, -1)
                # 次のタグは初期化された遷移行列に共起スコアと遷移スコアを適応させて計算
                next_tag_var = forward_var + trans_score + emit_score
                # 特定単語の各品詞スコアを格納
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[
                                            self.tag_to_ix[self.stop_tag]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        """双方向LSTMによる特徴量変換"""
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        """遷移行列のスコアを計算"""
        score = torch.zeros(1)
        # 各タグをIDに変換しTensor型に変更
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.start_tag]],
                                       dtype=torch.long), tags])
        # 遷移行列のスコアを合算
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        # 開始タグと終了タグをリセット
        score += self.transitions[self.tag_to_ix[self.stop_tag], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []  # 各単語の次のタグのインデックスリスト
        # 遷移スコアを初期化
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.start_tag]] = 0
        forward_var = init_vvars

        # 各単語に対して各品詞のスコアを計算
        for feat in feats:
            # 候補タグ
            bptrs_t = []  # 最大確率タグのID
            viterbivars_t = []  # 最大確率のタグのスコア
            for next_tag in range(self.tagset_size):
                # 次のタグ候補は上で初期化したものに遷移行列のベクトルを足し合わせたベクトル
                next_tag_var = forward_var + self.transitions[next_tag]
                # 最大（確率）のタグを取り出す
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 入力単語（特徴量）に遷移スコアを加えて遷移行列を構築
            forward_var = (torch.cat(viterbivars_t)+feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 終了タグのスコアを計算.終了タグの遷移行列を取り出す
        terminal_var = forward_var + self.transitions[
                                        self.tag_to_ix[self.stop_tag]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 終了タグから逆向きにパスを計算
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.start_tag]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        """Loss計算"""
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
