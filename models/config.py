import json


class Config:
    def __init__(self, filename):
        f = open(filename)
        self.config = json.load(f)

    @property
    def is_ctx_embedding(self):
        return self.config['is_ctx_embedding']

    @property
    def pretrained_ctx_embedding(self):
        return self.config['pretrained_ctx_embedding']

    @property
    def word_emb_mode(self):
        return self.config['word_emb_mode']

    @property
    def word_dropout(self):
        return self.config['word_dropout']

    @property
    def postag_dropout(self):
        return self.config['postag_dropout']

    @property
    def encoder_arch(self):
        return self.config['encoder_arch']

    @property
    def bidirectional(self):
        return self.config['bidirectional']

    @property
    def word_emb_dim(self):
        return self.config['word_emb_dim']

    @property
    def pos_emb_dim(self):
        return self.config['pos_emb_dim']

    @property
    def rnn_size(self):
        return self.config['rnn_size']

    @property
    def rnn_depth(self):
        return self.config['rnn_depth']

    @property
    def arc_mlp_size(self):
        return self.config['arc_mlp_size']

    @property
    def rel_mlp_size(self):
        return self.config['rel_mlp_size']

    @property
    def train_set(self):
        return self.config['train_set']

    @property
    def val_set(self):
        return self.config['val_set']

    @property
    def test_set(self):
        return self.config['test_set']

    @property
    def num_epoch(self):
        return self.config['num_epoch']

    @property
    def train_batch_size(self):
        return self.config['train_batch_size']

    @property
    def val_batch_size(self):
        return self.config['val_batch_size']

    @property
    def test_batch_size(self):
        return self.config['test_batch_size']
