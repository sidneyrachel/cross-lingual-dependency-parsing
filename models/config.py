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
