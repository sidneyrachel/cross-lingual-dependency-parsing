import torch
from torch import nn


class RNNEncoder(nn.Module):
    def __init__(self, word_field, word_emb_dim, pos_field, pos_emb_dim, rnn_size, rnn_depth, update_pretrained):
        super().__init__()

        self.word_embedding = nn.Embedding(len(word_field.vocab), word_emb_dim)
        # If we're using pre-trained word embeddings, we need to copy them.
        if word_field.vocab.vectors is not None:
            self.word_embedding.weight = nn.Parameter(word_field.vocab.vectors,
                                                      requires_grad=update_pretrained)

        # POS-tag embeddings will always be trained from scratch.
        self.pos_embedding = nn.Embedding(len(pos_field.vocab), pos_emb_dim)

        self.rnn = nn.LSTM(input_size=word_emb_dim + pos_emb_dim, hidden_size=rnn_size, batch_first=True,
                           bidirectional=True, num_layers=rnn_depth)

    def forward(self, words, postags):
        # Look u
        word_emb = self.word_embedding(words)
        pos_emb = self.pos_embedding(postags)
        word_pos_emb = torch.cat([word_emb, pos_emb], dim=2)

        rnn_out, _ = self.rnn(word_pos_emb)

        return rnn_out