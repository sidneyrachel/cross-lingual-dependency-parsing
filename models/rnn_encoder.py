import torch
from torch import nn
from variables.embedding import word_emb_name_to_dim_mapper
from variables import config as cf


class RNNEncoder(nn.Module):
    def __init__(
            self,
            word_field,
            word_emb_dim,
            pos_field,
            pos_emb_dim,
            rnn_size,
            rnn_depth,
            update_pretrained,
            pretrained_we_model
    ):
        super().__init__()
        self.word_emb_mode = cf.config.word_emb_mode
        self.is_ctx_embedding = cf.config.is_ctx_embedding
        self.word_embedding = nn.Embedding(len(word_field.vocab), word_emb_dim)
        # If we're using pre-trained word embeddings, we need to copy them.
        if word_field.vocab.vectors is not None:
            self.word_embedding.weight = nn.Parameter(
                word_field.vocab.vectors,
                requires_grad=update_pretrained
            )

        # POS-tag embeddings will always be trained from scratch.
        self.pos_embedding = nn.Embedding(len(pos_field.vocab), pos_emb_dim)
        self.pretrained_we_model = pretrained_we_model

        input_size = (word_emb_name_to_dim_mapper[cf.config.pretrained_ctx_embedding] + pos_emb_dim) \
            if self.is_ctx_embedding else (word_emb_dim + pos_emb_dim)

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=rnn_size,
            nonlinearity='tanh',
            batch_first=True,
            bidirectional=cf.config.bidirectional,
            num_layers=rnn_depth
        )

    def forward(
            self,
            words,
            postags,
            sent_length,
            we_tokenized_sent_length,
            word_offsets,
            input_ids,
            token_type_ids,
            attention_masks
    ):
        pos_emb = self.pos_embedding(postags)

        if self.is_ctx_embedding:
            word_emb = self.pretrained_we_model.get_full_word_embeddings(
                embedding_mode=self.word_emb_mode,
                sentence_lengths=sent_length,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_masks=attention_masks,
                word_offsets=word_offsets,
                tokenized_sentence_lengths=we_tokenized_sent_length
            )
        else:
            word_emb = self.word_embedding(words)

        word_pos_emb = torch.cat([word_emb, pos_emb], dim=2)

        rnn_out, _ = self.rnn(word_pos_emb)

        return rnn_out
