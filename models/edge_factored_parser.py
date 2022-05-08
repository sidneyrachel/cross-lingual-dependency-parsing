import torch
from torch import nn

from models.lstm_encoder import LSTMEncoder
from models.gru_encoder import GRUEncoder
from models.rnn_encoder import RNNEncoder
from models.biaffine_edge_scorer import BiaffineEdgeScorer
from variables import config as cf


class EdgeFactoredParser(nn.Module):
    def __init__(
            self,
            fields,
            word_emb_dim,
            pos_emb_dim,
            rnn_size,
            rnn_depth,
            arc_mlp_size,
            rel_mlp_size,
            rel_size,
            pretrained_we_model,
            update_pretrained=False
    ):
        super().__init__()

        self.word_field = fields[0][1]
        self.pos_field = fields[1][1]
        self.pretrained_we_model = pretrained_we_model

        # Sentence encoder module.
        encoder_arch = cf.config.encoder_arch

        if encoder_arch == 'rnn':
            self.encoder = RNNEncoder(
                word_field=self.word_field,
                word_emb_dim=word_emb_dim,
                pos_field=self.pos_field,
                pos_emb_dim=pos_emb_dim,
                rnn_size=rnn_size,
                rnn_depth=rnn_depth,
                update_pretrained=update_pretrained,
                pretrained_we_model=self.pretrained_we_model
            )
        elif encoder_arch == 'gru':
            self.encoder = GRUEncoder(
                word_field=self.word_field,
                word_emb_dim=word_emb_dim,
                pos_field=self.pos_field,
                pos_emb_dim=pos_emb_dim,
                rnn_size=rnn_size,
                rnn_depth=rnn_depth,
                update_pretrained=update_pretrained,
                pretrained_we_model=self.pretrained_we_model
            )
        else:
            self.encoder = LSTMEncoder(
                word_field=self.word_field,
                word_emb_dim=word_emb_dim,
                pos_field=self.pos_field,
                pos_emb_dim=pos_emb_dim,
                rnn_size=rnn_size,
                rnn_depth=rnn_depth,
                update_pretrained=update_pretrained,
                pretrained_we_model=self.pretrained_we_model
            )

        # Edge scoring module.
        self.edge_scorer = BiaffineEdgeScorer(
            rnn_size=2 * rnn_size if cf.config.bidirectional else rnn_size,
            arc_mlp_size=arc_mlp_size,
            rel_mlp_size=rel_mlp_size,
            rel_size=rel_size
        )

        # To deal with the padding positions later, we need to know the
        # encoding of the padding dummy word.
        self.pad_id = self.word_field.vocab.stoi[self.word_field.pad_token]

        # Loss function that we will use during training.
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def word_tag_dropout(
            self,
            words,
            postags,
            input_ids
    ):
        word_unk_id = self.word_field.vocab.stoi[self.word_field.unk_token]
        pos_unk_id = self.pos_field.vocab.stoi[self.pos_field.unk_token]
        input_unk_id = self.pretrained_we_model.get_unknown_token_id()

        w_dropout_mask = (torch.rand(size=words.shape, device=words.device) > cf.config.word_dropout).long()
        p_dropout_mask = (torch.rand(size=words.shape, device=postags.device) > cf.config.postag_dropout).long()
        i_dropout_mask = (torch.rand(size=words.shape, device=input_ids.device) > cf.config.word_dropout).long()

        w_indices = (w_dropout_mask == 0).nonzero(as_tuple=False).cpu().numpy()
        p_indices = (p_dropout_mask == 0).nonzero(as_tuple=False).cpu().numpy()
        i_indices = (i_dropout_mask == 0).nonzero(as_tuple=False).cpu().numpy()

        w_tuple_indices = tuple(map(tuple, w_indices))
        p_tuple_indices = tuple(map(tuple, p_indices))
        i_tuple_indices = tuple(map(tuple, i_indices))

        cloned_words = torch.clone(words)
        cloned_postags = torch.clone(postags)
        cloned_input_ids = torch.clone(input_ids)

        for index in w_tuple_indices:
            cloned_words[index] = word_unk_id

        for index in p_tuple_indices:
            cloned_postags[index] = pos_unk_id

        for index in i_tuple_indices:
            cloned_input_ids[index] = input_unk_id

        return cloned_words, cloned_postags, cloned_input_ids

    def forward(
            self,
            words,
            postags,
            heads,
            deprels,
            sent_length,
            we_tokenized_sent_length,
            word_offsets,
            input_ids,
            token_type_ids,
            attention_masks,
            evaluate=False
    ):
        # We don't want to evaluate the loss or attachment score for the positions
        # where we have a padding token. So we create a mask that will be zero for those
        # positions and one elsewhere.
        pad_mask = (words != self.pad_id).float()

        if not evaluate:
            # If we are training, apply the word/tag dropout to the word and tag tensors.
            words, postags, input_ids = self.word_tag_dropout(
                words=words,
                postags=postags,
                input_ids=input_ids
            )

        encoded = self.encoder(
            words=words,
            postags=postags,
            sent_length=sent_length,
            we_tokenized_sent_length=we_tokenized_sent_length,
            word_offsets=word_offsets,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_masks=attention_masks
        )
        edge_scores, rel_scores = self.edge_scorer(encoded)

        loss = self.compute_loss(edge_scores, rel_scores, heads, deprels, pad_mask)

        if evaluate:
            n_uas_errors, n_las_errors, n_tokens = self.evaluate(edge_scores, rel_scores, heads, deprels, pad_mask)
            return loss, n_uas_errors, n_las_errors, n_tokens
        else:
            return loss

    def compute_loss(self, edge_scores, rel_scores, heads, deprels, pad_mask):
        n_sentences, n_words, _ = edge_scores.shape
        _, _, _, n_rels = rel_scores.shape

        heads = heads.view(n_sentences * n_words)
        deprels = deprels.view(n_sentences * n_words)

        edge_scores = edge_scores.view(n_sentences * n_words, n_words)
        rel_scores = rel_scores.reshape(n_sentences * n_words, n_words, n_rels)
        rel_scores = rel_scores[torch.arange(len(heads)), heads]

        pad_mask = pad_mask.view(n_sentences * n_words)

        arc_loss = self.loss(edge_scores, heads)
        rel_loss = self.loss(rel_scores, deprels)

        avg_arc_loss = arc_loss.dot(pad_mask) / pad_mask.sum()
        avg_rel_loss = rel_loss.dot(pad_mask) / pad_mask.sum()

        return avg_arc_loss + avg_rel_loss

    def evaluate(self, edge_scores, rel_scores, heads, deprels, pad_mask):
        n_sentences, n_words, _ = edge_scores.shape
        _, _, _, n_rels = rel_scores.shape

        heads = heads.view(n_sentences * n_words)
        deprels = deprels.view(n_sentences * n_words)

        edge_scores = edge_scores.view(n_sentences * n_words, n_words)
        rel_scores = rel_scores.reshape(n_sentences * n_words, n_words, n_rels)
        rel_scores = rel_scores[torch.arange(len(heads)), heads]

        pad_mask = pad_mask.view(n_sentences * n_words)
        n_tokens = pad_mask.sum()

        edge_predictions = edge_scores.argmax(dim=1)
        rel_predictions = rel_scores.argmax(dim=1)

        pad_mask_bool = pad_mask.bool()
        n_arc_error_mask = (edge_predictions != heads) & pad_mask_bool
        n_label_error_mask = (edge_predictions == heads) & (rel_predictions != deprels) & pad_mask_bool

        n_arc_errors = n_arc_error_mask.float().sum()
        n_label_errors = n_label_error_mask.float().sum()

        n_uas_errors = n_arc_errors.item()
        n_las_errors = n_uas_errors + n_label_errors.item()
        n_tokens = n_tokens.item()

        return n_uas_errors, n_las_errors, n_tokens

    def predict(
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
        # This method is used to parse a sentence when the model has been trained.
        encoded = self.encoder(
            words=words,
            postags=postags,
            sent_length=sent_length,
            we_tokenized_sent_length=we_tokenized_sent_length,
            word_offsets=word_offsets,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_masks=attention_masks
        )
        edge_scores, rel_scores = self.edge_scorer(encoded)
        edge_preds = edge_scores.argmax(-1)
        rel_preds = rel_scores.argmax(-1).gather(-1, edge_preds.unsqueeze(-1)).squeeze(-1)

        return edge_preds, rel_preds
