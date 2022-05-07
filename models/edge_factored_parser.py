import torch
from torch import nn

from models.rnn_encoder import RNNEncoder
from models.biaffine_edge_scorer import BiaffineEdgeScorer


class EdgeFactoredParser(nn.Module):
    def __init__(
            self,
            fields,
            word_emb_dim,
            pos_emb_dim,
            rnn_size,
            rnn_depth,
            mlp_size,
            rel_size,
            pretrained_we_model,
            update_pretrained=False
    ):
        super().__init__()

        word_field = fields[0][1]
        pos_field = fields[1][1]

        # Sentence encoder module.
        self.encoder = RNNEncoder(
            word_field=word_field,
            word_emb_dim=word_emb_dim,
            pos_field=pos_field,
            pos_emb_dim=pos_emb_dim,
            rnn_size=rnn_size,
            rnn_depth=rnn_depth,
            update_pretrained=update_pretrained,
            pretrained_we_model=pretrained_we_model
        )

        # Edge scoring module.
        self.edge_scorer = BiaffineEdgeScorer(
            rnn_size=2 * rnn_size,
            mlp_size=mlp_size,
            rel_size=rel_size
        )

        # To deal with the padding positions later, we need to know the
        # encoding of the padding dummy word.
        self.pad_id = word_field.vocab.stoi[word_field.pad_token]

        # Loss function that we will use during training.
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def word_tag_dropout(self, words, postags, p_drop):
        # Randomly replace some of the positions in the word and postag tensors with a zero.
        # This solution is a bit hacky because we assume that zero corresponds to the "unknown" token.
        w_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()
        p_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()

        return words * w_dropout_mask, postags * p_dropout_mask

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
        if self.training:
            # If we are training, apply the word/tag dropout to the word and tag tensors.
            words, postags = self.word_tag_dropout(words, postags, 0.25)

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

        # We don't want to evaluate the loss or attachment score for the positions
        # where we have a padding token. So we create a mask that will be zero for those
        # positions and one elsewhere.
        pad_mask = (words != self.pad_id).float()

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
