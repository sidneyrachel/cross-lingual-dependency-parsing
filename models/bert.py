import os
import torch
from transformers import BertTokenizer, BertModel
from utils.embedding import extract_embedding_layer


class BERT:
    def __init__(
            self,
            model_name='bert-base-multilingual-cased',
            device=None
    ):
        self.model_dir = f"{os.getenv('WORD_EMB_DIR')}/bert/" if os.getenv('WORD_EMB_DIR') else ''
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir + model_name)
        self.model = BertModel.from_pretrained(self.model_dir + model_name)
        self.model.to(device)
        self.model.eval()

    def get_token_type_id_padding(self):
        # TODO: What if there are more than 1 sentences
        bos_token_id = 0

        return bos_token_id

    def get_input_id_padding(self):
        return self.tokenizer.pad_token_id

    def get_token_type_ids(
            self,
            tokenized_sentence
    ):
        # TODO: What if there are more than 1 sentences
        bos_token_id = 0

        return [bos_token_id] * len(tokenized_sentence)

    @staticmethod
    def get_attention_masks(
            tokenized_sentence
    ):
        return [1] * len(tokenized_sentence)

    @staticmethod
    def get_attention_mask_padding():
        return 0

    def get_input_ids(
            self,
            tokenized_sentence
    ):
        return self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

    def get_unknown_token_id(
            self
    ):
        return self.tokenizer.unk_token_id

    def tokenize_sentence(
            self,
            sentence
    ):
        return self.tokenizer.tokenize(
            self.tokenizer.cls_token + ' ' + sentence + ' ' + self.tokenizer.sep_token
        )

    def get_sub_words_len(self, word):
        sub_words = self.tokenize_sentence(sentence=word)

        return len(sub_words) - 2  # Omit [CLS] and [SEP]

    def get_embeddings(
            self,
            input_ids,
            token_type_ids,
            attention_masks,
            embedding_mode
    ):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                attention_mask=attention_masks
            )

            hidden_states = outputs.hidden_states
            layer_embeddings = torch.stack(hidden_states, dim=0)
            # Dimension before: layer, batch, token, embedding
            batch_embeddings = layer_embeddings.permute(1, 2, 0, 3)
            # Dimension after: batch, token, layer, embedding

        embedded_sentences = []

        for token_embeddings in batch_embeddings:
            embedded_sentence = extract_embedding_layer(
                embedding_mode=embedding_mode,
                token_embeddings=token_embeddings
            )

            embedded_sentences.append(torch.stack(embedded_sentence))

        return torch.stack(embedded_sentences)

    @staticmethod
    def get_full_word_embeddings_from_sub_word_embeddings(
            word_offsets,
            sentence_length,
            tokenized_sentence_length,
            embedded_sentence
    ):
        new_embedded_sentence = [torch.mean(embedded_sentence[0:1], dim=0)]
        i_bert = 2  # Skip padding and [CLS] token

        for i_offset in range(sentence_length):
            word_offset = word_offsets[i_offset]

            new_embedded_sentence.append(torch.mean(embedded_sentence[i_bert:i_bert + word_offset], dim=0))

            i_bert += word_offset

        if i_bert != tokenized_sentence_length:  # Add padding and omit [SEP] token
            raise Exception(f'Fail to get full word sentence embeddings. '
                            f'i_bert: {i_bert}. '
                            f'Tokenized sentence len: {tokenized_sentence_length}.')

        new_embedded_sentence = torch.stack(new_embedded_sentence)

        return new_embedded_sentence

    def get_full_word_embeddings(
            self,
            embedding_mode,
            sentence_lengths,
            input_ids,
            token_type_ids,
            attention_masks,
            word_offsets,
            tokenized_sentence_lengths
    ):
        embedded_sentences = self.get_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_masks=attention_masks,
            embedding_mode=embedding_mode
        )

        batch_size = embedded_sentences.size(0)
        max_length = max(sentence_lengths) + 1  # For the first padding

        for i in range(batch_size):
            sentence_length = sentence_lengths[i] + 1  # For the first padding

            embedded_sentences[i, :sentence_length, :] = self.get_full_word_embeddings_from_sub_word_embeddings(
                word_offsets=word_offsets[i],
                sentence_length=sentence_lengths[i],
                tokenized_sentence_length=tokenized_sentence_lengths[i],
                embedded_sentence=embedded_sentences[i]
            )

            embedded_sentences[i, sentence_length:max_length, :].zero_()

        embedded_sentences = embedded_sentences[:, :max_length, :]

        return embedded_sentences
