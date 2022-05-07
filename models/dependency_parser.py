import torchtext
import torch
import nltk
from utils.corpus import read_data
from utils.file import create_folder_if_not_exist
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import time
from models.edge_factored_parser import EdgeFactoredParser
from models.bert import BERT
from models.xlm import XLM
from variables import config as cf

plt.style.use('seaborn')

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


class DependencyParser:
    def __init__(
            self,
            lower=False
    ):
        self.device = 'cuda'
        self.pretrained_ctx_embedding = cf.config.pretrained_ctx_embedding

        if self.pretrained_ctx_embedding == 'bert':
            self.pretrained_we_model = BERT(
                model_name='bert-base-multilingual-cased',
                device=self.device
            )
        elif self.pretrained_ctx_embedding == 'xlm':
            self.pretrained_we_model = XLM(
                model_name='xlm-mlm-100-1280',
                device=self.device
            )

        pad = '<pad>'

        self.WORD = torchtext.legacy.data.Field(
            init_token=pad,
            pad_token=pad,
            sequential=True,
            lower=lower,
            batch_first=True
        )
        self.POS = torchtext.legacy.data.Field(
            init_token=pad,
            pad_token=pad,
            sequential=True,
            batch_first=True
        )
        self.RAW_WORD = torchtext.legacy.data.RawField()
        self.RAW_WE_TOKENIZED_WORD = torchtext.legacy.data.RawField()
        self.HEAD = torchtext.legacy.data.Field(
            init_token=0,
            pad_token=0,
            use_vocab=False,
            sequential=True,
            batch_first=True
        )
        self.DEPREL = torchtext.legacy.data.Field(
            init_token='root',
            pad_token='root',
            sequential=True,
            batch_first=True
        )
        self.SENT_LENGTH = torchtext.legacy.data.RawField()
        self.WE_TOKENIZED_SENT_LENGTH = torchtext.legacy.data.RawField()
        self.WORD_OFFSET = torchtext.legacy.data.RawField()
        self.INPUT_ID = torchtext.legacy.data.Field(
            init_token=self.pretrained_we_model.get_input_id_padding(),
            pad_token=self.pretrained_we_model.get_input_id_padding(),
            use_vocab=False,
            sequential=True,
            batch_first=True
        )
        self.TOKEN_TYPE_ID = torchtext.legacy.data.Field(
            init_token=self.pretrained_we_model.get_token_type_id_padding(),
            pad_token=self.pretrained_we_model.get_token_type_id_padding(),
            use_vocab=False,
            sequential=True,
            batch_first=True
        )
        self.ATTENTION_MASK = torchtext.legacy.data.Field(
            init_token=self.pretrained_we_model.get_attention_mask_padding(),
            pad_token=self.pretrained_we_model.get_attention_mask_padding(),
            use_vocab=False,
            sequential=True,
            batch_first=True
        )

        self.fields = [
            ('words', self.WORD),
            ('postags', self.POS),
            ('heads', self.HEAD),
            ('deprels', self.DEPREL),
            ('raw_words', self.RAW_WORD),
            ('sent_length', self.SENT_LENGTH),
            ('we_tokenized_sent_length', self.WE_TOKENIZED_SENT_LENGTH),
            ('word_offsets', self.WORD_OFFSET),
            ('input_ids', self.INPUT_ID),
            ('token_type_ids', self.TOKEN_TYPE_ID),
            ('attention_masks', self.ATTENTION_MASK),
            ('raw_we_tokenized_words', self.RAW_WE_TOKENIZED_WORD)
        ]

        self.model_prefix = 'trained_models/biaffine_dep_parser'
        create_folder_if_not_exist('trained_models')

        # Read training and validation data according to the predefined split.
        self.train_examples = read_data(
            corpus_file='external_resources/ud-treebanks-v2.9/UD_English-EWT/en_ewt-ud-train.conllu',
            datafields=self.fields,
            pretrained_we_model=self.pretrained_we_model
        )
        self.val_examples = read_data(
            corpus_file='external_resources/ud-treebanks-v2.9/UD_English-EWT/en_ewt-ud-dev.conllu',
            datafields=self.fields,
            pretrained_we_model=self.pretrained_we_model
        )

        # Load the pre-trained word embeddings that come with the torchtext library.
        use_pretrained = True
        if use_pretrained:
            print('We are using pre-trained word embeddings.')
            self.WORD.build_vocab(self.train_examples, vectors="glove.840B.300d")
        else:
            print('We are training word embeddings from scratch.')
            self.WORD.build_vocab(self.train_examples, max_size=10000)

        self.POS.build_vocab(self.train_examples)
        self.DEPREL.build_vocab(self.train_examples)

        # Create one of the models defined above.
        self.model = EdgeFactoredParser(
            self.fields,
            word_emb_dim=300,
            pos_emb_dim=32,
            rnn_size=256,
            rnn_depth=3,
            mlp_size=256,
            rel_size=len(self.DEPREL.vocab.stoi),
            pretrained_we_model=self.pretrained_we_model,
            update_pretrained=False
        )

        self.model.to(self.device)

    def save_model(self):
        save_path = '{}.pt'.format(self.model_prefix)
        file = open(save_path, mode='wb')
        torch.save(self.model.state_dict(), file)
        file.close()

    def load_model(self):
        save_path = '{}.pt'.format(self.model_prefix)
        self.model.load_state_dict(torch.load(save_path, map_location=torch.device(self.device)))

    def train(self):
        batch_size = 256

        train_iterator = torchtext.legacy.data.BucketIterator(
            self.train_examples,
            device=self.device,
            batch_size=batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True
        )

        val_iterator = torchtext.legacy.data.BucketIterator(
            self.val_examples,
            device=self.device,
            batch_size=batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True
        )

        train_batches = list(train_iterator)
        val_batches = list(val_iterator)

        # We use the betas recommended in the paper by Dozat and Manning. They also use
        # a learning rate cooldown, which we don't use here to keep things simple.
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)

        history = defaultdict(list)

        n_epochs = 30

        for i in range(1, n_epochs + 1):
            t0 = time.time()

            stats = Counter()

            self.model.train()
            for batch in train_batches:
                # print('words', batch.raw_we_tokenized_words[0])
                # print('input ids', batch.input_ids[0])
                loss = self.model(
                    words=batch.words,
                    postags=batch.postags,
                    heads=batch.heads,
                    deprels=batch.deprels,
                    sent_length=batch.sent_length,
                    we_tokenized_sent_length=batch.we_tokenized_sent_length,
                    word_offsets=batch.word_offsets,
                    input_ids=batch.input_ids,
                    token_type_ids=batch.token_type_ids,
                    attention_masks=batch.attention_masks
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                stats['train_loss'] += loss.item()

            train_loss = stats['train_loss'] / len(train_batches)
            history['train_loss'].append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for batch in val_batches:
                    loss, n_uas_errors, n_las_errors, n_tokens = self.model(
                        words=batch.words,
                        postags=batch.postags,
                        heads=batch.heads,
                        deprels=batch.deprels,
                        sent_length=batch.sent_length,
                        we_tokenized_sent_length=batch.we_tokenized_sent_length,
                        word_offsets=batch.word_offsets,
                        input_ids=batch.input_ids,
                        token_type_ids=batch.token_type_ids,
                        attention_masks=batch.attention_masks,
                        evaluate=True
                    )
                    stats['val_loss'] += loss.item()
                    stats['val_n_tokens'] += n_tokens
                    stats['val_n_uas_errors'] += n_uas_errors
                    stats['val_n_las_errors'] += n_las_errors

            val_loss = stats['val_loss'] / len(val_batches)
            uas = (stats['val_n_tokens'] - stats['val_n_uas_errors']) / stats['val_n_tokens']
            las = (stats['val_n_tokens'] - stats['val_n_las_errors']) / stats['val_n_tokens']

            history['val_loss'].append(val_loss)
            history['uas'].append(uas)
            history['las'].append(las)

            t1 = time.time()
            print(
                f'Epoch {i}: '
                f'train loss = {train_loss:.4f}, '
                f'val loss = {val_loss:.4f}, '
                f'UAS = {uas:.4f}, '
                f'LAS = {las: .4f}, '
                f'time = {t1 - t0:.4f}')

        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.plot(history['uas'])
        plt.plot(history['las'])
        plt.legend(['training loss', 'validation loss', 'UAS', 'LAS'])

    def parse(self, sentences):
        # This method applies the trained model to a list of sentences.

        # First, create a torchtext Dataset containing the sentences to tag.
        examples = []

        for tagged_words in sentences:
            words = [w for w, _ in tagged_words]
            postags = [t for _, t in tagged_words]
            heads = [0] * len(words)  # placeholder
            deprels = [''] * len(words)  # placeholder
            raw_words = [w for w, _ in tagged_words]
            sent_length = len(words)
            we_words = self.pretrained_we_model.tokenize_sentence(' '.join(words))
            raw_we_tokenized_words = we_words
            we_tokenized_sent_length = len(we_words)
            input_ids = self.pretrained_we_model.get_input_ids(we_words)
            token_type_ids = self.pretrained_we_model.get_token_type_ids(we_words)
            attention_masks = self.pretrained_we_model.get_attention_masks(we_words)
            word_offsets = [self.pretrained_we_model.get_sub_words_len(w) for w, _ in tagged_words]
            examples.append(torchtext.legacy.data.Example.fromlist([
                words,
                postags,
                heads,
                deprels,
                raw_words,
                sent_length,
                we_tokenized_sent_length,
                word_offsets,
                input_ids,
                token_type_ids,
                attention_masks,
                raw_we_tokenized_words
            ], self.fields))

        dataset = torchtext.legacy.data.Dataset(examples, self.fields)

        iterator = torchtext.legacy.data.Iterator(
            dataset,
            device=self.device,
            batch_size=len(examples),
            repeat=False,
            train=False,
            sort=False
        )

        # Apply the trained model to the examples.
        out_edges = []
        out_rels = []
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                predicted_edges, predicted_rels = self.model.predict(
                    words=batch.words,
                    postags=batch.postags,
                    sent_length=batch.sent_length,
                    we_tokenized_sent_length=batch.we_tokenized_sent_length,
                    word_offsets=batch.word_offsets,
                    input_ids=batch.input_ids,
                    token_type_ids=batch.token_type_ids,
                    attention_masks=batch.attention_masks
                )
                out_edges.extend(predicted_edges.cpu().numpy())
                out_rels.extend(predicted_rels.cpu().numpy())

        return out_edges, out_rels

    def parse_sentence(self, sentence):
        tokenized = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized)
        out_edges, out_rels = self.parse([tagged])
        edges, rels = out_edges[0], out_rels[0]

        for i, ((word, tag), head, rel) in enumerate(zip(tagged, edges[1:], rels[1:]), 1):
            print(f'{i:2} {word:10} {tag:4} {head} {self.DEPREL.vocab.itos[rel]}')
