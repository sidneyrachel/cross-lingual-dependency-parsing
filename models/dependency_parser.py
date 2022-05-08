import torchtext
import torch
import nltk
from utils.corpus import read_data, get_corpus_files
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
            logger,
            lower=False
    ):
        self.device = 'cuda'
        self.logger = logger
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

        self.model_prefix = f'trained_models/{cf.config.model_name}'
        create_folder_if_not_exist('trained_models')

        # Read training and validation data according to the predefined split.
        self.train_examples = read_data(
            corpus_files=get_corpus_files(langs=cf.config.train_set, set_name='train'),
            datafields=self.fields,
            pretrained_we_model=self.pretrained_we_model
        )
        self.val_examples = read_data(
            corpus_files=get_corpus_files(langs=cf.config.val_set, set_name='dev'),
            datafields=self.fields,
            pretrained_we_model=self.pretrained_we_model
        )
        self.test_examples = read_data(
            corpus_files=get_corpus_files(langs=cf.config.test_set, set_name='test'),
            datafields=self.fields,
            pretrained_we_model=self.pretrained_we_model
        )

        # Load the pre-trained word embeddings that come with the torchtext library.
        use_pretrained = False
        if use_pretrained:
            self.logger('We are using pre-trained word embeddings.')
            self.WORD.build_vocab(self.train_examples, vectors="glove.840B.300d")
        else:
            self.logger('We are training word embeddings from scratch.')
            self.WORD.build_vocab(self.train_examples, max_size=10000)

        self.POS.build_vocab(self.train_examples)
        self.DEPREL.build_vocab(self.train_examples)

        # Create one of the models defined above.
        self.model = EdgeFactoredParser(
            self.fields,
            word_emb_dim=cf.config.word_emb_dim,
            pos_emb_dim=cf.config.pos_emb_dim,
            rnn_size=cf.config.rnn_size,
            rnn_depth=cf.config.rnn_depth,
            arc_mlp_size=cf.config.arc_mlp_size,
            rel_mlp_size=cf.config.rel_mlp_size,
            rel_size=len(self.DEPREL.vocab.stoi),
            pretrained_we_model=self.pretrained_we_model,
            update_pretrained=False
        )

        self.model.to(self.device)

        train_iterator = torchtext.legacy.data.BucketIterator(
            self.train_examples,
            device=self.device,
            batch_size=cf.config.train_batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True
        )

        val_iterator = torchtext.legacy.data.BucketIterator(
            self.val_examples,
            device=self.device,
            batch_size=cf.config.val_batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True
        )

        test_iterator = torchtext.legacy.data.BucketIterator(
            self.test_examples,
            device=self.device,
            batch_size=cf.config.test_batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True
        )

        self.train_batches = list(train_iterator)
        self.val_batches = list(val_iterator)
        self.test_batches = list(test_iterator)

    def save_model(self, postfix=None):
        if postfix:
            save_path = '{}.pt'.format('_'.join([self.model_prefix, postfix]))
        else:
            save_path = '{}.pt'.format(self.model_prefix)

        file = open(save_path, mode='wb')
        torch.save(self.model.state_dict(), file)
        file.close()

    def load_model(self, postfix=None):
        save_path = '{}.pt'.format('_'.join([self.model_prefix, postfix]))
        self.model.load_state_dict(torch.load(save_path, map_location=torch.device(self.device)))

    def train(self):
        # We use the betas recommended in the paper by Dozat and Manning. They also use
        # a learning rate cooldown, which we don't use here to keep things simple.
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)

        history = defaultdict(list)

        n_epochs = cf.config.num_epoch
        best_las = 0
        best_epoch = 0

        for i in range(1, n_epochs + 1):
            t0 = time.time()

            stats = Counter()

            self.model.train()
            for batch in self.train_batches:
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

            train_loss = stats['train_loss'] / len(self.train_batches)
            history['train_loss'].append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for batch in self.val_batches:
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

            val_loss = stats['val_loss'] / len(self.val_batches)
            val_uas = (stats['val_n_tokens'] - stats['val_n_uas_errors']) / stats['val_n_tokens']
            val_las = (stats['val_n_tokens'] - stats['val_n_las_errors']) / stats['val_n_tokens']

            if val_las > best_las:
                best_las = val_las
                best_epoch = i
                self.save_model(postfix='best')

            history['val_loss'].append(val_loss)
            history['val_uas'].append(val_uas)
            history['val_las'].append(val_las)

            self.model.eval()
            with torch.no_grad():
                for batch in self.test_batches:
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
                    stats['test_loss'] += loss.item()
                    stats['test_n_tokens'] += n_tokens
                    stats['test_n_uas_errors'] += n_uas_errors
                    stats['test_n_las_errors'] += n_las_errors

            test_loss = stats['test_loss'] / len(self.test_batches)
            test_uas = (stats['test_n_tokens'] - stats['test_n_uas_errors']) / stats['test_n_tokens']
            test_las = (stats['test_n_tokens'] - stats['test_n_las_errors']) / stats['test_n_tokens']

            t1 = time.time()
            self.logger(
                f'Epoch {i}: '
                f'train loss = {train_loss:.4f}, '
                f'val loss = {val_loss:.4f}, '
                f'test loss = {test_loss:.4f}, '
                f'val UAS = {val_uas:.4f}, '
                f'val LAS = {val_las: .4f}, '
                f'test UAS = {test_uas:.4f}, '
                f'test LAS = {test_las: .4f}, '
                f'time = {t1 - t0:.4f}'
            )

        self.save_model(postfix='last')
        self.logger(f'Best epoch: {best_epoch}')

        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.plot(history['val_uas'])
        plt.plot(history['val_las'])
        plt.legend(['training loss', 'validation loss', 'validation UAS', 'validation LAS'])
        plt.savefig(f'figures/{cf.config.model_name}.png')
        plt.close()

    def evaluate(self, set_name):
        if set_name == 'test':
            batches = self.test_batches
        elif set_name == 'dev':
            batches = self.val_batches
        else:
            batches = []

        stats = Counter()

        self.model.eval()
        with torch.no_grad():
            for batch in batches:
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
                stats['loss'] += loss.item()
                stats['n_tokens'] += n_tokens
                stats['n_uas_errors'] += n_uas_errors
                stats['n_las_errors'] += n_las_errors

        loss = stats['loss'] / len(batches)
        uas = (stats['n_tokens'] - stats['n_uas_errors']) / stats['n_tokens']
        las = (stats['n_tokens'] - stats['n_las_errors']) / stats['n_tokens']

        self.logger(
            f'{set_name} loss = {loss:.4f}, '
            f'{set_name} UAS = {uas:.4f}, '
            f'{set_name} LAS = {las: .4f}'
        )

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
            self.logger(f'{i:2} {word:10} {tag:4} {head} {self.DEPREL.vocab.itos[rel]}')
