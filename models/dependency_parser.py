import torchtext
import torch
import nltk
from utils.corpus import read_data
from utils.file import create_folder_if_not_exist
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import time
from models.edge_factored_parser import EdgeFactoredParser

plt.style.use('seaborn')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class DependencyParser:
    def __init__(self, lower=False):
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
        self.HEAD = torchtext.legacy.data.Field(
            init_token=0,
            pad_token=0,
            use_vocab=False,
            sequential=True,
            batch_first=True
        )
        self.DEPREL = torchtext.legacy.data.Field(
            init_token=pad,
            pad_token=pad,
            sequential=True,
            batch_first=True
        )
        self.fields = [('words', self.WORD), ('postags', self.POS), ('heads', self.HEAD), ('deprels', self.DEPREL)]
        self.device = 'cuda'
        self.model_prefix = 'trained_models/biaffine_dep_parser'
        create_folder_if_not_exist('trained_models')

        # Read training and validation data according to the predefined split.
        self.train_examples = read_data(
            'external_resources/ud-treebanks-v2.9/UD_English-EWT/en_ewt-ud-train.conllu',
            self.fields
        )
        self.val_examples = read_data(
            'external_resources/ud-treebanks-v2.9/UD_English-EWT/en_ewt-ud-dev.conllu',
            self.fields
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
            sort=True)

        val_iterator = torchtext.legacy.data.BucketIterator(
            self.val_examples,
            device=self.device,
            batch_size=batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True)

        train_batches = list(train_iterator)
        val_batches = list(val_iterator)

        # We use the betas recommended in the paper by Dozat and Manning. They also use
        # a learning rate cooldown, which we don't use here to keep things simple.
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)

        history = defaultdict(list)

        n_epochs = 1

        for i in range(1, n_epochs + 1):
            t0 = time.time()

            stats = Counter()

            self.model.train()
            for batch in train_batches:
                loss = self.model(batch.words, batch.postags, batch.heads)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                stats['train_loss'] += loss.item()

            train_loss = stats['train_loss'] / len(train_batches)
            history['train_loss'].append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for batch in val_batches:
                    loss, n_err, n_tokens = self.model(batch.words, batch.postags, batch.heads, evaluate=True)
                    stats['val_loss'] += loss.item()
                    stats['val_n_tokens'] += n_tokens
                    stats['val_n_err'] += n_err

            val_loss = stats['val_loss'] / len(val_batches)
            uas = (stats['val_n_tokens'] - stats['val_n_err']) / stats['val_n_tokens']
            history['val_loss'].append(val_loss)
            history['uas'].append(uas)

            t1 = time.time()
            print(
                f'Epoch {i}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, UAS = {uas:.4f}, time = {t1 - t0:.4f}')

        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.plot(history['uas'])
        plt.legend(['training loss', 'validation loss', 'UAS'])

    def parse(self, sentences):
        # This method applies the trained model to a list of sentences.

        # First, create a torchtext Dataset containing the sentences to tag.
        examples = []
        for tagged_words in sentences:
            words = [w for w, _ in tagged_words]
            tags = [t for _, t in tagged_words]
            heads = [0] * len(words)  # placeholder
            deprels = [''] * len(words)  # placeholder
            examples.append(torchtext.legacy.data.Example.fromlist([words, tags, heads, deprels], self.fields))
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
        out = []
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                predicted = self.model.predict(batch.words, batch.postags)
                out.extend(predicted.cpu().numpy())

        return out

    def parse_sentence(self, sentence):
        tokenized = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized)
        edges = self.parse([tagged])[0]

        for i, ((word, tag), head) in enumerate(zip(tagged, edges[1:]), 1):
            print(f'{i:2} {word:10} {tag:4} {head}')
