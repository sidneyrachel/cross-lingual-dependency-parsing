import torchtext


def read_data(
        corpus_file,
        datafields,
        pretrained_we_model
):
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        words = []
        postags = []
        heads = []
        deprels = []
        raw_words = []
        word_offsets = []

        for line in f:
            if line[0] == '#':  # Skip comments.
                continue
            line = line.strip()

            if not line:
                # Blank line for the end of a sentence.
                sent_length = len(words)
                we_words = pretrained_we_model.tokenize_sentence(' '.join(words))
                raw_we_tokenized_words = we_words
                we_tokenized_sent_length = len(we_words)
                input_ids = pretrained_we_model.get_input_ids(we_words)
                token_type_ids = pretrained_we_model.get_token_type_ids(we_words)
                attention_masks = pretrained_we_model.get_attention_masks(we_words)

                examples.append(
                    torchtext.legacy.data.Example.fromlist([
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
                    ], datafields)
                )

                words = []
                postags = []
                heads = []
                deprels = []
                raw_words = []
                word_offsets = []
            else:
                columns = line.split('\t')

                # Skip dummy tokens used in ellipsis constructions, and multiword tokens.
                if '.' in columns[0] or '-' in columns[0]:
                    continue

                words.append(columns[1])
                postags.append(columns[3])
                heads.append(int(columns[6]))
                deprels.append(columns[7])
                raw_words.append(columns[1])
                word_offsets.append(pretrained_we_model.get_sub_words_len(columns[1]))

        return torchtext.legacy.data.Dataset(examples, datafields)
