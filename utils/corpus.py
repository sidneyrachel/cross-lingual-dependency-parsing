import torchtext


def read_data(corpus_file, datafields):
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        words = []
        postags = []
        heads = []
        deprels = []

        for line in f:
            if line[0] == '#':  # Skip comments.
                continue
            line = line.strip()

            if not line:
                # Blank line for the end of a sentence.
                examples.append(torchtext.legacy.data.Example.fromlist([words, postags, heads, deprels], datafields))
                words = []
                postags = []
                heads = []
                deprels = []
            else:
                columns = line.split('\t')

                # Skip dummy tokens used in ellipsis constructions, and multiword tokens.
                if '.' in columns[0] or '-' in columns[0]:
                    continue

                words.append(columns[1])
                postags.append(columns[3])
                heads.append(int(columns[6]))
                deprels.append(columns[7])

        return torchtext.legacy.data.Dataset(examples, datafields)
