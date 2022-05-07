import torch


def get_first_embeddings(
        token
):
    return token[0]


def get_last_embeddings(
        token
):
    return token[-1]


def get_sum_embeddings(
        token
):
    return torch.sum(token[1:], dim=0)


def get_second_to_last_embeddings(
        token
):
    return token[-2]


def get_last_four_sum_embeddings(
        token
):
    return torch.sum(token[-4:], dim=0)


def get_last_four_cat_embeddings(
        token
):
    return torch.cat((token[-4], token[-3], token[-2], token[-1]), dim=0)


def extract_embedding_layer(
        embedding_mode,
        token_embeddings
):
    if embedding_mode == 'first':
        get_embeddings_function = get_first_embeddings
    elif embedding_mode == 'last':
        get_embeddings_function = get_last_embeddings
    elif embedding_mode == 'sum':
        get_embeddings_function = get_sum_embeddings
    elif embedding_mode == 'second_to_last':
        get_embeddings_function = get_second_to_last_embeddings
    elif embedding_mode == 'last_four_sum':
        get_embeddings_function = get_last_four_sum_embeddings
    elif embedding_mode == 'last_four_cat':
        get_embeddings_function = get_last_four_cat_embeddings
    else:
        raise Exception(f'Embedding mode is unknown. Mode: {embedding_mode}.')

    embedded_sentence = []

    for token in token_embeddings:
        embedded_token = get_embeddings_function(token)
        embedded_sentence.append(embedded_token)

    return embedded_sentence
