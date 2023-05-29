from gensim.models import word2vec
from gensim.models import KeyedVectors
import os
import pickle
import random
import torch
import spacy
import numpy as np
import configparser
import msgpack


def get_texts(train_loader, vocabulary):
    texts = []
    for filename in train_loader.loaders:
        for value in train_loader.loaders[filename]:
            loader = list(train_loader.loaders[filename][value])
            for data, _ in loader:
                for text in data:
                    text = text.tolist()
                    for i in range(len(text)):
                        text[i] = vocabulary.to_word(text[i])
                    texts.append(text)
    print('texts', len(texts))
    return texts

def get_weights(model, vocabulary, embed_dim):
    weights = np.zeros((len(vocabulary), embed_dim))
    with open(config['data']['embedding_file'], 'rb') as f:
        embeddings = msgpack.load(f, raw=False)
    for i in range(len(vocabulary)):
        if vocabulary.to_word(i) == '<pad>':
            continue
        word = vocabulary.to_word(i)
        if word in model:
            weights[i] = model[word]
        elif word in embeddings:
            weights[i] = embeddings[word]
    return weights


def main():
    data_path = config['data']['path']
    embed_dim = int(config['model']['embed_dim'])
    vocabulary = pickle.load(open(os.path.join(data_path, config['data']['vocabulary']), 'rb'))
    train_loader = pickle.load(open(os.path.join(data_path, config['data']['train_loader']), 'rb'))
    texts = get_texts(train_loader, vocabulary)
    model = KeyedVectors.load_word2vec_format(config['data']['glove_file'], binary=False)
    weights = get_weights(model, vocabulary, embed_dim)
    pickle.dump(weights, open(os.path.join(data_path, config['data']['weights']), 'wb'))

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    # seed
    seed = int(config['data']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
