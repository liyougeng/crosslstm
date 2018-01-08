import json
from os import path as osp
import pickle as pkl

from config import *


def buildVocab(force=False, sp='train'):
    vocab_file = "./sentdata/train_vocab.pkl"
    if (not force) and osp.exists(vocab_file):
        with open(vocab_file, 'r') as fd:
            return pkl.load(fd)
    
    file_a = osp.join(BASE_DIR, sp, 'a.txt.json')
    file_b = osp.join(BASE_DIR, sp, 'b.txt.json')
    with open(file_a, 'r') as fd, open(file_b, 'r') as fdx:
        json_a = json.load(fd)
        json_b = json.load(fdx)
    sentences_a = json_a["sentences"]
    sentences_b = json_b["sentences"]
    
    assert len(sentences_a) == len(sentences_b)
    vocab = dict()
    for sent in sentences_a:
        for token in sent["tokens"]:
            vocab[token['word'].lower()] = vocab.get(token['word'].lower(), 0) + 1
    for sent in sentences_b:
        for token in sent["tokens"]:
            vocab[token['word'].lower()] = vocab.get(token['word'].lower(), 0) + 1
    vocab["<BOS>"] = 1
    vocab["<EOS>"] = 2
    vocab["<UNK>"] = 3 # add a PAD word? (for mask?)
    vocab["<PAD>"] = 4

    with open(vocab_file, 'w') as fd:
        pkl.dump(vocab, fd)
    return vocab
    
def id2word(vocab, idx):
    assert idx >= 0
    if idx >= len(vocab.items()): return "<UNK>"
    return vocab.items[idx][0]

def vocab2table(vocab):
    table = dict()
    for idx, key_cc_tuple in enumerate(vocab.items()):
        table[key_cc_tuple[0]] = idx
    return table

def word2id(table, word):
    if word not in table:
        return table['<UNK>']
    return table[word]

def load_word_vectors(word_path, glove_path, table, force=True):
    import os
    import torch
    if (not force) and os.path.isfile(word_path + 'train_w2v.pth') and os.path.isfile(word_path + 'train_vocab.pkl'):
        return torch.load(word_path + 'train_w2v.pth')
    print("[INFO] Words' init tensor is not found, we build a minimal word2vec set from glove.")
    word_cc = len(table.keys())
    with open(glove_path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    
    vectors = torch.zeros(word_cc, dim)
    words_in_glove = dict()
    with open(glove_path + '.txt', 'r') as f:
        cc = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            if contents[0] in table:
                idx = table[contents[0]]
                words_in_glove[contents[0]] = 1
                vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
                cc += 1
    print('[INFO] processed %d words of %d words in vocab.' %(cc, len(table)))
    cc = 0
    for word, idx in table.items():
        if word not in words_in_glove:
            cc += 1
            vectors[idx].normal_(-0.05, 0.05) # guassion? or uniform? word_scale?
    torch.save(vectors, word_path + 'train_w2v.pth')
    print('[INFO] processed %d words of %d words which are not in glove.' %(cc, len(table)))
    return vectors
    
def get_glove_inited_embeddings(table, force=True):
    # table is a dictionary which maps a word to its idx
    glove_emb = load_word_vectors('./sentdata/', './sentdata/glove/glove.840B.300d', table, force)
    for item in ['<BOS>', '<EOS>', '<UNK>', '<PAD>']:
        glove_emb[table[item]].zero_()
    return glove_emb