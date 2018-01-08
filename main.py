'''
    Please execute following cmd in shell before execute!

    for i in `ls *.py`; do pylint --errors-only $i; done
''' 

import json
import torch
from os import path as osp
import pickle as pkl
import re

import random
from random import shuffle, random, randint
import time

from config import *
from config import parse_args
from vocab import buildVocab, id2word, vocab2table, word2id, get_glove_inited_embeddings
from dataset import buildData
from model import TreeNode, PairCrossLSTM
from trainer import  trainer
from visualize import process_training_history

start_time = time.time()
print("[Timing] Start building vocab:")

vocab = buildVocab(force=force_rebuild_vocab)
table = vocab2table(vocab)

end_time = time.time()
print("[Timing] Build done! %.2f used." % (end_time - start_time))
start_time = end_time


vocab_size = len(table)
print("[TEST] Vocab size is: %d" % (vocab_size))
print("[TEST] Vocab building infomation, id of[The]: %d " % word2id(table, 'the'))
print("[Timing] Building train/dev/test dataset:")

train_data, _ = buildData('train', table)
dev_data, _ = buildData('dev', table)
test_data, _ = buildData('test', table)

end_time = time.time()
print("[Timing] Build done! %.2f used." % (end_time - start_time))
start_time = end_time

rndint = randint(0, len(train_data) - 1)
print("[TEST] sample data in training set, idx: %d" % rndint)
train_data[rndint][0].beautifyOutput()


if __name__ == "__main__":
    args = parse_args()

    print "-" * 80
    for arg in vars(args):
        print "key:{} = {}".format(arg, getattr(args, arg))
    print "-" * 80

    if args.use_glove:
        print("[INFO] Get glove embedding.")
        glove = get_glove_inited_embeddings(table, force_rebuild_vocab)
    else:
        print("[INFO] glove is disabled, init word_embedding from uniform distribution. ")
        glove = None

    print("[INFO] Start training...")
    start_time = time.time()
    
    train_history, last_name = trainer(args, table, train_data, dev_data, test_data, glove)
    
    end_time = time.time()
    hh = (end_time - start_time) / 3600
    mm = ((end_time - start_time) % 3600) / 60
    ss = (((end_time - start_time) % 3600) % 60) 
    print("[Timing] Cost: %d : %d : %.2f" % (hh, mm, ss))
    
    process_training_history(train_history, last_name[:-3])




