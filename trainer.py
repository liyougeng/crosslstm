import os
import random
import time
import math

import torch
from torch.autograd import Variable
import torch.nn as nn

from model import PairCrossLSTM
from metric import Metrics
from log import Log
import config

SELF_TRAIN_TEST =  True
SAVE_MODEL_PATH_PREFIX = './checkpoints/model_pkg_'
file_pattern = SAVE_MODEL_PATH_PREFIX + "in-{}_cell-{}_" + \
                    "pair-{}_vocab-{}_wscal-{}_{}_{}_{}{}{}{}{}{}lr-{}_epoch-{}.pth"
TRAIN, DEV, TEST = (0, 1 ,2)

def map_label_to_target(label, num_classes, use_cuda):
    target = torch.zeros(1,num_classes)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil==floor:
        target[0][floor-1] = 1
    else:
        target[0][floor-1] = ceil - label
        target[0][ceil-1] = label - floor
    if use_cuda:
        return Variable(target).cuda()
    return Variable(target)


def train(train_data, pair_net, criterion, optimizer, \
          cur_epoch=0, epoch = 15, print_every = 1000, batch_size = 25, num_classes = 5, \
          args = None, vocab_size = -1, \
          dev_data = None, test_data = None, \
          use_cuda = None):
    mex = Metrics(num_classes)
    learning_rate = args.lr  # learning rate may update over step
    idxs = range(len(train_data))
    iters = 1
    tmp_loss = 0.0
    history_metric = [[(0.0, 0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0, 0.0)]] # train,dev,test
                      #loss, pearson, spearman, mse

    start_time = time.time()
    pair_net.zero_grad()
    for ep in xrange(cur_epoch, epoch):
        random.shuffle(idxs)
        save_flag1, save_flag2, save_flag3 = False, False, False
        for i,idx in enumerate(idxs):
            iters += 1
            da, db, score = train_data[idx]
            
            prediction = pair_net(da, db)
            target = map_label_to_target(score, num_classes, use_cuda)
            
            loss = criterion(prediction, target)
            tmp_loss += loss.data[0]
            loss.backward()
            if iters % batch_size == 0:
                optimizer.step()
                pair_net.zero_grad()
            
            if (i+1) % print_every == 0:
                end_time = time.time()
                print("epoch:%d, %d-iters:%d, time escaplse: %.5f s, loss: %.3f" % 
                      (ep, print_every, i/print_every, end_time - start_time, tmp_loss / print_every))
                tmp_loss = 0.0
                start_time = end_time
        
        # train_data self test?
        if SELF_TRAIN_TEST:
            save_flag1, history_metric, fp = checkTrainedPerformance(args, vocab_size, ep, learning_rate, history_metric, \
                                                        TRAIN ,train_data, pair_net, criterion, num_classes, use_cuda)
        if dev_data is not None:
            save_flag2, history_metric, fp = checkTrainedPerformance(args, vocab_size, ep, learning_rate, history_metric, \
                                                        DEV, dev_data, pair_net, criterion, num_classes, use_cuda)
        if test_data is not None:
            save_flag3, history_metric, fp = checkTrainedPerformance(args, vocab_size, ep, learning_rate, history_metric, \
                                                        TEST, test_data, pair_net, criterion, num_classes, use_cuda)
        if save_flag1 or save_flag2 or save_flag3 or (ep == epoch - 1):
            # Save loss & draw loss image and save & save model
            checkpoint = {'model':pair_net.state_dict(), 'optim': optimizer, \
                            'cur_epoch': ep, 'epoch': epoch, 'args': args}
            torch.save(checkpoint, fp) # Load by pair_net.load_state_dict(torch.load(saved_model_file))
        
    return history_metric, fp

def test(dataset, pair_net, criterion, num_classes, use_cuda):
    # dataset is [(parsed_sent_tree_a, sent_tree_b,...)]
    indices = torch.arange(1, num_classes + 1)
    golds = torch.zeros(len(dataset))
    preds = torch.zeros(len(dataset))
    losses = 0.0
    # pair_net.eval()
    for i in range(len(dataset)):
        item = dataset[i]
        prediction = pair_net(item[0], item[1])
        target = map_label_to_target(item[2], num_classes, use_cuda)
        loss = criterion(prediction, target)
        losses += loss.data[0]
        output = prediction.data.squeeze().cpu()
        result = torch.dot(indices, torch.exp(output))
        preds[i] = result
        golds[i] = item[2]
        
    # pair_net.train()
    return preds, golds, losses / len(dataset)

def checkTrainedPerformance(args, vocab_size, ep, lr, history_metric, sp,
                            data, pair_net, criterion, num_classes, use_cuda):
    mex = Metrics(num_classes)
    splits = ['train', 'dev', 'test']
    p, g, l = test(data, pair_net, criterion, num_classes, use_cuda)
    pearson = mex.pearson(p, g)
    spearman = mex.spearman(p, g)
    mse = mex.mse(p, g)
    info = '[INFO] Test %s_data@epoch: %d, loss: %.6f, pearson: %.6f, spearman: %.6f, mse: %.6f' % (
                        splits[sp], ep, l, pearson, spearman, mse)
    file_path = file_pattern.format(args.input_dim, args.hidden_dim, 
                     args.pair_dim, vocab_size, args.word_scale, 
                     "glove" if args.use_glove else "",
                     "freeze" if args.freeze else "",
                     "_improved_" if args.use_improved else "",
                     "_linked_" if args.use_link else "",
                     "_cbf_"+str(args.combine_factor) if args.combine_factor >= 0 else "",
                     "_gate_"+str(args.gate_type) if args.gate_type != 0 else "",
                     "_ga_factor_"+str(args.gate_factor) if args.gate_factor != 1.0 else "",
                     "_ga_bias_"+str(args.gate_bias) if args.gate_bias != 0.0 else "",
                     lr, ep)
    Log(file_path, info)
    isBetter = False
    if history_metric[sp][-1][1] < pearson:
        isBetter = True
    history_metric[sp].append((l, pearson, spearman, mse))
    return isBetter, history_metric, file_path

def checkLast(args, vocab_size):
    
    paths = [] # store list of (model, epoch_of_it)
    for x in xrange(args.epoch):
        file_path = file_pattern.format(args.input_dim, args.hidden_dim, 
                 args.pair_dim, vocab_size, args.word_scale,  
                 "glove" if args.use_glove else "",
                 "freeze" if args.freeze else "",
                 "_improved_" if args.use_improved else "",
                 "_linked_" if args.use_link else "",
                 "_cbf_"+str(args.combine_factor) if args.combine_factor >= 0 else "",
                 "_gate_"+str(args.gate_type) if args.gate_type != 0 else "",
                 "_ga_factor_"+str(args.gate_factor) if args.gate_factor != 1.0 else "",
                 "_ga_bias_"+str(args.gate_bias) if args.gate_bias != 0.0 else "",
                 args.lr, x)
        if os.path.exists(file_path):
            paths.append((file_path, x))
    print("[Resume] Found [%d] history checkpoint files. " % len(paths))
    if len(paths) == 0 : return None
    return paths[-1]

def trainer(args, vocab, train_data, dev_data = None, test_data = None, glove = None):
    vocab_size = len(vocab)
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    pair_dim = args.pair_dim
    output_dim = 0 # not used
    num_classes = args.num_classes
    word_scale = args.word_scale
    use_cuda = args.use_cuda

    learning_rate = args.lr
    weight_decay = args.wc

    freeze = args.freeze

    if (args.combine_factor >= 0.0 and  (not args.use_link))  :
        print("[ERROR] combine_factor is enabled, thus you must enable the use_link!")
        exit(-1)

    pair_net = PairCrossLSTM(input_dim, hidden_dim, pair_dim, output_dim, vocab_size, \
                              num_classes, word_scale, use_cuda, glove, freeze,\
                              args.use_improved, args.use_link, args.combine_factor, \
                              args.gate_type, args.gate_factor, args.gate_bias)
    criterion = nn.KLDivLoss()

    if use_cuda:
        pair_net.cuda()
        criterion.cuda()
    
    cur_epoch = 0
    if args.weight is not None and args.weight != '' and os.path.exists(args.weight):
        history_parameters = torch.load(args.weight)
        if 'model' in history_parameters:
            model = history_parameters['model']
            if 'optim' in history_parameters:
                optimizer = history_parameters['optimizer']
        else: # version 1 history parameters
            model = history_parameters 
            optimizer = torch.optim.Adagrad(pair_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
    # check previous stored checkpoints.
        lastCheckpoint = checkLast(args, vocab_size)
        if lastCheckpoint is None:
            optimizer = torch.optim.Adagrad(pair_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            print("[Resume] restore model from %s" % lastCheckpoint[0])
            checkpoint = torch.load(lastCheckpoint[0])
            optimizer = checkpoint['optim']
            model_parameters = checkpoint['model']
            pair_net.load_state_dict(model_parameters)
            cur_epoch = lastCheckpoint[1] + 1

    epoch = args.epoch
    batch_size = args.batch_size
    display_iter = config.display_iter
    
    train_history, last_name = train(train_data, pair_net, criterion, optimizer, \
                               cur_epoch, epoch, display_iter, batch_size, num_classes, \
                               args, vocab_size, \
                               dev_data, test_data, \
                               use_cuda)

    
    return train_history, last_name
