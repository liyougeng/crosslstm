import torch
import torch.nn as nn

# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

# from tree import Tree
def gate_factory(x, gate_type=0, gate_factor=2, gate_bias=0, h=0):      # is h==0, then, input x = hx
    if gate_type == 0:
        return F.sigmoid(x + h)             # default version, range(0,1)
    elif gate_type == 1:
        return F.sigmoid(x + h) * gate_factor         # scaled gate: range(0,gate_factor)
    elif gate_type == 2:
        return F.sigmoid(x + h) * gate_factor + gate_bias     # shift but not scaled, range(bais,gate_factor+bais)
    elif gate_type == 3:
        return F.sigmoid(x + h) * 4 - 2     # shift and scaled: range(-2, 2)
    elif gate_type == 4:
        return (F.sigmoid(x) + F.sigmoid(h))/2 # two independent gate, range(0, 1)   vs default version
    elif gate_type == 5:
        return F.sigmoid(x) + F.sigmoid(h)   # two independent gate, range(0,2)
    elif gate_type == 6:
        return F.sigmoid(x) * F.sigmoid(h)   # multiple gate: range(0,1)
    else:
        print("[ERROR] Not supported gate method. return default.")  # get x+h
        return F.sigmoid(x + h)
    return x+h # not execuated,  please note: two independent gated failed when two gates bot scaled and shifted

class TreeNode(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, 
                 word_scale=0.05, glove_mat=None, cuda_enabled=True,
                 link_words=False, gate_type=0):
        super(TreeNode, self).__init__()
        
        self.cuda_enabled = cuda_enabled
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size

        if glove_mat is not None:
            self.init_embeddings(glove_mat)
        else:
            self.embeds = nn.Embedding(vocab_size, self.input_size)
            self.embeds.weight.data.uniform_(-word_scale, word_scale) # uniform vs guassion distribution
        
        if self.cuda_enabled: self.embeds = self.embeds.cuda()
        
        self.encode_inputx = nn.Linear(self.input_size, 4 * self.hidden_size)  # only used to encode word
        # hidden state of last child is input of parent, encode as input
        self.encode_inputh = nn.Linear(self.hidden_size, 4 * self.hidden_size) 
        self.encode_prevh = nn.Linear(self.hidden_size, 4 * self.hidden_size)  # encode hidden state 
        
        #self.iofu = nn.Linear(self.hidden_size, self.hidden_size * 4)
    
    def toggle_embeddings_state(self, state=False):
        self.embeds.weight.requires_grad = state
    
    def init_embeddings(self, mat): # for futures's glove, word2vec and others 
        vocab_size, self.input_size = mat.size() # override the input_size
        assert vocab_size >= self.vocab_size
        
        self.embeds = nn.Embedding(vocab_size, self.input_size)
        self.embeds.weight.data.copy_(mat)
        
    def forward(self, tree, prev_h, prev_c):
        assert len(tree.children) != 0 or tree.word is not None
        assert not (len(tree.children) != 0 and tree.word is not None) # move assert code into tree
        
        if tree.getType(): # mid node
            tree.zero_state = self.init_state()
            self.forward(tree.children[0], * tree.zero_state)
            for idx in range(1, len(tree.children)):
                self.forward(tree.children[idx], *tree.children[idx-1].state)
            
            prev_ch, _ = tree.children[-1].state
            if self.hidden_size == self.input_size: # when input_x(word_embedding) == hidden_size, use same weights
                hx = self.encode_inputx(prev_ch) + self.encode_prevh(prev_h) # copy prev_ch or reference it?????  reference makes connected
            else:
                # last child's hidden is input; prev_h is sibling's hidden   # backward of mid-Node
                hx = self.encode_inputh(prev_ch) + self.encode_prevh(prev_h) # above.   should use reference
            
        else: # Leaf Node
            input_idx = Variable(torch.LongTensor([tree.word]))
            input_x = self.embeds(input_idx.cuda() if self.cuda_enabled else input_idx) # embedding is cuda/cpu, so input_x is correponding one
            hx = self.encode_inputx(input_x) + self.encode_prevh(prev_h) # prev_h is given as params (siblings or zeros)
        
        #iofu = self.iofu(hx)
        i, o, f, u = torch.split(hx, hx.size(1) // 4, dim=1)
        i, o, f, u = F.sigmoid(i) , F.sigmoid(o), F.sigmoid(f), F.tanh(u)
        cell = torch.mul(i, u) + torch.mul(f, prev_c)
        hidden = torch.mul(o, F.tanh(cell))
        tree.state = (hidden, cell)
        return tree.state
        
    def init_state(self, requires_grad=True):
        h, c = Variable(torch.zeros(1, self.hidden_size), requires_grad=requires_grad), Variable(torch.zeros(1, self.hidden_size), requires_grad=requires_grad) # 
        if self.cuda_enabled:
            return h.cuda(), c.cuda()
        else:
            return h, c

class ImprovedTreeNode(nn.Module):
    '''
        1. gate function is another formula
        2. support word links
    '''
    def __init__(self, input_size, hidden_size, output_size, vocab_size, 
                 word_scale=0.05, glove_mat=None, cuda_enabled=True, 
                 link_words=False, gate_type=0, gate_factor=1.0, gate_bias=0.0):
        super(ImprovedTreeNode, self).__init__()
        
        self.cuda_enabled = cuda_enabled
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size

        if glove_mat is not None:
            self.init_embeddings(glove_mat)
        else:
            self.embeds = nn.Embedding(vocab_size, self.input_size)
            self.embeds.weight.data.uniform_(-word_scale, word_scale) # uniform vs guassion distribution
        
        if self.cuda_enabled: self.embeds = self.embeds.cuda()
        
        self.encode_inputx = nn.Linear(self.input_size, 4 * self.hidden_size)  # only used to encode word
        # hidden state of last child is input of parent, encode as input
        self.encode_inputh = nn.Linear(self.hidden_size, 4 * self.hidden_size) 
        self.encode_prevh = nn.Linear(self.hidden_size, 4 * self.hidden_size)  # encode hidden state 
        
        self.link_words = link_words
        self.last_child_state = None
        self.gate_type = gate_type
        self.gate_factor = gate_factor
        self.gate_bias = gate_bias
        #self.iofu = nn.Linear(self.hidden_size, self.hidden_size * 4)
    
    def toggle_embeddings_state(self, state=False):
        self.embeds.weight.requires_grad = state
    
    def toggle_words_link(self, state=False):
        self.link_words = state
        self.reset()
    
    def reset(self):
        self.last_child_state = None
        
    def init_embeddings(self, mat): # for futures's glove, word2vec and others 
        vocab_size, self.input_size = mat.size() # override the input_size
        assert vocab_size >= self.vocab_size
        
        self.embeds = nn.Embedding(vocab_size, self.input_size)
        self.embeds.weight.data.copy_(mat)
        
    def forward(self, tree, start_ele, prev_h, prev_c):
        assert len(tree.children) != 0 or tree.word is not None
        assert not (len(tree.children) != 0 and tree.word is not None) # move assert code into tree
        
        if self.link_words and start_ele and (self.last_child_state is not None):
            prev_h, prev_c = self.last_child_state # skip the first word of sentence.
        
        if tree.getType(): # mid node
            tree.zero_state = self.init_state()
            self.forward(tree.children[0], True, * tree.zero_state)
            for idx in range(1, len(tree.children)):
                self.forward(tree.children[idx], False, *tree.children[idx-1].state)
            
            prev_ch, _ = tree.children[-1].state
            if self.hidden_size == self.input_size: # when input_x(word_embedding) == hidden_size, use same weights
                hx = self.encode_inputx(prev_ch) + self.encode_prevh(prev_h) # copy prev_ch or reference it?????  reference makes connected
            else:
                # last child's hidden is input; prev_h is sibling's hidden   # backward of mid-Node
                hx = self.encode_inputh(prev_ch) + self.encode_prevh(prev_h) # above.   should use reference
            
        else: # Leaf Node
            input_idx = Variable(torch.LongTensor([tree.word]))
            input_x = self.embeds(input_idx.cuda() if self.cuda_enabled else input_idx) # embedding is cuda/cpu, so input_x is correponding one
            hx = self.encode_inputx(input_x) + self.encode_prevh(prev_h) # prev_h is given as params (siblings or zeros)
        
        #iofu = self.iofu(hx)
        # why the Improved failed? 
        # xi, xo, xf, xu = torch.split(x, x.size(1) // 4, dim=1)
        # hi, ho, hf, hu = torch.split(h, h.size(1) // 4, dim=1)
        # i = 2 * F.sigmoid(xi) - 1 + 2 * F.sigmoid(hi) - 1
        # o = 2 * F.sigmoid(xo) - 1 + 2 * F.sigmoid(ho) - 1
        # f = 2 * F.sigmoid(xf) - 1 + 2 * F.sigmoid(hf) - 1
        # u = F.tanh(xu + hu)
        i, o, f, u = torch.split(hx, hx.size(1) // 4, dim=1)
        i, o, f, u = gate_factory(i, self.gate_type, self.gate_factor, self.gate_bias), \
                     gate_factory(o, self.gate_type, self.gate_factor, self.gate_bias), \
                     gate_factory(f, self.gate_type , self.gate_factor, self.gate_bias), \
                     F.tanh(u)
        cell = torch.mul(i, u) + torch.mul(f, prev_c)
        hidden = torch.mul(o, F.tanh(cell))
        tree.state = (hidden, cell)
        if self.link_words and (not tree.getType()): # no condition on start_ele because mid word update the last child(is last of next wd)
            self.last_child_state = tree.state
        
        return tree.state
        
    def init_state(self, requires_grad=True):
        h, c = Variable(torch.zeros(1, self.hidden_size), requires_grad=requires_grad), \
                Variable(torch.zeros(1, self.hidden_size), requires_grad=requires_grad) # 
        if self.cuda_enabled:
            return h.cuda(), c.cuda()
        else:
            return h, c

class InnerLSTM(nn.Module):
    # TODO, problematic, solving it
    def __init__(self, embedding_dim, hidden_dim, vocab_size, use_gpu):
        super(InnerLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)


    def init_hidden(self):
        h0 = torch.zeros(1, self.hidden_dim)
        c0 = torch.zeros(1, self.hidden_dim)
        if self.use_gpu:
            return (Variable(h0.cuda(), Variable(c0.cuda())))
        else:
            return (Variable(h0), Variable(c0))

    def forward(self, tree_list):
        sentence = [tree.word for tree in tree_list]
        hidden = self.init_hidden()
        for word in sentence:
            word_idx = Variable(torch.LongTensor([word]))
            if self.use_gpu:
                word_idx = word_idx.cuda()
            embeds = self.word_embeddings(word_idx)
            x = embeds.view(1, 1, -1)
            lstm_out, hidden = self.lstm(x, hidden)
        return hidden[0][0], hidden[1][0]
    

class OuterLSTM(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, use_gpu):
        super(OuterLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.encode_x = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.encode_h = nn.Linear(hidden_dim, 4 * hidden_dim)
    
    def init_hidden(self):
        h0 = torch.zeros(1, self.hidden_dim)
        c0 = torch.zeros(1, self.hidden_dim)
        if self.use_gpu:
            return (Variable(h0.cuda(), Variable(c0.cuda())))
        else:
            return (Variable(h0), Variable(c0))

    def node_forward(self, hidden, cell, word_emb):
        hx = self.encode_h(hidden) + self.encode_x(word_emb)
        i, f, o, u =  torch.split(hx, hx.size(1) // 4, dim=1)
        i, f, o, u = F.sigmoid(i), F.sigmoid(f), F.sigmoid(o), F.tanh(u)
        c = torch.mul(f, cell) + torch.mul(i, u)
        h = torch.mul(o, F.tanh(c))
        return (h, c)
    
    def forward(self, sentence):
        h, c = self.init_hidden()
        for word in sentence:
            word_idx = Variable(torch.LongTensor([word]))
            if self.use_gpu:
                word_idx = word_idx.cuda()
            h, c = self.node_forward(h, c, self.word_embeddings(word_idx))
        return h, c
        
    

class PairCrossLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, pair_dim, output_dim, vocab_size, 
                 num_classes, word_scale=0.05, cuda_enbaled=False, glove_mat = None, freeze = False, 
                 use_improved=False, use_link=False, combine_factor=0.0, gate_type=0, gate_factor=1.0, gate_bias=0.0):
        super(PairCrossLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim #
        self.pair_dim = pair_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.freeze = freeze
        
        self.use_improved = use_improved
        self.use_link = use_link
        self.combine_factor = combine_factor

        if not use_improved:
            self.crosslstm = TreeNode(input_dim, hidden_dim, output_dim, vocab_size, word_scale,\
                                      glove_mat, cuda_enbaled, use_link, gate_type)
        else:
            self.crosslstm = ImprovedTreeNode(input_dim, hidden_dim, output_dim, vocab_size, word_scale, \
                                      glove_mat, cuda_enbaled, use_link, gate_type, gate_factor, gate_bias)

        if self.freeze:
            self.crosslstm.toggle_embeddings_state(freeze)

        if self.combine_factor > 1: # concat the hidden
            self.wh = nn.Linear(2 * self.hidden_dim * 2, self.pair_dim)
        else:
            self.wh = nn.Linear(2 * self.hidden_dim, self.pair_dim)
        self.wp = nn.Linear(self.pair_dim, self.num_classes)

    def forward(self, sent_a, sent_b):

        prev_ha, prev_ca = self.crosslstm.init_state()
        prev_hb, prev_cb = self.crosslstm.init_state()
        if not self.use_improved:
            ha, _ = self.crosslstm(sent_a, prev_ha, prev_ca)
            hb, _ = self.crosslstm(sent_b, prev_hb, prev_cb)
        else:
            self.crosslstm.toggle_words_link(self.use_link)
            ha, _ = self.crosslstm(sent_a, True, prev_ha, prev_ca)
            if self.combine_factor >= 0.0:
                lh, _ = self.crosslstm.last_child_state
                if self.combine_factor > 1.0:
                    ha = torch.cat(ha, lh)
                else:
                    ha = ha * self.combine_factor + lh * (1 - self.combine_factor)

            self.crosslstm.toggle_words_link(self.use_link)
            hb, _ = self.crosslstm(sent_b, True, prev_hb, prev_cb)   
            if self.combine_factor >= 0.0:
                lh, _ = self.crosslstm.last_child_state
                if self.combine_factor > 1.0:
                    hb = torch.cat(hb, lh)
                else:
                    hb = hb * self.combine_factor + lh * (1 - self.combine_factor)
                

        mult_dist = torch.mul(ha, hb)
        abs_dist = torch.abs(torch.add(ha, -hb))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out))
        return out
