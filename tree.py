
class Tree(object):
    def __init__(self):
        self.children = list()
        self.node_type = None
        self.word = None
        self.raw_word = None # bak for raw word not idx
        
        self._width = -1 # hidden element
        self._height = 1
    
    def addChild(self, tree):
        self.children.append(tree)
        
    def isMid(self):
        # Tree for mid node and False for leafNode
        return True if len(self.children) > 0 else False
    
    def getType(self):
        return self.isMid()
    
    def hasNodetype(self):
        return False if self.node_type is None else True
    
    def debug_out(self, level=0):
        out = "".join([" "for _ in range(level)])
        out += "%s " % self.node_type
        if len(self.children) == 0:
            out += "=%s" % self.word
        else:
            out += ":"
        print out
        for c in self.children:
            c.debug_out(level+1)
    
    def output(self):
        assert self.word is not None or len(self.children) != 0
        assert not (self.word is not None and len(self.children) > 0)
        assert self.node_type is not None
        ss = ""
        for c in self.children:
            ss += c.output()
        return ss + (" " + self.raw_word if self.word is not None else "")

    def getWidth(self):
        if self._width > 0: return self._width, self._height # skip re-calculate
        if len(self.children) == 0: 
            self._width = max(len(self.node_type), len(self.raw_word)) + 1
            return self._width, self._height
        
        self._width = 0
        for c in self.children:
            w, h = c.getWidth()
            self._width += w
            self._height = max(h, self._height)
        self._height += 1
        self._width += (len(self.node_type) if self.isMid() 
                        else max(len(self.node_type), len(self.raw_word)) + 1)
        return self._width, self._height
    
    def getWH(self):
        return self.getWidth()
    
    def compact(self):
        if len(self.children) == 0: #leaf
            return
        elif len(self.children) == 1:
            #print('Throw %s of %s' %(self.children[0].node_type, self.node_type ))
            if  not self.children[0].isMid():
                self.node_type = self.children[0].node_type
                self.word = self.children[0].word
                self.raw_word = self.children[0].raw_word
                self.children = list()
            else:
                self.children= self.children[0].children
            self.compact()
        else:
            for ix, c in enumerate(self.children):
                c.compact()
    
    def tolist(self, container = None):
        isRoot = False
        if container is None:
            container = list()
            isRoot = True
        if not self.isMid():
            container.append(self)
        else:
            for c in self.children:
                c.tolist(container)
        if isRoot:
            return container
    
    def beautifyOutput(self, canvas=None, offx=None, offy=None):
        isRoot = True if canvas is None else False
        if canvas is None: # root node
            width, height = self.getWidth()
            row = [' ' for i in range(width)]
            canvas = [list(row) for i in range(height*5)] # height-5 for each line
        if offx is None: offx = 0
        if offy is None: offy = 0
        
        title_x = offx + (self._width - len(self.node_type))/2 + 1
        if not isRoot: canvas[offy][offx + self._width/2] = '|'
        canvas[offy+1][title_x:title_x+len(self.node_type)] = self.node_type
        canvas[offy+2][offx + self._width/2] = '|'
        canvas[offy+3][offx + self._width/2] = '|'
        
        if self.isMid():
            left_child_w, _ = self.children[0].getWidth()
            right_child_w, _ = self.children[-1].getWidth()
            banner_x1 = offx + left_child_w/2
            banner_x2 = banner_x1 + self._width - (left_child_w + right_child_w)/2
            canvas[offy+4][banner_x1:banner_x2] = '-'*(banner_x2 - banner_x1)
            offy += 5
            for c in self.children:
                c.beautifyOutput(canvas, offx, offy)
                offx += c.getWidth()[0]
        else:
            canvas[offy+4][offx+1 : offx+1+len(self.raw_word)] = self.raw_word
        if isRoot:
            for i in range(len(canvas)):
                print "".join(canvas[i])
        
        
