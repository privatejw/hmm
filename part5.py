#!/usr/bin/python3
import argparse
import re
import pandas as pd
from functools import lru_cache

class HMM:
    def __init__(self):
        self.tags = []
        self.words = []
        self.b = {}
        self.a = {}
    
    def e(self, tag, word):
        word = word if word in self.words else '#UNK#'
        return self.b.get(tag, {}).get(word, 0)
    
    def q(self, tu, v):
        count_tu = 0.0
        count_tu_v = 0.0
        t,u = tu
        if tu == ('START', 'START'):
            count_tu = len(self.tY)
            for yy in self.tY:
                if yy[0] == v:
                    count_tu_v += 1
        elif v == 'STOP':
            for i in range(len(self.tY)):
                for j in range(len(self.tY[i])):
                    if j+1 < len(self.tY[i]):
                        if self.tY[i][j] == t and self.tY[i][j+1] == u:
                            count_tu += 1
                            if j+1 == len(self.tY[i]):
                                count_tu_v += 1
        else:
            for i in range(len(self.tY)):
                for j in range(len(self.tY[i])):
                    if j+1 < len(self.tY[i]):
                        if self.tY[i][j] == t and self.tY[i][j+1] == u:
                            count_tu += 1
                            next_y = self.tY[i][j+2] if j+2 < len(self.tY[i]) else 0
                            if next_y == v:
                                count_tu_v += 1
        if count_tu:
            return count_tu_v/count_tu
        return 0

    def setWords(self, x, k):
        counter = {'#UNK#':k}
        to_delete = []
        for line in x:
            for word in line:
                counter[word] = counter.get(word,0) + 1
        for word in counter:
            if counter[word] < k:
                to_delete.append(word)
        for word in to_delete:
            del counter[word]
        self.words =  list(counter.keys())
    
    def setTags(self, y):
        counter = {}
        for line in y:
            for tag in line:
                counter[tag] = 0
        self.tags = list(counter)
    
    def calcB(self):
        self.b = pd.DataFrame(0.0, columns=self.tags, index=self.words)
        for i in range(len(self.tX)):
            for j in range(len(self.tX[i])):
                word = self.tX[i][j]
                word = word if word in self.words else '#UNK#'
                tag = self.tY[i][j]
                self.b[tag][word] += 1
        for tag in self.tags:
            esum = self.b[tag].sum()
            self.b[tag] = self.b[tag].apply(lambda x: x/esum)
    
    def calcA(self):
        tp = self.tags + ['START', 'STOP']
        cols = []
        for t in tp:
            for u in tp:
                cols.append((t,u))
        self.a = pd.DataFrame(0.0, columns=cols, index=tp)
        for v in tp:
            for tu in cols:
                if v == 'START' or 'STOP' in tu:
                    self.a[tu][v] = 0
                else:
                    self.a[tu][v] = self.q(tu, v)
    
    def train(self, tX, tY=[], k=3):
        self.tX = tX
        self.tY = tY
        
        self.setWords(tX, k)
        self.setTags(tY)
        self.calcB()
        self.calcA()
    
    @lru_cache(maxsize=None)
    def pii(self, x, k, uv):
        if k == -1:
            if uv == ('START','START'): return 1
            return 0
        T = self.tags + ['START', 'STOP']
        u,v = uv
        if k == 0:
            if u == 'START': return self.a[('START','START')][v]
            return 0
        return max([self.pii(x, k-1,(t,u)) * self.a[(t,u)][v] * self.e(v, x[k]) for t in T])
    
    def viterbi(self, x):
        # create table to store data
        T = self.tags + ['START', 'STOP']
        TT1 = []
        TT2 = []
        for i in range(len(x)):
            TT1.extend([i]*len(T))
            TT2.extend(T)
        TT = pd.MultiIndex.from_arrays([TT1,TT2], names=('i','prev_tags'))
        df = pd.DataFrame(0.0, columns=TT, index=T)
        
        # generate the table
        for i in range(len(x)):
            for tag1 in self.tags:
                for tag2 in self.tags:
                    df[i][tag1][tag2] = self.pii(x, i, (tag1,tag2))
            
        # generate the tags by backtracking
        opt_y = ['O'] * len(x)
        #opt_y.insert(0,'START')
        opt_y.append('STOP')
        
        for i in range(len(x))[::-1]:
            # get the tag with the highest score
            prev_score = 0
            for tag1 in T:
                for tag2 in T:
                    score = self.a[(tag1,tag2)][opt_y[i+1]] * df[i][tag2][opt_y[i+1]]
                    if score > prev_score:
                        opt_y[i] = tag2
                        prev_score = score
        return opt_y[:-1]
    
    @lru_cache(maxsize=None)
    def alpha(self, x, u, j):
        if j == 0:
            return self.a['START'][u]
        return sum([self.alpha(x,v,j-1) * self.a[v][u] * self.e(v,x[j-1]) for v in self.tags])
    
    @lru_cache(maxsize=None)
    def beta(self, x, u, j):
        if j == len(x):
            return self.a[u]['STOP'] * self.e(u,x[j-1])
        return sum([self.a[u][v] * self.e(u,x[j]) * self.beta(x,v,j+1) for v in self.tags])
    
    def mmd(self, x):
        opt_y = ['O'] * len(x)
        #checkscore = [0] * len(x)
        for i in range(len(x)):
            opt_score = 0
            for tag in self.tags:
                score = self.alpha(x,tag,i) * self.beta(x,tag,i)
                #checkscore[i] += score
                if score > opt_score:
                    opt_score = score
                    opt_y[i] = tag
        #print(checkscore) # every element in checkscore should be identical to pass the checkscore test
        return opt_y
    
    def eval(self, test_text, part=5):
        if part == 5:
            # predict y1*,...,yn* = arg max(y1,...,yn) p(x1, . . . , xn, y1, . . . , yn)
            s = ""
            x = []
            for word in test_text.split('\n'):
                if word == "":
                    if x:
                        y = self.viterbi(tuple(x))
                        s += self.getText(x,y) + '\n'
                        x = []
                else:
                    x.append(word)
        
        else:
            s = "Invalid part. Terminating... "
        
        return s
    
    def getText(self, x, y):
        s = ""
        for i in range(len(x)):
            s += x[i] + ' ' + y[i] + '\n'
        return s

def main(lang, in_file, part, k):
    # get contents of file as message
    with open(('./'+lang+'/train'), 'r', encoding='UTF-8') as input_file:
        message = input_file.read()
    
    tX = []
    tY = []
    x = []
    y = []
    # parse message to get tX and tY
    for line in message.split('\n'):
        # end of tweet is indicated by newline
        if line == "":
            if x:
                tX.append(x)
                tY.append(y)
                x = []
                y = []
            continue
        # otherwise, append to x and y
        val = line.split(' ')
        x.append(val[0])
        y.append(val[1])
    
    # read test file
    with open('./%s/%s'%(lang,in_file), 'r', encoding='UTF-8') as input_file:
        testData = input_file.read()
    
    hmm = HMM()
    hmm.train(tX, tY, k)
    s = hmm.eval(testData, part)
    
    # write output to file
    with open('./%s/dev.p%d.out'%(lang,part), 'w', encoding='UTF-8', newline='\n') as out_file:
        out_file.write(s)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str, dest='lang', default='EN', help="Indicate which folder to get results from. Defaults to 'EN'")
    parser.add_argument('-f', type=str, dest='in_file', default='dev.in', help="Indicate file to predict. Defaults to 'dev.in'")
    parser.add_argument('-p', type=int, dest='part', default=2, help="Indicate which part to do. Defaults to 2")
    parser.add_argument('-k', type=int, dest='k', default=3, help="Indicate value of k. Defaults to 3")
    
    args = parser.parse_args()
    main(args.lang, args.in_file, args.part, args.k)