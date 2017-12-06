#!/usr/bin/python3
import argparse
import re
import pandas as pd
from functools import lru_cache

""" Viterbi algorithm
Base case: 
pi(o,v) = 1 if v=START else 0; pi(1,v) = a START,v * bv(x1)

Recursive case:
pi(i,v) = max u pi(i-1,u) * a u,v * bv(xi)

Final case:
pi(n+1, STOP) = max u pi(n, u) * a u,STOP

Table stores the optimal score as well as the optimal parent

If optimal parent is not stored, our space complexity is halved, while the backtracking algorithm to generate the y sequence is O(n*t) instead of O(n). 
O(n,t) algo is as follows:
y* n+1 = STOP
y* n   = argmax u pi(v,u) * a u,STOP
y* i-1 = argmax u pi(i-1, u) * a u,yi*
"""

class HMM:
    def __init__(self):
        self.tags = []
        self.words = []  # contains every word that appears at least k times in the training set
        self.b = {}
        self.a = {}
        self.words5 = [] # contains every word in the training set
        self.b5 = {}
    
    def e(self, tag, word):
        word = word if word in self.words else '#UNK#'
        return self.b.get(tag, {}).get(word, 0)
    
    def q(self, yp, y):
        count_yp = 0.0
        count_yp_y = 0.0
        if yp == 'START':
            count_yp = len(self.tY)
            for yy in self.tY:
                if yy[0] == y:
                    count_yp_y += 1
        elif y == 'STOP':
            for i in range(len(self.tY)):
                for j in range(len(self.tY[i])):
                    if self.tY[i][j] == yp:
                        count_yp += 1
                        if j+1 == len(self.tY[i]):
                            count_yp_y += 1
        else:
            for i in range(len(self.tY)):
                for j in range(len(self.tY[i])):
                    if self.tY[i][j] == yp:
                        count_yp += 1
                        next_y = self.tY[i][j+1] if j+1 < len(self.tY[i]) else 0
                        if next_y == y:
                            count_yp_y += 1
        return count_yp_y/count_yp

    def setWords(self, x, k):
        counter = {'#UNK#':k}
        to_delete = []
        for line in x:
            for word in line:
                counter[word] = counter.get(word,0) + 1
        for word in counter:
            if counter[word] < k:
                to_delete.append(word)
        self.words5 = list(counter.keys()) # for part 5
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
        transition_params = self.tags + ['START', 'STOP']
        self.a = pd.DataFrame(0.0, columns=transition_params, index=transition_params)
        for i in transition_params:
            for j in transition_params:
                if j == 'START' or i == 'STOP':
                    self.a[i][j] = 0
                    continue
                self.a[i][j] = self.q(i, j)
    
    def train(self, tX, tY=[], k=3):
        self.tX = tX
        self.tY = tY
        
        self.setWords(tX, k)
        self.setTags(tY)
        self.calcB()
        self.calcA()
    
    @lru_cache(maxsize=None)
    def pii(self, x, k, v):
        if k == -1:
            if v == 'START': return 1
            return 0
        T = self.tags + ['START', 'STOP']
        return max([self.pii(x, k-1,u) * self.a[u][v] * self.e(v, x[k]) for u in T])
    
    def viterbi(self, x):
        # create table to store data
        df = pd.DataFrame(0.0, columns=range(len(x)), index=self.tags)
        
        # generate the table
        for i in range(len(x)):
            for tag in self.tags:
                df[i][tag] = self.pii(x, i, tag)
            
        # generate the tags by backtracking
        opt_y = ['O'] * len(x)
        opt_y.append('STOP')
        for i in range(len(x))[::-1]:
            # get the tag with the highest score
            prev_score = 0
            for tag in self.tags:
                score = self.a[tag][opt_y[i+1]] * df[i][tag]
                if score > prev_score:
                    opt_y[i] = tag
                    prev_score = score
        return opt_y[:-1]
    
    def viterbi2(self, x):
        # generate the table df
        YP = ['START']
        tps = self.tags
        df = pd.DataFrame(0.0, columns=range(len(x)), index=tps)
        for i in range(len(x)):
            word = x[i]
            word = word if word in self.words else '#UNK#'
            for tag in tps:
                for yp in YP:
                    score = self.a[yp][tag] * self.e(tag, word)
                    if score > df[i][tag]:
                        df[i][tag] = score
            YP = tps
        
        # generate the tags by backtracking
        opt_y = ['O'] * len(x)
        opt_y.append('STOP')
        for i in range(len(x))[::-1]:
            prev_score = 0
            for tag in tps:
                score = self.a[tag][opt_y[i+1]] * df[i][tag]
                if score > prev_score:
                    opt_y[i] = tag
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
    
    def mmd2(self, x):
        if len(x) == 0:
            return []
        
        pi_forward = pd.DataFrame(0, columns = range(len(x)), index = self.tags)
        pi_backward = pd.DataFrame(0, columns = range(len(x)), index = self.tags)

        for j1 in range(len(self.tags)):
            pi_forward.iloc[j1, 0] = self.a['START'][self.tags[j1]]
            pi_backward.iloc[j1, len(x)-1] = self.a[self.tags[j1]]['STOP']*self.e(self.tags[j1], x[len(x)-1])

        for i in range(1, len(x)):
            for j2 in range(len(self.tags)):
                Sum = 0
                for j1 in range(len(self.tags)):
                    Sum = Sum + pi_forward.iloc[j1, i-1]*self.a[self.tags[j1]][self.tags[j2]]*self.e(self.tags[j1], x[i-1])
                pi_forward.iloc[j2, i] = Sum

        for i in range(len(x) - 2, -1, -1):
            for j2 in range(len(self.tags)):
                Sum = 0
                for j1 in range(len(self.tags)):
                    Sum = Sum + pi_backward.iloc[j1, i+1]*self.a[self.tags[j2]][self.tags[j1]]*self.e(self.tags[j2], x[i])
                pi_backward.iloc[j2, i] = Sum

        y = []
        #checkscore = [0] * len(x)
        for i in range(len(x)):
            max = 0
            k = 0
            for j in range(len(self.tags)):
                score = pi_forward.iloc[j, i] * pi_backward.iloc[j, i]
                #checkscore[i] += score
                if score > max:
                    max = score
                    k = j
            y.append(self.tags[k])
        #print(checkscore) # every element in checkscore should be identical to pass the checkscore test

        return y
    
    # -------------- part 5 --------------
    def e5(self, tag, word):
        word = word if word in self.words5 else '#UNK#'
        return self.b5.get(tag, {}).get(word, 0)
    def calcB5(self):
        self.b5 = pd.DataFrame(0.0, columns=self.tags, index=self.words5)
        for i in range(len(self.tX)):
            for j in range(len(self.tX[i])):
                word = self.tX[i][j]
                tag = self.tY[i][j]
                self.b5[tag][word] += 1
                if word not in self.words:
                    self.b5[tag]['#UNK#'] += 1
        for tag in self.tags:
            esum = self.b5[tag].sum()
            self.b5[tag] = self.b5[tag].apply(lambda x: x/esum)
    @lru_cache(maxsize=None)
    def pii5(self, x, k, v):
        if k == -1:
            if v == 'START': return 1
            return 0
        T = self.tags + ['START', 'STOP']
        return max([self.pii5(x, k-1,u) * self.a[u][v] * self.e5(v, x[k]) for u in T])
    def viterbi5(self, x):
        # create table to store data
        df = pd.DataFrame(0.0, columns=range(len(x)), index=self.tags)
        
        # generate the table
        for i in range(len(x)):
            for tag in self.tags:
                df[i][tag] = self.pii5(x, i, tag)
            
        # generate the tags by backtracking
        opt_y = ['O'] * len(x)
        opt_y.append('STOP')
        for i in range(len(x))[::-1]:
            # get the tag with the highest score
            prev_score = 0
            for tag in self.tags:
                score = self.a[tag][opt_y[i+1]] * df[i][tag]
                if score > prev_score:
                    opt_y[i] = tag
                    prev_score = score
        return opt_y[:-1]
    @lru_cache(maxsize=None)
    def alpha5(self, x, u, j):
        if j == 0:
            return self.a['START'][u]
        return sum([self.alpha5(x,v,j-1) * self.a[v][u] * self.e5(v,x[j-1]) for v in self.tags])
    @lru_cache(maxsize=None)
    def beta5(self, x, u, j):
        if j == len(x):
            return self.a[u]['STOP'] * self.e5(u,x[j-1])
        return sum([self.a[u][v] * self.e5(u,x[j]) * self.beta5(x,v,j+1) for v in self.tags])
    def mmd5(self, x):
        opt_y = ['O'] * len(x)
        #checkscore = [0] * len(x)
        for i in range(len(x)):
            opt_score = 0
            for tag in self.tags:
                score = self.alpha5(x,tag,i) * self.beta5(x,tag,i)
                #checkscore[i] += score
                if score > opt_score:
                    opt_score = score
                    opt_y[i] = tag
        #print(checkscore) # every element in checkscore should be identical to pass the checkscore test
        return opt_y
    # -------------- part 5 end --------------
    
    def eval(self, test_text, part=2):
        if part == 2:
            s = ""
            opt_tags = {}
                
            for tag in self.tags:
                for word in self.words:
                    curr = self.e(tag, word)
                    if curr > opt_tags.get(word, (0,0))[1]:
                        opt_tags[word] = (tag, curr)
            
            for x in test_text.split('\n'):
                # end of tweet is indicated by newline
                if x == "":
                    s += '\n'
                    continue
                # otherwise, predict tag
                opt_tag = opt_tags.get(x, ('O', 0))[0]
                s += x + ' ' + opt_tag + '\n'
        
        elif part == 3:
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
        
        elif part == 4:
            s = ""
            x = []
            for word in test_text.split('\n'):
                if word == "":
                    if x:
                        y = self.mmd(tuple(x))
                        s += self.getText(x,y) + '\n'
                        x = []
                else:
                    x.append(word)
        
        elif part == 5:
            self.calcB5()
            # predict y1*,...,yn* = arg max(y1,...,yn) p(x1, . . . , xn, y1, . . . , yn)
            s = ""
            x = []
            for word in test_text.split('\n'):
                if word == "":
                    if x:
                        y = self.viterbi5(tuple(x))
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