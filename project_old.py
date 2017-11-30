#!/usr/bin/python3
import argparse
import re
import pandas as pd

Y = []
X = []
tp = {}
ep = {}
all_tags = {}

# part 2a: 
def e(x, y):
    count_y = 0.0
    count_y_x = 0.0
    for i in range(len(X)):
        for j in range(len(X[i])):
            if Y[i][j] == y:
                count_y += 1
                if X[i][j] == x:
                    count_y_x += 1
    return count_y_x/count_y

# part 2b: takes in data file as message and count k. Returns the list of all words that appear at least k times. 
def filterKwords(message, k):
    counter = {'#UNK#':k}
    to_delete = []
    for line in message.split('\n'):
        if line == "": continue
        x = line.split(' ')[0]
        counter[x] = counter.get(x,0) + 1
    for word in counter:
        if counter[word] < k:
            to_delete.append(word)
    for word in to_delete:
        del counter[word]
    return list(counter.keys())

# part 3a:
def q(y, yp):
    count_yp = 0.0
    count_yp_y = 0.0
    if yp == 'START':
        count_yp = len(Y)
        for yy in Y:
            if yy[0] == y:
                count_yp_y += 1
    elif y == 'STOP':
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i][j] == yp:
                    count_yp += 1
                    if j+1 == len(Y[i]):
                        count_yp_y += 1
    else:
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i][j] == yp:
                    count_yp += 1
                    next_y = Y[i][j+1] if j+1 < len(Y[i]) else 0
                    if next_y == y:
                        count_yp_y += 1
    return count_yp_y/count_yp

# part 3b:
"""
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
def viterbi(x):
    # generate the table df
    YP = ['START']
    df = pd.DataFrame(0.0, columns=range(len(x)), index=list(ep))
    for i in range(len(x)):
        word = x[i]
        word = word if word in all_words else '#UNK#'
        for tag in ep:
            e_score = ep[tag].get(word, 0)
            for yp in YP:
                score = 100 * tp[yp][tag] * e_score # multiply by 100 to avoid underflow
                if score > df[i][tag]:
                    df[i][tag] = score
        YP = list(ep)
    
    # generate the tags by backtracking
    s = ''
    Yn = ['O'] * len(x)
    Yn.append('STOP')
    for i in range(len(x))[::-1]:
        prev_score = 0
        for tag in ep:
            score = tp[tag][Yn[i+1]] * df[i][tag]
            if score > prev_score:
                Yn[i] = tag
                prev_score = score
        s = x[i] + ' ' + Yn[i] + '\n' + s
    return s
# viterbi2 only takes into account the previous value
def viterbi2(x):
    s = ''
    
    yp = 'START'
    opt_Y = ['O']*len(x)
    for i in range(len(x)):
        word = x[i]
        s += word + ' '
        
        prev_score = 0
        for tag in all_tags:
            score = tp[yp][tag]*all_tags[tag].get(word, 0)
            if score > prev_score:
                opt_Y[i] = tag
                prev_score = score
    
        s += opt_Y[i] + '\n'
        yp = opt_Y[i]
    return s

def main(lang, in_file, part, k):
    # get contents of file as message
    with open(('./'+lang+'/train'), 'r', encoding='UTF-8') as input_file:
        message = input_file.read()
    global all_words
    all_words = filterKwords(message, k)
    
    x = []
    y = []
    # get X and Y from message
    for line in message.split('\n'):
        # end of tweet is indicated by newline
        if line == "":
            if x:
                X.append(x)
                Y.append(y)
                x = []
                y = []
            continue
        # otherwise, append to x and y
        val = line.split(' ')
        x.append(val[0])
        y.append(val[1])
        all_tags[val[1]] = 0
    transition_params = list(all_tags.keys())
    transition_params.insert(0,'START')
    transition_params.append('STOP')
    
    # calculate the probabilities of each word generated from each tag and store result in all_tags
    global ep
    ep = pd.DataFrame(0.0, columns=transition_params, index=all_words)
    for i in range(len(X)):
        for j in range(len(X[i])):
            word = X[i][j]
            word = word if word in all_words else '#UNK#'
            tag = Y[i][j]
            ep[tag][word] += 1
    for tag in transition_params:
        esum = ep[tag].sum()
        ep[tag] = ep[tag].apply(lambda x: x/esum)
    #print(ep.loc['#UNK#'])
    #quit()
    for tag in all_tags:
        y_ = {}
        count_y = 0
        for i in range(len(X)):
            for j in range(len(X[i])):
                if Y[i][j] == tag:
                    count_y += 1
                    word = X[i][j]
                    y_[word] = y_.get(word, 0.0) + 1
        for word in y_:
            y_[word] = y_[word] / count_y
        all_tags[tag] = y_
    
    # read in_file
    with open(('./'+lang+'/'+in_file), 'r', encoding='UTF-8') as input_file:
        message = input_file.read()
    
    if part == 2:
        # for each word, store the tag that maximises y* = arg max(y) e(x|y)
        y_predictor = {}
        for tag in all_tags:
            for word in all_tags[tag]:
                prev = y_predictor.get(word, (tag,0))[1]
                curr = all_tags[tag][word]
                if curr > prev:
                    y_predictor[word] = (tag,curr)
        # predict sentiments for in_file
        s = ""
        for x in message.split('\n'):
            # end of tweet is indicated by newline
            if x == "":
                s += '\n'
                continue
            # otherwise, predict tag
            predicted_tag = y_predictor.get(x, 'O')[0]
            s += x + ' ' + predicted_tag + '\n'
        with open(('./'+lang+'/dev.p2.out'), 'w', encoding='UTF-8', newline='\n') as out_file:
            out_file.write(s[:-1])

    if part == 3:
        # calculate the transition parameters
        global tp
        tp = pd.DataFrame(0.0, columns=transition_params, index=transition_params)
        for yp in transition_params:
            for y in transition_params:
                if y == 'START' or yp == 'STOP':
                    tp[yp][y] = 0
                    continue
                tp[yp][y] = q(y, yp)
        
        # predict y1*,...,yn* = arg max(y1,...,yn) p(x1, . . . , xn, y1, . . . , yn)
        s = ""
        x = []
        for word in message.split('\n'):
            if word == "":
                if x:
                    s += viterbi(x) + '\n'
                    x = []
            else:
                x.append(word)
        # write output to file
        with open(('./'+lang+'/dev.p3.out'), 'w', encoding='UTF-8', newline='\n') as out_file:
            out_file.write(s[:-1])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str, dest='lang', default='EN', help="Indicate which folder to get results from. Defaults to 'EN'")
    parser.add_argument('-f', type=str, dest='in_file', default='dev.in', help="Indicate file to predict. Defaults to 'dev.in'")
    parser.add_argument('-p', type=int, dest='part', default=2, help="Indicate which part to do. Defaults to 2")
    parser.add_argument('-k', type=int, dest='k', default=3, help="Indicate value of k. Defaults to 3")
    
    args = parser.parse_args()
    main(args.lang, args.in_file, args.part, args.k)