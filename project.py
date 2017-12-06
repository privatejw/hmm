import numpy as np
import pandas as pd

X = []
Y = ['O', 'B-neutral', 'B-negative', 'B-positive', 'I-neutral', 'I-negative', 'I-positive']
k = 3
train_path = './EN/train'
dev_in_path = './EN/dev.in'
dev_p2_out_path = './EN/dev.p2.out'
dev_p3_out_path = './EN/dev.p3.out'
dev_p4_out_path = './EN/dev.p4.out'


count_yx = pd.DataFrame(0, columns = X, index = Y)
Y1 = ['START'] + Y
Y2 = Y + ['STOP']
count_yy = pd.DataFrame(0, columns = Y2, index = Y1)

with open(train_path) as data:
	y0 = 'START'
	for line in data:
		if line != '\n':
			x = line.split(' ')[0]
			y = line.split(' ')[1].replace('\n', '')
			if not x in list(count_yx):
				count_yx[x] = 0
			count_yx.loc[y, x] += 1
			count_yy.loc[y0, y] += 1
			y0 = y
		else:
			count_yy.loc[y0, 'STOP'] += 1
			y0 = 'START'
a = count_yx.sum()
UNK = a[a < k].index
count_yx['#UNK#'] = count_yx[UNK].sum(axis = 1)
count_yx = count_yx.drop(UNK, axis = 1)

def e(y, x):
	if x in list(count_yx):	
		return count_yx.loc[y, x]/count_yx.loc[y:y].sum(axis = 1).values[0]
	else:
		return count_yx.loc[y, '#UNK#']/count_yx.loc[y:y].sum(axis = 1).values[0]

def q(y1, y2):
	return count_yy.loc[y1, y2]/count_yy.loc[y1:y1].sum(axis =1).values[0]

#def I2B(tag):
#	return Y[Y.index(tag)-3]

def predict2():
    dev_in = open(dev_in_path, 'r')
    dev_p2_out = open(dev_p2_out_path, 'w')
    # after_O = True
    for line in dev_in:
    	if line != '\n':
    		x = line.replace('\n', '')
    		tag = Y[0]
    		e_max = -1
    		for y in Y:
    			if e(y, x) > e_max:
    				tag = y
    				e_max = e(y, x)
    		#if after_O and tag in Y[3:]:
    		#	tag = I2B(tag)
    		#if tag == 'O':
    		#	after_O = True
    		#else:
    		#	after_O = False
    		dev_p2_out.write(x + ' ' + tag + '\n')
    	else:
    		dev_p2_out.write(line)
    		after_O = True
    dev_in.close()
    dev_p2_out.close()
    return

def viterbi(x):
	if len(x) == 0:
		return []

	pi_probability = pd.DataFrame(0, columns = x, index = Y)
	pi_path = pd.DataFrame('O', columns = x, index = Y)

	for j1 in range(len(Y)):
		pi_probability.iloc[j1, 0] = q('START', Y[j1]) * e(Y[j1], x[0])
		pi_path.iloc[j1, 0] = 'START'

	for i in range(1, len(x)):
		for j2 in range(len(Y)):
			for j1 in range(len(Y)):
				if pi_probability.iloc[j1, i-1] * q(Y[j1], Y[j2]) * e(Y[j2], x[i]) > pi_probability.iloc[j2, i]:
					pi_probability.iloc[j2, i] = pi_probability.iloc[j1, i-1] * q(Y[j1], Y[j2]) * e(Y[j2], x[i])
					pi_path.iloc[j2, i] = Y[j1]
	last_y = 0
	Max = 0
	for j1 in range(len(Y)):
		if pi_probability.iloc[j1, len(X)-1] * q(Y[j1], 'STOP') > Max:
			Max = pi_probability.iloc[j1, len(x)-1] * q(Y[j1], 'STOP')
			last_y = j1
	y = [Y[last_y]]
	for i in range(len(x) - 1, 0, -1):
		y.insert(0, pi_path.iloc[last_y, i])
		last_y = Y.index(pi_path.iloc[last_y, i])

	return y
		
def predict3():
	dev_in = open(dev_in_path, 'r')
	dev_p3_out = open(dev_p3_out_path, 'w')
	x = []
	for line in dev_in:
		if line != '\n':
			x.append(line.replace('\n', ''))
		else:
			y = viterbi(x)
			for i in range(len(x)):
				dev_p3_out.write(x[i] + ' ' + y[i] + '\n')
			dev_p3_out.write(line)
			x = []

def max_marginal(x):
	if len(x) == 0:
		return []
	
	pi_forward = pd.DataFrame(0, columns = x, index = Y)
	pi_backward = pd.DataFrame(0, columns = x, index = Y)

	for j1 in range(len(Y)):
		pi_forward.iloc[j1, 0] = q('START', Y[j1])
		pi_backward.iloc[j1, len(x)-1] = q(Y[j1], 'STOP')*e(Y[j1], x[len(x)-1])

	for i in range(1, len(x)):
		for j2 in range(len(Y)):
			Sum = 0
			for j1 in range(len(Y)):
				Sum = Sum + pi_forward.iloc[j1, i-1]*q(Y[j1], Y[j2])*e(Y[j1], x[i-1])
			pi_forward.iloc[j2, i] = Sum

	for i in range(len(x) - 2, -1, -1):
		for j2 in range(len(Y)):
			Sum = 0
			for j1 in range(len(Y)):
				Sum = Sum + pi_backward.iloc[j1, i+1]*q(Y[j2], Y[j1])*e(Y[j2], x[i])
			pi_backward.iloc[j2, i] = Sum

	y = []
	for i in range(len(x)):
		max = 0
		k = 0
		for j in range(len(Y)):
			if pi_forward.iloc[j, i] * pi_backward.iloc[j, i] > max:
				max = pi_forward.iloc[j, i] * pi_backward.iloc[j, i]
				k = j
		y.append(Y[k])

	return y

def predict4():
	dev_in = open(dev_in_path, 'r')
	dev_p4_out = open(dev_p4_out_path, 'w')
	x = []
	for line in dev_in:
		if line != '\n':
			x.append(line.replace('\n', ''))
		else:
			y = max_marginal(x)
			for i in range(len(x)):
				dev_p4_out.write(x[i] + ' ' + y[i] + '\n')
			dev_p4_out.write(line)
			x = []

predict4()