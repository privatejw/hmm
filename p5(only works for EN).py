import numpy as np
import pandas as pd

X = []
Y = ['O', 'B-neutral', 'B-negative', 'B-positive', 'I-neutral', 'I-negative', 'I-positive']
ka = 3
kb = 80
train_path = './EN/train'
dev_in_path = './EN/dev.in'
dev_p5_out_path = './EN/dev.p5.out'


count_yx = pd.DataFrame(0, columns = X, index = Y)
Y1 = ['START'] + Y
Y2 = Y + ['STOP']
count_yy = pd.DataFrame(0, columns = Y2, index = Y1)
count_pxy = pd.DataFrame(0, columns = X, index = Y)

with open(train_path) as data:
	y0 = 'START'
	px = '#None#'
	for line in data:
		if line != '\n':
			x = line.split(' ')[0]
			y = line.split(' ')[1].replace('\n', '')
			if not x in list(count_yx):
				count_yx[x] = 0
				count_pxy[x] = 0
			count_yx.loc[y, x] += 1
			count_yy.loc[y0, y] += 1
			if px != '#None#':
				count_pxy.loc[y, px] += 1
			y0 = y
			px = x
		else:
			count_yy.loc[y0, 'STOP'] += 1
			y0 = 'START'
			px = '#None#'

a = count_yx.sum()
UNKa = a[a < ka].index
count_yx['#UNK#'] = count_yx[UNKa].sum(axis = 1)
count_yx = count_yx.drop(UNKa, axis = 1)

count_pxy = count_pxy[['good','best','worst','wonderful','great','was','were','is','it',
                       'are','after','before','in','to','on','up','into','with','or','of',
                       'by','about','like']]
print(count_pxy)

def e(y, x):
	if x in list(count_yx):	
		return count_yx.loc[y, x]/count_yx.loc[y:y].sum(axis = 1).values[0]
	else:
		return count_yx.loc[y, '#UNK#']/count_yx.loc[y:y].sum(axis = 1).values[0]

def q(y1, y2):
	return count_yy.loc[y1, y2]/count_yy.loc[y1:y1].sum(axis =1).values[0]

def p(px, y):
	if px in list(count_pxy):	
		return count_pxy.loc[y, px]/count_pxy[[px]].sum().values[0]
	else:
		# return count_pxy.loc[y, '#UNK#']/count_pxy[['#UNK#']].sum().values[0]
		# return count_pxy.loc[y, '#UNK#']/count_pxy.loc[y:y].sum(axis = 1).values[0]
		return 1/len(Y)

def new_viterbi(x):
	if len(x) == 0:
		return []

	pi_probability = pd.DataFrame(0, columns = range(len(x)), index = Y)
	pi_path = pd.DataFrame('O', columns = range(len(x)), index = Y)

	for j1 in range(len(Y)):
		pi_probability.iloc[j1, 0] = q('START', Y[j1]) * e(Y[j1], x[0])
		pi_path.iloc[j1, 0] = 'START'

	for i in range(1, len(x)):
		for j2 in range(len(Y)):
			for j1 in range(len(Y)):
				if pi_probability.iloc[j1, i-1] * q(Y[j1], Y[j2]) * e(Y[j2], x[i]) * p(x[i-1], Y[j2])> pi_probability.iloc[j2, i]:
					pi_probability.iloc[j2, i] = pi_probability.iloc[j1, i-1] * q(Y[j1], Y[j2]) * e(Y[j2], x[i]) * p(x[i-1], Y[j2])
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
		
def predict5():
	dev_in = open(dev_in_path, 'r')
	dev_p5_out = open(dev_p5_out_path, 'w')
	x = []
	for line in dev_in:
		if line != '\n':
			x.append(line.replace('\n', ''))
		else:
			y = new_viterbi(x)
			for i in range(len(x)):
				dev_p5_out.write(x[i] + ' ' + y[i] + '\n')
			dev_p5_out.write(line)
			x = []

predict5()
