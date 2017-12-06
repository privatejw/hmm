import pandas as pd

##Read the data with pandas dataframe
train_df = pd.read_csv('train',sep = ' ', names =['Text', 'Sentiment'], quoting = 3)
train_df['Emission'] = train_df['Sentiment'] + ' ' + train_df['Text']

##Get sentiment tags count
sentiment_count = pd.Series(' '.join(train_df['Sentiment'].astype(str)).split()).value_counts()

##Get emission count observed in training set
emission_count = train_df['Emission'].value_counts()

##Get the count of word given a tag.
O_df = train_df[train_df['Sentiment'] == 'O']
iNeg_df = train_df[train_df['Sentiment'] == 'I-negative']
iPos_df = train_df[train_df['Sentiment'] == 'I-positive']
iNeu_df = train_df[train_df['Sentiment'] == 'I-neutral']
bPos_df = train_df[train_df['Sentiment'] == 'B-positive']
bNeg_df = train_df[train_df['Sentiment'] == 'B-negative']
bNeu_df = train_df[train_df['Sentiment'] == 'B-neutral']

O_wordcount = dict(O_df['Text'].value_counts())
iNeg_wordcount = dict(iNeg_df['Text'].value_counts())
iPos_wordcount = dict(iPos_df['Text'].value_counts())
iNeu_wordcount = dict(iNeu_df['Text'].value_counts())
bPos_wordcount = dict(bPos_df['Text'].value_counts())
bNeg_wordcount = dict(bNeg_df['Text'].value_counts())
bNeu_wordcount = dict(bNeu_df['Text'].value_counts())

##Convert column values to a list to speed things up
corpus = train_df['Text'].tolist()

#Question 2 Part 1 and 2 
def get_emission(x,y):
    
    if x in corpus:
        
        if y == 'O':
            if x in O_wordcount:
                numerator = O_wordcount[x]
            else:
                numerator = 0
            
        elif y == 'I-negative':
            if x in iNeg_wordcount:
                numerator = iNeg_wordcount[x] 
            else:
                numerator = 0
            
        elif y == 'I-positive':
            if x in iPos_wordcount:
                numerator = iPos_wordcount[x] 
            else:
                numerator = 0

        elif y == 'I-neutral':
            if x in iNeu_wordcount:
                numerator = iNeu_wordcount[x] 
            else:
                numerator = 0
            
        elif y == 'B-positive':
            if x in bPos_wordcount:
                numerator = bPos_wordcount[x] 
            else:
                numerator = 0
            
        elif y == 'B-negative':
            if x in bNeg_wordcount:
                numerator = bNeg_wordcount[x] 
            else:
                numerator = 0
            
        elif y == 'B-neutral':
            if x in bNeu_wordcount:
                numerator = bNeu_wordcount[x] 
            else:
                numerator = 0
            
    else:
        # numerator is zero is there is no smoothing
        numerator = 1
        
    #Don't have to add 1 if there is no smoothing
    y_count = sentiment_count[y] + 1
    
    return numerator/y_count

##Question 2 Part 3
def argmax_y_emission(x):
    sent_tags = ['B-positive', 'B-negative', 'B-neutral', 'I-positive', 'I-negative', 'I-neutral', 'O']
    y = []
    scores = []
    for tag in sent_tags: 
        score = get_emission(x,tag)
        y.append(tag)
        scores.append(score)
    max_index = scores.index(max(scores))
    return y[max_index]

##Function to create dev.p2.out
def part2_fscore (x):
    predict = open(x,'r',encoding = 'utf-8')
    predict_list = predict.readlines()
    predict_file = open('dev.p2.out', 'a', encoding = 'utf-8')
    s = ''
    for word in predict_list:
        if word != '\n':
            tag = argmax_y_emission(word.strip())
            s += word.strip() + ' ' + tag + '\n'
        else:
            s += '\n'
    predict_file.write(s)
    predict.close()
    predict_file.close()
    return 'Finished!'

#################################################################################################################################
#read the training set as a string and split the words and tags into separate list 
f = open('train','r',encoding = 'utf-8')
a = f.readlines()

#get the counts of start and stop 
def start_stop_count():
    start_count = a.count('\n')
    stop_count = start_count
    return start_count, stop_count

#get the counts of start -> tag and tag -> stop transitions
def start_stop_transition_count():
    d = dict()
    d['start,' + a[0].split()[1]] = 1 
    for i in range(0,len(a)):
        if a[i] == '\n':
            if i != len(a)-1: 
                previous_sent = a[i-1].split()[1]
                next_sent = a[i+1].split()[1]
                current_sent = 'stop'
            
                if previous_sent + ',' + current_sent not in d:
                    d[previous_sent + ',' + current_sent] = 1
                else: 
                    d[previous_sent + ',' + current_sent] += 1
                
                if 'start,' + next_sent not in d:
                    d['start,' + next_sent] =1 
                else:
                    d['start,' + next_sent] += 1
                
            else:
                previous_sent = a[i-1].split()[1]
                current_sent = 'stop'
                d[previous_sent + ',' + current_sent] += 1
    return d

#compute the start - > tag and tag -> stop transition probabilties 
def start_stop_transition_probabilities():
    d= dict()
    transition = start_stop_transition_count()
    start_count,stop_count = start_stop_count()
    for key in transition:
        base = key.split(',')[1]
        if base == 'stop':
            d[key] = transition[key] / sentiment_count[key.split(',')[0]]
        else:
            d[key] = transition[key] / start_count
    return d

##get tag -> tag counts 
def normal_transition_count():
    d = dict()
    for i in range(0,len(a)-1):
        if a[i] != '\n':
            if a[i+1] != '\n':
                current_sent = a[i].split()[1]
                next_sent = a[i+1].split()[1]
                
                if current_sent + ',' + next_sent not in d:
                    d[current_sent + ',' + next_sent] = 1 
                else:
                    d[current_sent + ',' + next_sent] += 1 
        else:
            current_sent = ''
            next_sent = ''
    return d

## get tag -> tag transition probabilities
def normal_transition_probabilities():
    transition = normal_transition_count()
    d = dict()
    for key in transition:
        base = key.split(',')[0]
        d[key] = transition[key]/float(sentiment_count[base])
    return d
        
normal_transition_prob = normal_transition_probabilities()
start_stop_prob = start_stop_transition_probabilities()

##function to create a dictionary of transition probabilties 
def get_transition_prob(u,v):
    transition = u + ',' + v
    start_stop = start_stop_prob
    normal = normal_transition_prob
    
    if transition in start_stop:
        return start_stop[transition]
    elif transition in normal:
        return normal[transition]
    else:
        return 0
    
#############################################################################################
##Original Viterbi
def interpret_sentence(x):
    sent_tags =['O', 'B-neutral', 'I-neutral','B-positive','I-positive', 'B-negative','I-negative']
    length = len(x.split(' '))
    d = {'start' : 0, 'stop' : [0,'']}
    states = ['start']
    
    for i in range(1,length+1):
        d[i] = {'B-neutral':[0,''], 'I-neutral':[0,''], 'O':[0,''], 'B-negative':[0,''], 'I-negative':[0,''], 'B-positive':[0,''], 'I-positive':[0,'']}
        states.append(i)
    states.append('stop')

    for state in states:
        if state == 'start':
            d[state] = 1 
            
        elif state == 'stop':
            node_score = []
            parent = []
            for key_u in sent_tags:
                previous_score = d[length][key_u][0]
                transition_prob = get_transition_prob(key_u,'stop')
                node_score.append(previous_score*transition_prob)
                parent.append(key_u)
            max_index = node_score.index(max(node_score))
            d[state][0] = max(node_score)
            d[state][1] = parent[max_index]
            
        else:
            if state != 1:
                for key_v in sent_tags:
                    node_score = []
                    parent_score = [] 
                    parent = []
                    emission_prob = get_emission(x.split()[state-1],key_v)
                    for key_u in sent_tags:
                        previous_score = d[state-1][key_u][0]
                        transition_prob = get_transition_prob(key_u,key_v)
                        node_score.append(previous_score*transition_prob*emission_prob)
                        parent_score.append(previous_score*transition_prob)
                        parent.append(key_u)
                    max_index = parent_score.index(max(parent_score))
                    d[state][key_v][0] = max(node_score)
                    d[state][key_v][1] = parent[max_index]
            else:
                for key_v in sent_tags:
                    u = 'start'
                    previous_score = d['start']
                    transition_prob = get_transition_prob(u,key_v)
                    emission_prob = get_emission(x.split()[state-1],key_v)
                    d[state][key_v][0] = previous_score*transition_prob*emission_prob
                    d[state][key_v][1] = 'start'

#########backward
    output_states = []
    for j in range(length,0,-1):
        output_states.append(j)
    y = ''
    output_tags = []
    for o_state in output_states:
        
        if o_state == length:
            y = d['stop'][1]
            output_tags.append(x.split()[o_state -1] + ' ' + y + '\n')
        else:
            y = d[o_state +1][y][1]
            output_tags.append(x.split()[o_state -1] + ' ' + y + '\n')
                

    return output_tags[::-1]

###############################################################################
##Top k viterbi
def k_interpret_sentence(x,k=1):
    sent_tags =['O', 'B-neutral', 'I-neutral','B-positive','I-positive','B-negative','I-negative']
    #sent_tags = ['B-neutral', 'I-neutral', 'O', 'B-negative', 'I-negative', 'B-positive', 'I-positive']
    length = len(x.split(' '))
    d = {'start' : 0, 'stop' : [[],[]]}
    states = ['start']
    
    for i in range(1,length+1):
        #d[i] = {'B-positive': 0, 'B-negative': 0, 'B-neutral': 0, 'I-positive': 0, 'I-negative': 0, 'I-neutral': 0, 'O': 0}
        d[i] = {'B-neutral':[[],[]], 'I-neutral':[[],[]], 'O':[[],[]], 'B-negative':[[],[]], 'I-negative':[[],[]], 'B-positive':[[],[]], 'I-positive':[[],[]]}
        states.append(i)
    states.append('stop')

    for state in states:
        if state == 'start':
            d[state] = 1 
            
        elif state == 'stop':
            #u = max(d[length],key = d[length].get)
            #u = max(d[length].keys(), key = lambda k: d[length][k])
            node_score = []
            parent = []
            for key_u in sent_tags:
                transition_prob = get_transition_prob(key_u,'stop')
                for p in range(0,len(d[length][key_u][0])):
                    previous_score = d[length][key_u][0][p]
                    node_score.append(previous_score*transition_prob)
                    parent.append([key_u,p])
                    
            max_indexes = sorted(range(len(node_score)), key=lambda i: node_score[i])[-k:]
            
            for index in max_indexes:
                d['stop'][1].append(parent[index])
                d['stop'][0].append(node_score[index])
            
        else:
            if state == 1:
                for key_v in sent_tags:
                    u = 'start'
                    previous_score = d['start']
                    transition_prob = get_transition_prob(u,key_v)
                    emission_prob = get_emission(x.split()[state-1],key_v)
                    d[state][key_v][0].append(previous_score*transition_prob*emission_prob)
                    d[state][key_v][1].append('start')
                                   
            else:
                for key_v in sent_tags:
                    node_score = []
                    parent_score = [] 
                    parent = []
                    emission_prob = get_emission(x.split()[state-1],key_v)
                    for key_u in sent_tags:
                    #u = max(d[state-1],key = d[state-1].get)
                    #u = max(d[state-1].keys(), key = lambda k: d[state-1][k])
                        transition_prob = get_transition_prob(key_u,key_v)
                        for p in range(0,len(d[state-1][key_u][0])):
                            previous_score = d[state-1][key_u][0][p]
                            node_score.append(previous_score*transition_prob*emission_prob)
                            parent_score.append(previous_score*transition_prob)
                            parent.append([key_u,p])
                    max_indexes_p = sorted(range(len(parent_score)), key=lambda i: parent_score[i])[-k:]
                    max_indexes_n = sorted(range(len(node_score)), key=lambda i: node_score[i])[-k:]
                    
                    for index in max_indexes_p:
                        d[state][key_v][1].append(parent[index])
                        
                    for index in max_indexes_n:
                        d[state][key_v][0].append(node_score[index])
                    #print(node_score)


#########backward
    output_states = []
    for j in range(length,0,-1):
        output_states.append(j)
    y = []
    output_tags = []
    for o_state in output_states:
        if o_state == length:
            for i in range(0,k):
                output_tags.append([d['stop'][1][i][0]])
                y.append(d['stop'][1][i])

        else:
            for i in range(0,k):
                y[i] = d[o_state +1][y[i][0]][1][y[i][1]]
                output_tags[i].append(y[i][0])

    s = []
    words = x.split()
    result = output_tags[0][::-1]
    for i in range(0,len(words)):
        s .append(words[i] + ' ' + result[i] + '\n')
    return s
    
def convert_data(x):
    devep = open(x,'r',encoding = 'utf-8')
    devep_g = devep.readlines()
    s = ''
    for word in devep_g:
        if word != '\n':
            word = word.strip()
            s += word + ' ' 
        else:
            s += '\n'
    devep.close()
    return s 

def read_data(s):
    list_of_sentence = s.split('\n')
    del list_of_sentence[-1]
    output_file = open('dev.p3.out', 'a', encoding = 'utf-8')
    s = ''
    for sentence in list_of_sentence:
        out = interpret_sentence(sentence.rstrip())
        for i in out:
            s+= i
        s += '\n'
            
    output_file.write(s)
    output_file.close()
    return 'Finished!'

def k_read_data(s):
    list_of_sentence = s.split('\n')
    del list_of_sentence[-1]
    output_file = open('dev.p4.out', 'a', encoding = 'utf-8')
    s = ''
    for sentence in list_of_sentence:
        out = k_interpret_sentence(sentence.rstrip(),5)
        for i in out:
            s+= i
        s += '\n'
            
    output_file.write(s)
    output_file.close()
    return 'Finished!'
    
f.close()

##Only uncomment one of them at any point in time!    

##Create dev.p2.out
print (part2_fscore('dev.in'))

##Create dev.p3.out
#print (read_data(convert_data('dev.in')))

##Create dev.p4.out
#print (k_read_data(convert_data('dev.in')))