if __name__ == '__main__':
    with open(('./EN/train'), 'r', encoding='UTF-8') as input_file:
        message = input_file.read()
    
    x = []
    goodstuff = ['B-positive', 'I-positive']
    for line in message.split('\n'):
        # end of tweet is indicated by newline
        if line == "":
            continue
        # otherwise, append to x and y
        val = line.split(' ')
        if val[1] in goodstuff:
            print(val[0])
            x.append(val[0])
    print(x)