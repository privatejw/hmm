# hmm
Machine Learning Hidden Markov Model

Run using
```terminal
$ python hmm.py
```

If using a language other than EN, such as CN
```terminal
$ python hmm.py -l CN
```

To specify part 4, 
```terminal
$ python hmm.py -l CN -p 4
```

For help, 
```terminal
$ python hmm.py -h

usage: hmm.py [-h] [-l LANG] [-f IN_FILE] [-p PART] [-k K]

optional arguments:
  -h, --help  show this help message and exit
  -l LANG     Indicate which folder to get results from. Defaults to 'EN'
  -f IN_FILE  Indicate file to predict. Defaults to 'dev.in'
  -p PART     Indicate which part to do. Defaults to 2
  -k K        Indicate value of k. Defaults to 3
```

For part 5, we tried several other methods which were not used in the final file. These methods have been detailed in p5(only works for EN).py, 
```terminal
$ python "p5(only works for EN).py"
```

Please ensure that you are running the script in the current directory. 