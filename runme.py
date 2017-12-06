# short script to get all the results

import os

langs = ['EN', 'CN', 'FR', 'SG']
parts = [2, 3, 4]

for lang in langs:
    for part in parts:
        cmd = 'python3 evalResult.py ./%s/dev.out ./%s/dev.p%d.out'%(lang,lang,part)
        print(cmd)
        os.system(cmd)
        print('\n')