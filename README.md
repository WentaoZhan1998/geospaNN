# NN-GLS
NNGLS package repository for the [NN-GLS paper](https://arxiv.org/pdf/2304.09157.pdf)
=======
This is the package repository for the method proposed in the NN-GLS paper. To install (currently), \
please download the wheel in https://github.com/WentaoZhan1998/NN-GLS/dist/nngls-0.1.0-py3-none-any.whl to the working directory and 
run

```commandline\
wget https://github.com/WentaoZhan1998/NN-GLS/dist/nngls-0.1.0-py3-none-any.whl
pip3 install ./nngls-0.1.0-py3-none-any.whl.
```

```commandline
#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'groupSort' function below.
#
# The function is expected to return a 2D_INTEGER_ARRAY.
# The function accepts INTEGER_ARRAY arr as parameter.
#
import numpy as np

def groupSort(arr):
    res_table = {}
    for item in arr:
        if item not in res_table.keys():
            res_table[item] = 0
        res_table[item] += 1
    
    res = np.concatenate([np.array(list(res_table.keys()), dtype = int).reshape(-1,1), 
    np.array(list(res_table.values()), dtype = int).reshape(-1,1)], axis = 1)
    res = res[np.lexsort((res[:,0], -res[: ,1]))]
    return res
```

```commandline

```

