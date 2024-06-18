```python
import torch
import geospaNN
import numpy as np
import time
import pandas as pd

def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

sigma = 1
phi = 3
tau = 0.01
theta = torch.tensor([sigma, phi / np.sqrt(2), tau])

p = 5; funXY = f5

t = np.empty(0)
size_vec = np.empty(0)
epoch_vec = np.empty(0)


for n in [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
    print(n)
    for rand in range(1, 5 + 1):
        print(rand)
        nn = 20
        batch_size = 50

        torch.manual_seed(2023 + rand)
        X, Y, coord, cov, corerr = geospaNN.Simulation(n, p, nn, funXY, theta, range = [0,1])
        data = geospaNN.make_graph(X, Y, coord, nn)
        train_loader, data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, batch_size = batch_size, neighbor_size = 20,
                                                                         test_proportion = 0.2)

        start_time = time.time()
        mlp = torch.nn.Sequential(
                    torch.nn.Linear(p, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(5, 1)
                )
        model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp = mlp, theta = torch.tensor([1.5, 5, 0.1]))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        Update_init = 0; Update_step = 1; Update_bound = 0.1; patience_half = 5; patience = 10;
        lr_scheduler = geospaNN.LRScheduler(optimizer, patience = patience_half, factor=0.5)
        early_stopping = geospaNN.EarlyStopping(patience=patience, min_delta = 0.001)

        # Training/evaluation loop
        for epoch in range(100):
            print(epoch)
            # Train for one epoch
            model.train()
            if (epoch >= Update_init) & (epoch % Update_step == 0):
                model.theta.requires_grad = True
            else:
                model.theta.requires_grad = False
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                decorrelated_preds, decorrelated_targets, est = model(batch)
                loss = torch.nn.functional.mse_loss(decorrelated_preds[:batch_size], decorrelated_targets[:batch_size])
                metric = torch.nn.functional.mse_loss(est[:batch_size], batch.y[:batch_size])
                loss.backward()
                optimizer.step()
            # Compute predictions on held-out test test
            model.eval()
            _, _, val_est = model(data_val)
            val_loss = torch.nn.functional.mse_loss(val_est, data_val.y).item()
            lr_scheduler(val_loss)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print('End at epoch' + str(epoch))
                break
        end_time = time.time()
        usertime = end_time - start_time
        t = np.append(t, usertime)
        size_vec = np.append(size_vec, n)
        epoch_vec = np.append(epoch_vec, epoch)

        df = pd.DataFrame({'time': t,
                   'epoch': epoch_vec,
                   'size': size_vec
                   })

        df.to_csv("/Users/zhanwentao/Documents/Abhi/NN/NN-GLS/running_time.csv")
```

    200
    1
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    INFO: Early stopping counter 1 of 10
    21
    INFO: Early stopping counter 2 of 10
    22
    INFO: Early stopping counter 3 of 10
    23
    INFO: Early stopping counter 4 of 10
    24
    INFO: Early stopping counter 5 of 10
    25
    Epoch 00026: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    26
    INFO: Early stopping counter 7 of 10
    27
    INFO: Early stopping counter 8 of 10
    28
    INFO: Early stopping counter 9 of 10
    29
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch29
    2
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    INFO: Early stopping counter 1 of 10
    22
    23
    24
    25
    26
    INFO: Early stopping counter 1 of 10
    27
    INFO: Early stopping counter 2 of 10
    28
    29
    INFO: Early stopping counter 1 of 10
    30
    INFO: Early stopping counter 2 of 10
    31
    INFO: Early stopping counter 3 of 10
    32
    INFO: Early stopping counter 4 of 10
    33
    INFO: Early stopping counter 5 of 10
    34
    Epoch 00035: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    35
    INFO: Early stopping counter 7 of 10
    36
    INFO: Early stopping counter 8 of 10
    37
    INFO: Early stopping counter 9 of 10
    38
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch38
    3
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    INFO: Early stopping counter 2 of 10
    13
    INFO: Early stopping counter 3 of 10
    14
    INFO: Early stopping counter 4 of 10
    15
    INFO: Early stopping counter 5 of 10
    16
    Epoch 00017: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    17
    INFO: Early stopping counter 7 of 10
    18
    INFO: Early stopping counter 8 of 10
    19
    INFO: Early stopping counter 9 of 10
    20
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch20
    4
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    INFO: Early stopping counter 1 of 10
    21
    INFO: Early stopping counter 2 of 10
    22
    INFO: Early stopping counter 3 of 10
    23
    INFO: Early stopping counter 4 of 10
    24
    INFO: Early stopping counter 5 of 10
    25
    Epoch 00026: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    26
    INFO: Early stopping counter 7 of 10
    27
    INFO: Early stopping counter 8 of 10
    28
    INFO: Early stopping counter 9 of 10
    29
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch29
    5
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42
    43
    44
    45
    46
    47
    48
    49
    50
    51
    52
    53
    54
    55
    56
    57
    58
    59
    60
    61
    INFO: Early stopping counter 1 of 10
    62
    INFO: Early stopping counter 2 of 10
    63
    64
    65
    66
    67
    68
    69
    70
    INFO: Early stopping counter 1 of 10
    71
    INFO: Early stopping counter 2 of 10
    72
    INFO: Early stopping counter 3 of 10
    73
    INFO: Early stopping counter 4 of 10
    74
    75
    76
    77
    78
    79
    80
    81
    82
    83
    84
    85
    86
    87
    88
    INFO: Early stopping counter 1 of 10
    89
    INFO: Early stopping counter 2 of 10
    90
    INFO: Early stopping counter 3 of 10
    91
    INFO: Early stopping counter 4 of 10
    92
    INFO: Early stopping counter 5 of 10
    93
    94
    95
    96
    97
    98
    INFO: Early stopping counter 1 of 10
    99
    INFO: Early stopping counter 2 of 10
    500
    1
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    INFO: Early stopping counter 1 of 10
    16
    17
    18
    19
    INFO: Early stopping counter 1 of 10
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42
    43
    INFO: Early stopping counter 1 of 10
    44
    45
    46
    47
    INFO: Early stopping counter 1 of 10
    48
    INFO: Early stopping counter 2 of 10
    49
    50
    51
    52
    53
    54
    55
    INFO: Early stopping counter 1 of 10
    56
    57
    58
    59
    60
    61
    62
    63
    INFO: Early stopping counter 1 of 10
    64
    65
    INFO: Early stopping counter 1 of 10
    66
    INFO: Early stopping counter 2 of 10
    67
    68
    69
    INFO: Early stopping counter 1 of 10
    70
    71
    72
    73
    74
    75
    76
    INFO: Early stopping counter 1 of 10
    77
    INFO: Early stopping counter 2 of 10
    78
    79
    80
    81
    INFO: Early stopping counter 1 of 10
    82
    INFO: Early stopping counter 2 of 10
    83
    84
    85
    86
    INFO: Early stopping counter 1 of 10
    87
    INFO: Early stopping counter 2 of 10
    88
    89
    90
    INFO: Early stopping counter 1 of 10
    91
    INFO: Early stopping counter 2 of 10
    92
    INFO: Early stopping counter 3 of 10
    93
    94
    INFO: Early stopping counter 1 of 10
    95
    INFO: Early stopping counter 2 of 10
    96
    97
    98
    INFO: Early stopping counter 1 of 10
    99
    INFO: Early stopping counter 2 of 10
    2
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    INFO: Early stopping counter 1 of 10
    14
    INFO: Early stopping counter 2 of 10
    15
    INFO: Early stopping counter 3 of 10
    16
    INFO: Early stopping counter 4 of 10
    17
    INFO: Early stopping counter 5 of 10
    18
    Epoch 00019: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    19
    INFO: Early stopping counter 7 of 10
    20
    INFO: Early stopping counter 8 of 10
    21
    INFO: Early stopping counter 9 of 10
    22
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch22
    3
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    INFO: Early stopping counter 1 of 10
    23
    24
    25
    INFO: Early stopping counter 1 of 10
    26
    27
    28
    INFO: Early stopping counter 1 of 10
    29
    INFO: Early stopping counter 2 of 10
    30
    31
    INFO: Early stopping counter 1 of 10
    32
    INFO: Early stopping counter 2 of 10
    33
    INFO: Early stopping counter 3 of 10
    34
    35
    INFO: Early stopping counter 1 of 10
    36
    INFO: Early stopping counter 2 of 10
    37
    INFO: Early stopping counter 3 of 10
    38
    INFO: Early stopping counter 4 of 10
    39
    40
    INFO: Early stopping counter 1 of 10
    41
    INFO: Early stopping counter 2 of 10
    42
    INFO: Early stopping counter 3 of 10
    43
    INFO: Early stopping counter 4 of 10
    44
    INFO: Early stopping counter 5 of 10
    45
    Epoch 00046: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    46
    INFO: Early stopping counter 7 of 10
    47
    INFO: Early stopping counter 8 of 10
    48
    INFO: Early stopping counter 9 of 10
    49
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch49
    4
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    INFO: Early stopping counter 1 of 10
    13
    INFO: Early stopping counter 2 of 10
    14
    INFO: Early stopping counter 3 of 10
    15
    INFO: Early stopping counter 4 of 10
    16
    INFO: Early stopping counter 5 of 10
    17
    Epoch 00018: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    INFO: Early stopping counter 1 of 10
    35
    INFO: Early stopping counter 2 of 10
    36
    37
    38
    39
    40
    41
    42
    43
    44
    45
    46
    47
    48
    49
    50
    51
    INFO: Early stopping counter 1 of 10
    52
    53
    INFO: Early stopping counter 1 of 10
    54
    55
    INFO: Early stopping counter 1 of 10
    56
    INFO: Early stopping counter 2 of 10
    57
    INFO: Early stopping counter 3 of 10
    58
    INFO: Early stopping counter 4 of 10
    59
    60
    61
    INFO: Early stopping counter 1 of 10
    62
    INFO: Early stopping counter 2 of 10
    63
    INFO: Early stopping counter 3 of 10
    64
    INFO: Early stopping counter 4 of 10
    65
    66
    INFO: Early stopping counter 1 of 10
    67
    INFO: Early stopping counter 2 of 10
    68
    69
    INFO: Early stopping counter 1 of 10
    70
    INFO: Early stopping counter 2 of 10
    71
    INFO: Early stopping counter 3 of 10
    72
    INFO: Early stopping counter 4 of 10
    73
    INFO: Early stopping counter 5 of 10
    74
    INFO: Early stopping counter 6 of 10
    75
    76
    INFO: Early stopping counter 1 of 10
    77
    INFO: Early stopping counter 2 of 10
    78
    INFO: Early stopping counter 3 of 10
    79
    Epoch 00080: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 4 of 10
    80
    INFO: Early stopping counter 5 of 10
    81
    INFO: Early stopping counter 6 of 10
    82
    INFO: Early stopping counter 7 of 10
    83
    84
    INFO: Early stopping counter 1 of 10
    85
    INFO: Early stopping counter 2 of 10
    86
    INFO: Early stopping counter 3 of 10
    87
    INFO: Early stopping counter 4 of 10
    88
    INFO: Early stopping counter 5 of 10
    89
    Epoch 00090: reducing learning rate of group 0 to 1.2500e-03.
    INFO: Early stopping counter 6 of 10
    90
    INFO: Early stopping counter 7 of 10
    91
    INFO: Early stopping counter 8 of 10
    92
    INFO: Early stopping counter 9 of 10
    93
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch93
    5
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    INFO: Early stopping counter 1 of 10
    16
    INFO: Early stopping counter 2 of 10
    17
    INFO: Early stopping counter 3 of 10
    18
    19
    20
    INFO: Early stopping counter 1 of 10
    21
    INFO: Early stopping counter 2 of 10
    22
    INFO: Early stopping counter 3 of 10
    23
    INFO: Early stopping counter 4 of 10
    24
    INFO: Early stopping counter 5 of 10
    25
    Epoch 00026: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    26
    INFO: Early stopping counter 7 of 10
    27
    INFO: Early stopping counter 8 of 10
    28
    INFO: Early stopping counter 9 of 10
    29
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch29
    1000
    1
    0
    1
    2
    3
    4
    5
    6
    7
    INFO: Early stopping counter 1 of 10
    8
    INFO: Early stopping counter 2 of 10
    9
    INFO: Early stopping counter 3 of 10
    10
    11
    12
    13
    14
    15
    16
    INFO: Early stopping counter 1 of 10
    17
    18
    INFO: Early stopping counter 1 of 10
    19
    20
    INFO: Early stopping counter 1 of 10
    21
    INFO: Early stopping counter 2 of 10
    22
    INFO: Early stopping counter 3 of 10
    23
    INFO: Early stopping counter 4 of 10
    24
    25
    INFO: Early stopping counter 1 of 10
    26
    INFO: Early stopping counter 2 of 10
    27
    INFO: Early stopping counter 3 of 10
    28
    INFO: Early stopping counter 4 of 10
    29
    30
    INFO: Early stopping counter 1 of 10
    31
    INFO: Early stopping counter 2 of 10
    32
    INFO: Early stopping counter 3 of 10
    33
    34
    35
    36
    37
    38
    INFO: Early stopping counter 1 of 10
    39
    40
    41
    INFO: Early stopping counter 1 of 10
    42
    INFO: Early stopping counter 2 of 10
    43
    INFO: Early stopping counter 3 of 10
    44
    45
    INFO: Early stopping counter 1 of 10
    46
    INFO: Early stopping counter 2 of 10
    47
    48
    INFO: Early stopping counter 1 of 10
    49
    INFO: Early stopping counter 2 of 10
    50
    INFO: Early stopping counter 3 of 10
    51
    INFO: Early stopping counter 4 of 10
    52
    53
    INFO: Early stopping counter 1 of 10
    54
    INFO: Early stopping counter 2 of 10
    55
    INFO: Early stopping counter 3 of 10
    56
    INFO: Early stopping counter 4 of 10
    57
    INFO: Early stopping counter 5 of 10
    58
    Epoch 00059: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    59
    INFO: Early stopping counter 7 of 10
    60
    INFO: Early stopping counter 8 of 10
    61
    INFO: Early stopping counter 9 of 10
    62
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch62
    2
    0
    1
    2
    3
    4
    INFO: Early stopping counter 1 of 10
    5
    INFO: Early stopping counter 2 of 10
    6
    INFO: Early stopping counter 3 of 10
    7
    INFO: Early stopping counter 4 of 10
    8
    9
    10
    11
    12
    13
    14
    INFO: Early stopping counter 1 of 10
    15
    16
    INFO: Early stopping counter 1 of 10
    17
    18
    INFO: Early stopping counter 1 of 10
    19
    INFO: Early stopping counter 2 of 10
    20
    INFO: Early stopping counter 3 of 10
    21
    INFO: Early stopping counter 4 of 10
    22
    INFO: Early stopping counter 5 of 10
    23
    Epoch 00024: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    24
    INFO: Early stopping counter 7 of 10
    25
    INFO: Early stopping counter 8 of 10
    26
    INFO: Early stopping counter 9 of 10
    27
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch27
    3
    0
    1
    2
    3
    4
    5
    INFO: Early stopping counter 1 of 10
    6
    INFO: Early stopping counter 2 of 10
    7
    8
    9
    10
    INFO: Early stopping counter 1 of 10
    11
    12
    13
    14
    INFO: Early stopping counter 1 of 10
    15
    INFO: Early stopping counter 2 of 10
    16
    17
    INFO: Early stopping counter 1 of 10
    18
    19
    20
    INFO: Early stopping counter 1 of 10
    21
    INFO: Early stopping counter 2 of 10
    22
    INFO: Early stopping counter 3 of 10
    23
    24
    25
    INFO: Early stopping counter 1 of 10
    26
    27
    INFO: Early stopping counter 1 of 10
    28
    INFO: Early stopping counter 2 of 10
    29
    INFO: Early stopping counter 3 of 10
    30
    INFO: Early stopping counter 4 of 10
    31
    INFO: Early stopping counter 5 of 10
    32
    Epoch 00033: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    33
    INFO: Early stopping counter 7 of 10
    34
    INFO: Early stopping counter 8 of 10
    35
    INFO: Early stopping counter 9 of 10
    36
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch36
    4
    0
    1
    2
    3
    4
    5
    6
    INFO: Early stopping counter 1 of 10
    7
    8
    9
    10
    11
    12
    13
    INFO: Early stopping counter 1 of 10
    14
    15
    16
    INFO: Early stopping counter 1 of 10
    17
    INFO: Early stopping counter 2 of 10
    18
    INFO: Early stopping counter 3 of 10
    19
    20
    INFO: Early stopping counter 1 of 10
    21
    INFO: Early stopping counter 2 of 10
    22
    INFO: Early stopping counter 3 of 10
    23
    INFO: Early stopping counter 4 of 10
    24
    INFO: Early stopping counter 5 of 10
    25
    INFO: Early stopping counter 6 of 10
    26
    INFO: Early stopping counter 7 of 10
    27
    INFO: Early stopping counter 8 of 10
    28
    INFO: Early stopping counter 9 of 10
    29
    Epoch 00030: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch29
    5
    0
    1
    2
    3
    INFO: Early stopping counter 1 of 10
    4
    INFO: Early stopping counter 2 of 10
    5
    INFO: Early stopping counter 3 of 10
    6
    7
    8
    9
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    INFO: Early stopping counter 2 of 10
    13
    INFO: Early stopping counter 3 of 10
    14
    INFO: Early stopping counter 4 of 10
    15
    INFO: Early stopping counter 5 of 10
    16
    Epoch 00017: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    17
    INFO: Early stopping counter 7 of 10
    18
    INFO: Early stopping counter 8 of 10
    19
    INFO: Early stopping counter 9 of 10
    20
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch20
    2000
    1
    0
    1
    2
    3
    INFO: Early stopping counter 1 of 10
    4
    INFO: Early stopping counter 2 of 10
    5
    6
    7
    8
    9
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    INFO: Early stopping counter 2 of 10
    13
    14
    INFO: Early stopping counter 1 of 10
    15
    INFO: Early stopping counter 2 of 10
    16
    17
    INFO: Early stopping counter 1 of 10
    18
    INFO: Early stopping counter 2 of 10
    19
    20
    INFO: Early stopping counter 1 of 10
    21
    22
    23
    INFO: Early stopping counter 1 of 10
    24
    25
    INFO: Early stopping counter 1 of 10
    26
    INFO: Early stopping counter 2 of 10
    27
    INFO: Early stopping counter 3 of 10
    28
    INFO: Early stopping counter 4 of 10
    29
    INFO: Early stopping counter 5 of 10
    30
    Epoch 00031: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    31
    INFO: Early stopping counter 7 of 10
    32
    INFO: Early stopping counter 8 of 10
    33
    34
    INFO: Early stopping counter 1 of 10
    35
    INFO: Early stopping counter 2 of 10
    36
    INFO: Early stopping counter 3 of 10
    37
    INFO: Early stopping counter 4 of 10
    38
    INFO: Early stopping counter 5 of 10
    39
    Epoch 00040: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 6 of 10
    40
    INFO: Early stopping counter 7 of 10
    41
    INFO: Early stopping counter 8 of 10
    42
    INFO: Early stopping counter 9 of 10
    43
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch43
    2
    0
    1
    2
    3
    4
    5
    6
    INFO: Early stopping counter 1 of 10
    7
    8
    9
    10
    11
    12
    13
    14
    INFO: Early stopping counter 1 of 10
    15
    INFO: Early stopping counter 2 of 10
    16
    17
    INFO: Early stopping counter 1 of 10
    18
    INFO: Early stopping counter 2 of 10
    19
    INFO: Early stopping counter 3 of 10
    20
    INFO: Early stopping counter 4 of 10
    21
    INFO: Early stopping counter 5 of 10
    22
    INFO: Early stopping counter 6 of 10
    23
    INFO: Early stopping counter 7 of 10
    24
    Epoch 00025: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 8 of 10
    25
    INFO: Early stopping counter 9 of 10
    26
    27
    INFO: Early stopping counter 1 of 10
    28
    INFO: Early stopping counter 2 of 10
    29
    INFO: Early stopping counter 3 of 10
    30
    INFO: Early stopping counter 4 of 10
    31
    INFO: Early stopping counter 5 of 10
    32
    INFO: Early stopping counter 6 of 10
    33
    INFO: Early stopping counter 7 of 10
    34
    35
    INFO: Early stopping counter 1 of 10
    36
    INFO: Early stopping counter 2 of 10
    37
    INFO: Early stopping counter 3 of 10
    38
    INFO: Early stopping counter 4 of 10
    39
    INFO: Early stopping counter 5 of 10
    40
    Epoch 00041: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 6 of 10
    41
    INFO: Early stopping counter 7 of 10
    42
    INFO: Early stopping counter 8 of 10
    43
    INFO: Early stopping counter 9 of 10
    44
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch44
    3
    0
    1
    2
    3
    4
    INFO: Early stopping counter 1 of 10
    5
    6
    7
    8
    9
    INFO: Early stopping counter 1 of 10
    10
    INFO: Early stopping counter 2 of 10
    11
    INFO: Early stopping counter 3 of 10
    12
    INFO: Early stopping counter 4 of 10
    13
    INFO: Early stopping counter 5 of 10
    14
    Epoch 00015: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    15
    INFO: Early stopping counter 7 of 10
    16
    INFO: Early stopping counter 8 of 10
    17
    INFO: Early stopping counter 9 of 10
    18
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch18
    4
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    4
    5
    6
    7
    8
    9
    INFO: Early stopping counter 1 of 10
    10
    INFO: Early stopping counter 2 of 10
    11
    12
    13
    14
    15
    16
    INFO: Early stopping counter 1 of 10
    17
    INFO: Early stopping counter 2 of 10
    18
    19
    INFO: Early stopping counter 1 of 10
    20
    INFO: Early stopping counter 2 of 10
    21
    22
    INFO: Early stopping counter 1 of 10
    23
    INFO: Early stopping counter 2 of 10
    24
    25
    INFO: Early stopping counter 1 of 10
    26
    27
    INFO: Early stopping counter 1 of 10
    28
    29
    INFO: Early stopping counter 1 of 10
    30
    INFO: Early stopping counter 2 of 10
    31
    32
    INFO: Early stopping counter 1 of 10
    33
    INFO: Early stopping counter 2 of 10
    34
    INFO: Early stopping counter 3 of 10
    35
    36
    INFO: Early stopping counter 1 of 10
    37
    INFO: Early stopping counter 2 of 10
    38
    39
    INFO: Early stopping counter 1 of 10
    40
    INFO: Early stopping counter 2 of 10
    41
    INFO: Early stopping counter 3 of 10
    42
    INFO: Early stopping counter 4 of 10
    43
    INFO: Early stopping counter 5 of 10
    44
    Epoch 00045: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    45
    INFO: Early stopping counter 7 of 10
    46
    INFO: Early stopping counter 8 of 10
    47
    INFO: Early stopping counter 9 of 10
    48
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch48
    5
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    4
    INFO: Early stopping counter 1 of 10
    5
    6
    7
    INFO: Early stopping counter 1 of 10
    8
    INFO: Early stopping counter 2 of 10
    9
    INFO: Early stopping counter 3 of 10
    10
    INFO: Early stopping counter 4 of 10
    11
    INFO: Early stopping counter 5 of 10
    12
    Epoch 00013: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    13
    INFO: Early stopping counter 7 of 10
    14
    INFO: Early stopping counter 8 of 10
    15
    INFO: Early stopping counter 9 of 10
    16
    17
    INFO: Early stopping counter 1 of 10
    18
    19
    INFO: Early stopping counter 1 of 10
    20
    INFO: Early stopping counter 2 of 10
    21
    INFO: Early stopping counter 3 of 10
    22
    INFO: Early stopping counter 4 of 10
    23
    INFO: Early stopping counter 5 of 10
    24
    Epoch 00025: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 6 of 10
    25
    INFO: Early stopping counter 7 of 10
    26
    INFO: Early stopping counter 8 of 10
    27
    INFO: Early stopping counter 9 of 10
    28
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch28
    5000
    1
    0
    1
    2
    3
    INFO: Early stopping counter 1 of 10
    4
    5
    INFO: Early stopping counter 1 of 10
    6
    INFO: Early stopping counter 2 of 10
    7
    8
    9
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    13
    14
    INFO: Early stopping counter 1 of 10
    15
    16
    17
    18
    INFO: Early stopping counter 1 of 10
    19
    20
    INFO: Early stopping counter 1 of 10
    21
    INFO: Early stopping counter 2 of 10
    22
    INFO: Early stopping counter 3 of 10
    23
    INFO: Early stopping counter 4 of 10
    24
    INFO: Early stopping counter 5 of 10
    25
    Epoch 00026: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    26
    INFO: Early stopping counter 7 of 10
    27
    INFO: Early stopping counter 8 of 10
    28
    INFO: Early stopping counter 9 of 10
    29
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch29
    2
    0
    1
    2
    3
    4
    5
    INFO: Early stopping counter 1 of 10
    6
    7
    INFO: Early stopping counter 1 of 10
    8
    INFO: Early stopping counter 2 of 10
    9
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    INFO: Early stopping counter 2 of 10
    13
    INFO: Early stopping counter 3 of 10
    14
    INFO: Early stopping counter 4 of 10
    15
    INFO: Early stopping counter 5 of 10
    16
    17
    18
    INFO: Early stopping counter 1 of 10
    19
    INFO: Early stopping counter 2 of 10
    20
    INFO: Early stopping counter 3 of 10
    21
    INFO: Early stopping counter 4 of 10
    22
    INFO: Early stopping counter 5 of 10
    23
    Epoch 00024: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    24
    INFO: Early stopping counter 7 of 10
    25
    INFO: Early stopping counter 8 of 10
    26
    INFO: Early stopping counter 9 of 10
    27
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch27
    3
    0
    1
    2
    3
    INFO: Early stopping counter 1 of 10
    4
    INFO: Early stopping counter 2 of 10
    5
    INFO: Early stopping counter 3 of 10
    6
    INFO: Early stopping counter 4 of 10
    7
    INFO: Early stopping counter 5 of 10
    8
    INFO: Early stopping counter 6 of 10
    9
    INFO: Early stopping counter 7 of 10
    10
    Epoch 00011: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 8 of 10
    11
    INFO: Early stopping counter 9 of 10
    12
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch12
    4
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    13
    14
    15
    INFO: Early stopping counter 1 of 10
    16
    INFO: Early stopping counter 2 of 10
    17
    18
    INFO: Early stopping counter 1 of 10
    19
    INFO: Early stopping counter 2 of 10
    20
    INFO: Early stopping counter 3 of 10
    21
    INFO: Early stopping counter 4 of 10
    22
    INFO: Early stopping counter 5 of 10
    23
    Epoch 00024: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    24
    INFO: Early stopping counter 7 of 10
    25
    INFO: Early stopping counter 8 of 10
    26
    INFO: Early stopping counter 9 of 10
    27
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch27
    5
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    4
    5
    6
    7
    8
    INFO: Early stopping counter 1 of 10
    9
    INFO: Early stopping counter 2 of 10
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    INFO: Early stopping counter 2 of 10
    13
    INFO: Early stopping counter 3 of 10
    14
    INFO: Early stopping counter 4 of 10
    15
    INFO: Early stopping counter 5 of 10
    16
    Epoch 00017: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    17
    INFO: Early stopping counter 7 of 10
    18
    INFO: Early stopping counter 8 of 10
    19
    INFO: Early stopping counter 9 of 10
    20
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch20
    10000
    1
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    INFO: Early stopping counter 2 of 10
    4
    INFO: Early stopping counter 3 of 10
    5
    6
    INFO: Early stopping counter 1 of 10
    7
    INFO: Early stopping counter 2 of 10
    8
    INFO: Early stopping counter 3 of 10
    9
    INFO: Early stopping counter 4 of 10
    10
    INFO: Early stopping counter 5 of 10
    11
    12
    INFO: Early stopping counter 1 of 10
    13
    INFO: Early stopping counter 2 of 10
    14
    INFO: Early stopping counter 3 of 10
    15
    INFO: Early stopping counter 4 of 10
    16
    INFO: Early stopping counter 5 of 10
    17
    INFO: Early stopping counter 6 of 10
    18
    Epoch 00019: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 7 of 10
    19
    INFO: Early stopping counter 8 of 10
    20
    INFO: Early stopping counter 9 of 10
    21
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch21
    2
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    4
    INFO: Early stopping counter 1 of 10
    5
    INFO: Early stopping counter 2 of 10
    6
    7
    INFO: Early stopping counter 1 of 10
    8
    INFO: Early stopping counter 2 of 10
    9
    INFO: Early stopping counter 3 of 10
    10
    INFO: Early stopping counter 4 of 10
    11
    INFO: Early stopping counter 5 of 10
    12
    INFO: Early stopping counter 6 of 10
    13
    INFO: Early stopping counter 7 of 10
    14
    Epoch 00015: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 8 of 10
    15
    INFO: Early stopping counter 9 of 10
    16
    17
    INFO: Early stopping counter 1 of 10
    18
    INFO: Early stopping counter 2 of 10
    19
    INFO: Early stopping counter 3 of 10
    20
    INFO: Early stopping counter 4 of 10
    21
    INFO: Early stopping counter 5 of 10
    22
    INFO: Early stopping counter 6 of 10
    23
    INFO: Early stopping counter 7 of 10
    24
    Epoch 00025: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 8 of 10
    25
    INFO: Early stopping counter 9 of 10
    26
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch26
    3
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    INFO: Early stopping counter 2 of 10
    4
    5
    INFO: Early stopping counter 1 of 10
    6
    INFO: Early stopping counter 2 of 10
    7
    8
    9
    10
    INFO: Early stopping counter 1 of 10
    11
    INFO: Early stopping counter 2 of 10
    12
    INFO: Early stopping counter 3 of 10
    13
    14
    INFO: Early stopping counter 1 of 10
    15
    INFO: Early stopping counter 2 of 10
    16
    INFO: Early stopping counter 3 of 10
    17
    INFO: Early stopping counter 4 of 10
    18
    INFO: Early stopping counter 5 of 10
    19
    INFO: Early stopping counter 6 of 10
    20
    INFO: Early stopping counter 7 of 10
    21
    INFO: Early stopping counter 8 of 10
    22
    INFO: Early stopping counter 9 of 10
    23
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch23
    4
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    INFO: Early stopping counter 2 of 10
    4
    INFO: Early stopping counter 3 of 10
    5
    INFO: Early stopping counter 4 of 10
    6
    INFO: Early stopping counter 5 of 10
    7
    INFO: Early stopping counter 6 of 10
    8
    INFO: Early stopping counter 7 of 10
    9
    INFO: Early stopping counter 8 of 10
    10
    INFO: Early stopping counter 9 of 10
    11
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch11
    5
    0
    1
    2
    3
    INFO: Early stopping counter 1 of 10
    4
    INFO: Early stopping counter 2 of 10
    5
    6
    INFO: Early stopping counter 1 of 10
    7
    8
    9
    10
    INFO: Early stopping counter 1 of 10
    11
    INFO: Early stopping counter 2 of 10
    12
    13
    INFO: Early stopping counter 1 of 10
    14
    INFO: Early stopping counter 2 of 10
    15
    INFO: Early stopping counter 3 of 10
    16
    INFO: Early stopping counter 4 of 10
    17
    INFO: Early stopping counter 5 of 10
    18
    Epoch 00019: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    19
    INFO: Early stopping counter 7 of 10
    20
    INFO: Early stopping counter 8 of 10
    21
    22
    INFO: Early stopping counter 1 of 10
    23
    INFO: Early stopping counter 2 of 10
    24
    INFO: Early stopping counter 3 of 10
    25
    INFO: Early stopping counter 4 of 10
    26
    INFO: Early stopping counter 5 of 10
    27
    Epoch 00028: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 6 of 10
    28
    INFO: Early stopping counter 7 of 10
    29
    INFO: Early stopping counter 8 of 10
    30
    INFO: Early stopping counter 9 of 10
    31
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch31
    20000
    1
    0
    1
    2
    3
    4
    5
    6
    7
    8
    INFO: Early stopping counter 1 of 10
    9
    INFO: Early stopping counter 2 of 10
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    INFO: Early stopping counter 2 of 10
    13
    INFO: Early stopping counter 3 of 10
    14
    INFO: Early stopping counter 4 of 10
    15
    INFO: Early stopping counter 5 of 10
    16
    17
    18
    INFO: Early stopping counter 1 of 10
    19
    INFO: Early stopping counter 2 of 10
    20
    INFO: Early stopping counter 3 of 10
    21
    INFO: Early stopping counter 4 of 10
    22
    INFO: Early stopping counter 5 of 10
    23
    INFO: Early stopping counter 6 of 10
    24
    Epoch 00025: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 7 of 10
    25
    INFO: Early stopping counter 8 of 10
    26
    INFO: Early stopping counter 9 of 10
    27
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch27
    2
    0
    1
    2
    3
    4
    5
    6
    INFO: Early stopping counter 1 of 10
    7
    8
    9
    10
    INFO: Early stopping counter 1 of 10
    11
    INFO: Early stopping counter 2 of 10
    12
    INFO: Early stopping counter 3 of 10
    13
    14
    15
    INFO: Early stopping counter 1 of 10
    16
    INFO: Early stopping counter 2 of 10
    17
    INFO: Early stopping counter 3 of 10
    18
    INFO: Early stopping counter 4 of 10
    19
    INFO: Early stopping counter 5 of 10
    20
    Epoch 00021: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    21
    22
    INFO: Early stopping counter 1 of 10
    23
    INFO: Early stopping counter 2 of 10
    24
    INFO: Early stopping counter 3 of 10
    25
    INFO: Early stopping counter 4 of 10
    26
    INFO: Early stopping counter 5 of 10
    27
    INFO: Early stopping counter 6 of 10
    28
    INFO: Early stopping counter 7 of 10
    29
    Epoch 00030: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 8 of 10
    30
    31
    INFO: Early stopping counter 1 of 10
    32
    INFO: Early stopping counter 2 of 10
    33
    INFO: Early stopping counter 3 of 10
    34
    INFO: Early stopping counter 4 of 10
    35
    INFO: Early stopping counter 5 of 10
    36
    INFO: Early stopping counter 6 of 10
    37
    INFO: Early stopping counter 7 of 10
    38
    INFO: Early stopping counter 8 of 10
    39
    INFO: Early stopping counter 9 of 10
    40
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch40
    3
    0
    1
    2
    3
    4
    5
    INFO: Early stopping counter 1 of 10
    6
    INFO: Early stopping counter 2 of 10
    7
    INFO: Early stopping counter 3 of 10
    8
    INFO: Early stopping counter 4 of 10
    9
    INFO: Early stopping counter 5 of 10
    10
    Epoch 00011: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    11
    INFO: Early stopping counter 7 of 10
    12
    INFO: Early stopping counter 8 of 10
    13
    INFO: Early stopping counter 9 of 10
    14
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch14
    4
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    4
    5
    INFO: Early stopping counter 1 of 10
    6
    INFO: Early stopping counter 2 of 10
    7
    INFO: Early stopping counter 3 of 10
    8
    9
    INFO: Early stopping counter 1 of 10
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    INFO: Early stopping counter 2 of 10
    13
    14
    INFO: Early stopping counter 1 of 10
    15
    INFO: Early stopping counter 2 of 10
    16
    INFO: Early stopping counter 3 of 10
    17
    INFO: Early stopping counter 4 of 10
    18
    INFO: Early stopping counter 5 of 10
    19
    INFO: Early stopping counter 6 of 10
    20
    Epoch 00021: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 7 of 10
    21
    22
    INFO: Early stopping counter 1 of 10
    23
    INFO: Early stopping counter 2 of 10
    24
    INFO: Early stopping counter 3 of 10
    25
    INFO: Early stopping counter 4 of 10
    26
    INFO: Early stopping counter 5 of 10
    27
    Epoch 00028: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 6 of 10
    28
    INFO: Early stopping counter 7 of 10
    29
    INFO: Early stopping counter 8 of 10
    30
    INFO: Early stopping counter 9 of 10
    31
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch31
    5
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    4
    5
    INFO: Early stopping counter 1 of 10
    6
    7
    INFO: Early stopping counter 1 of 10
    8
    9
    10
    INFO: Early stopping counter 1 of 10
    11
    INFO: Early stopping counter 2 of 10
    12
    13
    INFO: Early stopping counter 1 of 10
    14
    INFO: Early stopping counter 2 of 10
    15
    INFO: Early stopping counter 3 of 10
    16
    INFO: Early stopping counter 4 of 10
    17
    INFO: Early stopping counter 5 of 10
    18
    INFO: Early stopping counter 6 of 10
    19
    20
    INFO: Early stopping counter 1 of 10
    21
    INFO: Early stopping counter 2 of 10
    22
    INFO: Early stopping counter 3 of 10
    23
    INFO: Early stopping counter 4 of 10
    24
    INFO: Early stopping counter 5 of 10
    25
    INFO: Early stopping counter 6 of 10
    26
    INFO: Early stopping counter 7 of 10
    27
    INFO: Early stopping counter 8 of 10
    28
    29
    INFO: Early stopping counter 1 of 10
    30
    INFO: Early stopping counter 2 of 10
    31
    INFO: Early stopping counter 3 of 10
    32
    INFO: Early stopping counter 4 of 10
    33
    INFO: Early stopping counter 5 of 10
    34
    Epoch 00035: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    35
    INFO: Early stopping counter 7 of 10
    36
    INFO: Early stopping counter 8 of 10
    37
    INFO: Early stopping counter 9 of 10
    38
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch38
    50000
    1
    0
    1
    2
    3
    INFO: Early stopping counter 1 of 10
    4
    INFO: Early stopping counter 2 of 10
    5
    INFO: Early stopping counter 3 of 10
    6
    INFO: Early stopping counter 4 of 10
    7
    INFO: Early stopping counter 5 of 10
    8
    Epoch 00009: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    9
    10
    11
    INFO: Early stopping counter 1 of 10
    12
    INFO: Early stopping counter 2 of 10
    13
    INFO: Early stopping counter 3 of 10
    14
    INFO: Early stopping counter 4 of 10
    15
    INFO: Early stopping counter 5 of 10
    16
    Epoch 00017: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 6 of 10
    17
    INFO: Early stopping counter 7 of 10
    18
    INFO: Early stopping counter 8 of 10
    19
    INFO: Early stopping counter 9 of 10
    20
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch20
    2
    0
    1
    2
    3
    INFO: Early stopping counter 1 of 10
    4
    INFO: Early stopping counter 2 of 10
    5
    INFO: Early stopping counter 3 of 10
    6
    INFO: Early stopping counter 4 of 10
    7
    INFO: Early stopping counter 5 of 10
    8
    Epoch 00009: reducing learning rate of group 0 to 5.0000e-03.
    INFO: Early stopping counter 6 of 10
    9
    INFO: Early stopping counter 7 of 10
    10
    INFO: Early stopping counter 8 of 10
    11
    12
    13
    INFO: Early stopping counter 1 of 10
    14
    INFO: Early stopping counter 2 of 10
    15
    INFO: Early stopping counter 3 of 10
    16
    INFO: Early stopping counter 4 of 10
    17
    INFO: Early stopping counter 5 of 10
    18
    INFO: Early stopping counter 6 of 10
    19
    INFO: Early stopping counter 7 of 10
    20
    INFO: Early stopping counter 8 of 10
    21
    INFO: Early stopping counter 9 of 10
    22
    Epoch 00023: reducing learning rate of group 0 to 2.5000e-03.
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch22
    3
    0
    1
    2
    3
    4
    5
    INFO: Early stopping counter 1 of 10
    6
    INFO: Early stopping counter 2 of 10
    7
    8
    INFO: Early stopping counter 1 of 10
    9
    INFO: Early stopping counter 2 of 10
    10
    INFO: Early stopping counter 3 of 10
    11
    INFO: Early stopping counter 4 of 10
    12
    INFO: Early stopping counter 5 of 10
    13
    14
    INFO: Early stopping counter 1 of 10
    15
    INFO: Early stopping counter 2 of 10
    16
    INFO: Early stopping counter 3 of 10
    17
    INFO: Early stopping counter 4 of 10
    18
    INFO: Early stopping counter 5 of 10
    19
    INFO: Early stopping counter 6 of 10
    20
    INFO: Early stopping counter 7 of 10
    21
    INFO: Early stopping counter 8 of 10
    22
    INFO: Early stopping counter 9 of 10
    23
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch23
    4
    0
    1
    2
    INFO: Early stopping counter 1 of 10
    3
    INFO: Early stopping counter 2 of 10
    4
    INFO: Early stopping counter 3 of 10
    5
    INFO: Early stopping counter 4 of 10
    6
    INFO: Early stopping counter 5 of 10
    7
    INFO: Early stopping counter 6 of 10
    8
    INFO: Early stopping counter 7 of 10
    9
    INFO: Early stopping counter 8 of 10
    10
    INFO: Early stopping counter 9 of 10
    11
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch11
    5
    0
    1
    2
    3
    INFO: Early stopping counter 1 of 10
    4
    5
    INFO: Early stopping counter 1 of 10
    6
    7
    INFO: Early stopping counter 1 of 10
    8
    9
    INFO: Early stopping counter 1 of 10
    10
    INFO: Early stopping counter 2 of 10
    11
    INFO: Early stopping counter 3 of 10
    12
    INFO: Early stopping counter 4 of 10
    13
    INFO: Early stopping counter 5 of 10
    14
    INFO: Early stopping counter 6 of 10
    15
    INFO: Early stopping counter 7 of 10
    16
    INFO: Early stopping counter 8 of 10
    17
    INFO: Early stopping counter 9 of 10
    18
    INFO: Early stopping counter 10 of 10
    INFO: Early stopping
    End at epoch18
    100000
    1



```python

```
