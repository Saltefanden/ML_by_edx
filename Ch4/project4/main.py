import numpy as np
import matplotlib.pyplot as plt
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

Ks = list(range(1,5))
seeds = list(range(5))

for K in Ks:
    costs = []
    for seed in seeds:
        mixture, post = common.init(X, K, seed)
        _, _, cost = kmeans.run(X, mixture, post)
        costs.append(cost)
    
    min_seed = costs.index(min(costs))
    print(K, min(costs), min_seed)
    mixture, post = common.init(X, K, min_seed)
    common.plot(X, mixture, post, f'{K =}, {min_seed =}')
    plt.savefig(f'figs/plot_{K}_{min_seed}.png')
    plt.close()
