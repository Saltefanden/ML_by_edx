x = [
    [-1, -1],
    [1, -1],
    [-1, 1],
    [1, 1],
]

y = [
    1,
    -1,
    -1,
    1
]


F = lambda z: 2*z - 3

f = [
    lambda xi, w, F=F: F( w[0][0] + (w[1][0] * xi[0] + w[2][0] * xi[1])),
    lambda xi, w, F=F: F( w[0][1] + (w[1][1] * xi[0] + w[2][1] * xi[1])),
]

# w = [
#     [w00, w01], # w0
#     [w10, w11], # w1
#     [w20, w21], # w2
# ]

w = [
    [0, 0],
    [0, 0],
    [0, 0],
]

w = [
    [1, 1],
    [2, -2],
    [2, -2],
]

w = [
    [1, 1],
    [-2, 2],
    [-2, 2],
]

all_new_points = [[func(point, w) for func in f] for point in x]

print(f'{all_new_points = }\n{y = }')




w = [
    [1, 1],
    [1, -1],
    [-1, 1]
]

import math

Fs = [
    lambda z: 5*z-2,
    lambda z: max(0, z),
    lambda z: math.tanh(z),
    lambda z: z
]

all_new_points = [[[func(point, w, activation) for func in f] for point in x] for activation in Fs]
import matplotlib.pyplot as plt
count = 0
for points in all_new_points:
    count += 1
    plt.scatter(*zip(*points), c = y)
    plt.savefig('figs/fig' + str(count) + '.png')
    plt.close()
