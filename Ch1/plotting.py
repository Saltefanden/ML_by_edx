from data_def import xpos, xneg
import matplotlib.pyplot as plt



plt.plot(list(map(lambda x: x[0], xpos)),list(map(lambda x: x[1], xpos)), linestyle='None', marker='x')
plt.plot(list(map(lambda x: x[0], xneg)),list(map(lambda x: x[1], xneg)), linestyle='None', marker='_')
# move decision boundary by correct amount depending on the value of theta0
a = plt.arrow(0, 0, 791, 671, label = 'Decision Boundary', edgecolor='black', facecolor='black')
a = plt.arrow(0, 0, 254*2, 298*2, label = 'Decision maxiter=1', edgecolor='red', facecolor='red')
a = plt.arrow(0, 0, 238, 849, label = 'Decision maxiter=2', edgecolor='green', facecolor='green')
a = plt.arrow(0, 0, 791, 671, label = 'Decision maxiter=3', edgecolor='blue', facecolor='blue')
# b = plt.arrow(0, 0, -671, 791, label = 'theta', edgecolor='black', facecolor='black')
plt.legend()
plt.axis('square')

plt.savefig('./lol.png')