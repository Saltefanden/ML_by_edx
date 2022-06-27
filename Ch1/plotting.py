from data_def import xpos, xneg, x_def, y_def
import matplotlib.pyplot as plt
from Perceptron import perceptron
import random 

decision_boundary_limits = [min(map(lambda x: x[0], x_def)), max(map(lambda x: x[0], x_def))]
decision_boundary_x2 = lambda x, theta, theta0: -theta[0]/theta[1] * x - theta0/theta[1]

xy = list(zip(x_def,y_def))
random.shuffle(xy)
x_def, y_def = list(zip(*xy))

theta, theta0 = perceptron(x_def, y_def,max_iter=10000)

plt.plot(list(map(lambda x: x[0], xpos)),list(map(lambda x: x[1], xpos)), linestyle='None', marker='x')
plt.plot(list(map(lambda x: x[0], xneg)),list(map(lambda x: x[1], xneg)), linestyle='None', marker='_')
# move decision boundary by correct amount depending on the value of theta0
plt.plot([decision_boundary_limits[0], decision_boundary_limits[1]], [decision_boundary_x2(decision_boundary_limits[0], theta, theta0), decision_boundary_x2(decision_boundary_limits[1], theta, theta0)], label='Decision boundary')

# plt.legend(location='northwest')
plt.axis('square')
plt.xlim(decision_boundary_limits)
plt.ylim([min(map(lambda x: x[1], x_def)), max(map(lambda x: x[1], x_def))])

print(theta, theta0)
plt.savefig('./lol1.png')