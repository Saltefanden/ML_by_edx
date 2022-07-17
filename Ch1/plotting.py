from data_def import xpos, xneg, x_def, y_def
import matplotlib.pyplot as plt
from Perceptron import perceptron
from StochasticGradientDescend import stochastic_gradient_descend
import random 
import Project1.sentiment_analysis.project1 as p1
import numpy as np

decision_boundary_limits = [min(map(lambda x: x[0], x_def)), max(map(lambda x: x[0], x_def))]
decision_boundary_x2 = lambda x, theta, theta0: -theta[0]/theta[1] * x - theta0/theta[1]
positive_margin_boundary_x2 = lambda x, theta, theta0: -theta[0]/theta[1] * x + ( 1 - theta0)/theta[1]
negative_margin_boundary_x2 = lambda x, theta, theta0: -theta[0]/theta[1] * x + (-1 - theta0)/theta[1]

xy = list(zip(x_def,y_def))
random.shuffle(xy)
x_def, y_def = list(zip(*xy))

theta, theta0 = perceptron(x_def, y_def,max_iter=10000)
# theta, theta0 = stochastic_gradient_descend(x_def, y_def,max_iter=100_000, R_lambda=1)
# theta, theta0 = p1.average_perceptron(np.array(x_def), np.array(y_def),10_000)

plt.plot(list(map(lambda x: x[0], xpos)),list(map(lambda x: x[1], xpos)), linestyle='None', marker='x')
plt.plot(list(map(lambda x: x[0], xneg)),list(map(lambda x: x[1], xneg)), linestyle='None', marker='_')
# move decision boundary by correct amount depending on the value of theta0
plt.plot([decision_boundary_limits[0], decision_boundary_limits[1]], [decision_boundary_x2(decision_boundary_limits[0], theta, theta0), decision_boundary_x2(decision_boundary_limits[1], theta, theta0)], label='Decision boundary')
plt.plot([decision_boundary_limits[0], decision_boundary_limits[1]], [positive_margin_boundary_x2(decision_boundary_limits[0], theta, theta0), positive_margin_boundary_x2(decision_boundary_limits[1], theta, theta0)], label='Positive Margin Boundary', linestyle='--', color = 'r')
plt.plot([decision_boundary_limits[0], decision_boundary_limits[1]], [negative_margin_boundary_x2(decision_boundary_limits[0], theta, theta0), negative_margin_boundary_x2(decision_boundary_limits[1], theta, theta0)], label='Negative Margin Boundary', linestyle='--', color = 'b')

# plt.legend(location='northwest')
plt.axis('square')
plt.xlim(decision_boundary_limits)
plt.ylim([min(map(lambda x: x[1], x_def)), max(map(lambda x: x[1], x_def))])

print(theta, theta0)
print(f"y = {-theta[0]/theta[1]:.2f} x {- theta0/theta[1]:+.2f}")
plt.savefig('./lol1.png')