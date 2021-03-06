import random

dot = lambda x, y: sum(map(lambda x, y: x*y, x, y)) 
norm = lambda x, p: sum(map(lambda x: x**p, x))**(1/p)
eucl = lambda x: norm(x, 2)

def stochastic_gradient_descend(X: tuple[int, int], Y: int, max_iter=40, R_lambda=1):
    hinge_loss = lambda z: max(0, 1 - z)
    theta = [0] * len(X[0])
    theta0 = 0
    for t in range(max_iter):
        learning_rate = 1/(1+t)
        i = random.randint(0,len(X)-1)
        xi, yi = X[i], Y[i]
        norm_theta = eucl(theta)
        if hinge_loss(yi*(dot(theta, xi) + theta0)) > 0:
            theta = list(map(lambda theta, xi: 
                theta  + learning_rate * (yi*xi - R_lambda * theta), 
                theta, xi))
            theta0 += learning_rate * yi
        else:
            theta = list(map(lambda theta, xi: 
                theta  + learning_rate * (0 - R_lambda * theta), 
                theta, xi))
    return theta, theta0


# print(base_perceptron([(1,2), (3,2), (3,3)], [1, -1, 1]))
# print(perceptron([(1,2), (3,2), (3,3)], [1, -1, 1]))

# print(perceptron(x_def, y_def, max_iter=4))