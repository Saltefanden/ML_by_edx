from data_def import x_def, y_def

dot = lambda x, y: sum(map(lambda x, y: x*y, x, y)) 
norm = lambda x, p: sum(map(lambda x: x**p, x))**(1/p)
eucl = lambda x: norm(x, 2)



def base_perceptron(X, Y, max_iter=40):
    wrongly_classfied = lambda theta, x, y: y*dot(theta, x) <= 0
    theta = [0] * len(X[0])
    for _ in range(max_iter):
        for x, y in zip(X,Y):
            if wrongly_classfied(theta, x, y):
                theta = list(map(lambda theta, x: theta + y*x, theta, x))
    return theta


def perceptron(X, Y, max_iter=40):
    wrongly_classfied = lambda theta, theta0, x, y: y*(dot(theta, x) + theta0) <= 0
    theta = [0] * len(X[0])
    theta0 = 0
    for i in range(max_iter):
        converged = True
        for x, y in zip(X, Y):
            if wrongly_classfied(theta, theta0, x, y):
                theta = list(map(lambda theta, x: theta + y*x, theta, x))
                theta0 += y * eucl(theta)**2 /abs(theta0+1)
                converged = False
        if converged:
            break
    print(f"{converged=}")
    return theta, theta0

# print(base_perceptron([(1,2), (3,2), (3,3)], [1, -1, 1]))
# print(perceptron([(1,2), (3,2), (3,3)], [1, -1, 1]))

# print(perceptron(x_def, y_def, max_iter=4))