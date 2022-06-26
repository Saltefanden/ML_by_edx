class Datapoint():
    def __init__(self, data: list, label: int):
        self.data = data
        self.label = label


def dot(a: list, b: list) -> float:
    if len(a) != len(b):
        raise TypeError(f"Dimension mismatch {len(a)=}, {len(b)=}")
    
    return sum(x*y for x,y in zip(a,b))

def test_perceptron(theta, datapoints):
    return all([((dot(theta, datapoint.data)>0) * 2-1) == datapoint.label for datapoint in datapoints])


def loud_perceptron(datapoints):
    theta = [0,0]
    th_list = []
    mistakes = 0 
    while (not test_perceptron(theta, datapoints)):
        for datapoint in datapoints:
            if dot(datapoint.data, theta)*datapoint.label<=0:
                mistakes+=1
                theta = [th + datapoint.label * datapoint.data[idx] for idx, th in enumerate(theta)]
                th_list.append(theta)
    print(f"{mistakes = },\n{th_list = }")
    return theta

def loud_perceptron_with_offset(datapoints, max_mistakes = 100):
    theta = [0,0]
    theta0 = 0
    th_list = []
    mistakes = 0 
    while (not test_perceptron(theta, datapoints)):
        for datapoint in datapoints:
            if dot(datapoint.data, theta)*datapoint.label<=0:
                mistakes+=1
                if mistakes >=max_mistakes:
                    print(f"Ran through too many mistakes: {mistakes=}")
                    return None
                theta = [th + datapoint.label * datapoint.data[idx] for idx, th in enumerate(theta)]
                th_list.append(theta)
                theta0 += datapoint.label
    print(f"{mistakes = },\n{th_list = }")
    return theta


def Ex1a():
    print('x1, x2, x3:')
    dps = [
        Datapoint([-1, -1], 1),
        Datapoint([1, 0], -1),
        Datapoint([-1, 1.5], 1),
    ]
    loud_perceptron(dps)

    print('\nx2, x3, x1:')
    dps = [
        Datapoint([1, 0], -1),
        Datapoint([-1, 1.5], 1),
        Datapoint([-1, -1], 1),
    ]
    loud_perceptron(dps)


def Ex1c():
    print('x1, x2, x3=[-1, 10]:')
    dps = [
        Datapoint([-1, -1], 1),
        Datapoint([1, 0], -1),
        Datapoint([-1, 10], 1),
    ]
    loud_perceptron(dps)
    
    
    print('x2, x3=[-1, 10], x1:')
    dps = [
        Datapoint([1, 0], -1),
        Datapoint([-1, 10], 1),
        Datapoint([-1, -1], 1),
    ]
    loud_perceptron(dps)


def Ex2a():
    theta = [0,0]
    theta0 = 0
    th_list = []

    d1 = Datapoint([-4,2],1)
    d3 = Datapoint([-1, -1],-1)
    d4 = Datapoint([2,2],-1)

    datapoints = [d1, d3, d4, d3]

    for datapoint in datapoints:
        theta = [th + datapoint.label * datapoint.data[idx] for idx, th in enumerate(theta)]
        th_list.append(theta)
        theta0 += datapoint.label

    print(f"{th_list = }\n{theta0=}")


def Ex2b():
    # This is really just down to finding a correct theta
    dps = [
        Datapoint([-4,2],1),
        Datapoint([-2,1],1),
        Datapoint([1,-2],-1),
        Datapoint([-1, -1],-1),
        Datapoint([2,2],-1),
    ]
    out = loud_perceptron_with_offset(dps, max_mistakes=4000)
    if out == None:
        from sklearn.linear_model import Perceptron
        import numpy as np

        X = np.ndarray(shape=(len(dps),2))
        y = np.ndarray(shape=(len(dps),))
        for idx, dp in enumerate(dps):
            X[idx,:] = dp.data
            y[idx] = dp.label

        classifier = Perceptron(max_iter=40, random_state=0)
        classifier.fit(X,y)
        print(f"Theta={classifier.coef_}.\t Theta0={classifier.intercept_}")



def main():
    # Ex1a()
    # Ex1c()
    # Ex2a()
    Ex2b()

    pass

if __name__ == '__main__':
    main()

