class Point():
    def __init__(self, x1: float, x2: float):
        self.x1 = x1 
        self.x2 = x2
        self.vec = [x1,x2]

class Datapoint():
    def __init__(self, data: Point, label: int):
        self.data = data
        self.label = label


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise TypeError(f"Dimension mismatch {len(a)=}, {len(b)=}")
    
    return sum(x*y for x,y in zip(a,b))


def perceptron(datapoints: list[Datapoint]) -> list[float]:
    theta = [0,0]
    while (not test_perceptron(theta, datapoints)):
        for datapoint in datapoints:
            if dot(datapoint.data.vec, theta)*datapoint.label<=0:
                theta = [theta[idx] + datapoint.label * datapoint.data.vec[idx] for idx, _ in enumerate(theta)]
    return theta

def test_perceptron(theta, datapoints):
    return all([((dot(theta, datapoint.data.vec)>0) * 2-1) == datapoint.label for datapoint in datapoints])


def main():
    datapoints = [
        Datapoint(Point(1.2,1.7), 1),
        Datapoint(Point(-1, 1), 1),
        Datapoint(Point(-0.43, 0.2), 1),
        Datapoint(Point(0.2, -1.5), -1),
        Datapoint(Point(-5, -0.5), 1),
        Datapoint(Point(-1, -10), -1),
        Datapoint(Point(-1, 0), 1),
        Datapoint(Point(-18, -0.5), 1),
    ] 

    th = perceptron(datapoints)
    print(th)

    for datapoint in datapoints:
        print("Dotproduct = ", f"{dot(th, datapoint.data.vec):.2f}", ",\t class: ", (dot(th, datapoint.data.vec)>0) * 2 -1, ",\t label =", datapoint.label)

    print(test_perceptron(th, datapoints))


if __name__ == '__main__':
    main()



