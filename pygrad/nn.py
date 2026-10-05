import random

from pygrad.value import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, in_dim):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(in_dim)]
        self.b = Value(0)

    def parameters(self):
        yield from self.w
        yield self.b

    def __call__(self, X):
        return sum((wi * xi for wi, xi in zip(self.w, X)), self.b)


class LinearLayer(Module):
    def __init__(self, in_dim, out_dim):
        self.neurons = [Neuron(in_dim) for _ in range(out_dim)]

    def parameters(self):
        for n in self.neurons:
            yield from n.parameters()

    def __call__(self, X):
        return [n(X) for n in self.neurons]


class ReLULayer(Module):
    def __call__(self, X):
        return [x.relu() for x in X]


class MLP(Module):
    def __init__(self, *dims):
        self.layers = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            if self.layers:
                self.layers.append(ReLULayer())
            self.layers.append(LinearLayer(in_dim, out_dim))

    def parameters(self):
        for l in self.layers:
            yield from l.parameters()

    def __call__(self, X):
        for l in self.layers:
            X = l(X)
        return X
