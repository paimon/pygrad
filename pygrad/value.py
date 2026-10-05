from pygrad.utils import top_sort


def wrap_value(val):
    return val if isinstance(val, Value) else Value(float(val))


class Value:
    operation = 'I'

    def __init__(self, data, *prev):
        self.data = data
        self.prev = prev
        self.grad = 0

    def _backward(self):
        pass

    def backward(self):
        self.grad = 1.0
        values = top_sort(self)
        for v in reversed(values):
            v._backward()

    def __repr__(self):
        return f'Value({self.data})'

    def __neg__(self):
        return -1.0 * self

    def __add__(self, other):
        return Add(self, wrap_value(other))

    __radd__ = __add__

    def __mul__(self, other):
        return Mul(self, wrap_value(other))

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + (-wrap_value(other))

    def __rsub__(self, other):
        return (-self) + wrap_value(other)

    def __pow__(self, deg):
        return Pow(self, float(deg))

    def __truediv__(self, other):
        return Mul(self, wrap_value(other) ** (-1.0))

    def __rtruediv__(self, other):
        return wrap_value(other) / self

    def relu(self):
        return ReLU(self)


class Add(Value):
    operation = '+'

    def __init__(self, first, second):
        super().__init__(first.data + second.data, first, second)

    def _backward(self):
        self.prev[0].grad += self.grad
        self.prev[1].grad += self.grad


class Mul(Value):
    operation = '*'

    def __init__(self, first, second):
        super().__init__(first.data * second.data, first, second)

    def _backward(self):
        self.prev[0].grad += self.grad * self.prev[1].data
        self.prev[1].grad += self.grad * self.prev[0].data


class Pow(Value):
    operation = '^'

    def __init__(self, val, deg):
        super().__init__(val.data ** deg, val)
        self.deg = deg

    def _backward(self):
        prev = self.prev[0]
        grad = self.deg * (prev.data ** (self.deg - 1))
        prev.grad += self.grad * grad


class ReLU(Value):
    operation = 'ReLU'

    def __init__(self, val):
        super().__init__(max(0, val.data), val)

    def _backward(self):
        self.prev[0].grad += self.grad * float(self.data > 0)
