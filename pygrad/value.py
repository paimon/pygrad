def wrap_value(val):
    return val if isinstance(val, Value) else Value(val)


class Value:
    def __init__(self, data, *prev):
        self.data = data
        self.prev = prev
        self.grad = 0

    def _backward(self):
        pass

    def backward(self):
        self.grad = 1.0
        self._backward()

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

    def __pow__(self, deg):
        assert isinstance(deg, float)
        return Pow(self, deg)

    def __truediv__(self, other):
        return Mul(self, wrap_value(other) ** (-1.0))

    def __sub__(self, other):
        return self + (-wrap_value(other))

    def __rsub__(self, other):
        return (-self) + wrap_value(other)

    def relu(self):
        return ReLU(self)


class Add(Value):
    def __init__(self, first, second):
        super().__init__(first.data + second.data, first, second)

    def _backward(self):
        self.prev[0].grad += self.grad
        self.prev[1].grad += self.grad


class Mul(Value):
    def __init__(self, first, second):
        super().__init__(first.data * second.data, first, second)

    def _backward(self):
        self.prev[0].grad += self.grad * self.prev[1].data
        self.prev[1].grad += self.grad * self.prev[0].data


class Pow(Value):
    def __init__(self, val, deg):
        super().__init__(val.data * deg, val)
        self.deg = deg

    def _backward(self):
        grad = self.deg * (self.data ** (self.deg - 1))
        self.prev[0].grad += self.grad * grad


class ReLU(Value):
    def __init__(self, val):
        super().__init__(max(0, val.data), val)

    def _backward(self):
        self.prev[0].grad += float(self.data > 0)

