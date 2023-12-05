import math
import random


class nano:
    def __init__(self, val, g=None, p=None):
        self.value = float(val)
        self.grad = 0
        self.parents = p
        self._backward = None

    def __add__(self, other) -> "nano":
        if type(other) != nano:
            other = nano(other)
        out = nano(self.value + other.value, p=(self, other))

        def _back():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _back
        return out

    def __neg__(self):
        return self * (-1)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other) -> "nano":
        if type(other) != nano:
            other = nano(other)
        out = nano(self.value * other.value, p=(self, other))

        def _back():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value

        out._backward = _back
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other) -> "nano":
        out = nano(self.value**other, p=(self,))

        def _back():
            self.grad += other * self.value ** (other - 1) * out.grad

        out._backward = _back
        return out

    def relu(self) -> "nano":
        out = nano(self.value * (self.value > 0), p=(self,))

        def _back():
            self.grad += out.grad * (self.value > 0)

        out._backward = _back
        return out

    def sigmoid(self):
        v = math.exp(self.value) / (1 + math.exp(self.value))
        out = nano(v, p=(self,))

        def _back():
            self.grad += v * (1 - v) * out.grad

        out._backward = _back
        return out

    def __str__(self) -> str:
        return f"nano: {self.value}, grad: {self.grad}"

    def __repr__(self) -> str:
        return f"nano: {self.value}, grad: {self.grad}"

    def item(self) -> float:
        return self.value

    def __int__(self) -> int:
        return int(self.value)

    def backward(self):
        visited = []

        def recursion(node):
            if node in visited or node is None or node.parents is None:
                return
            visited.append(node)
            node._backward()
            for parent in node.parents:
                recursion(parent)

        self.grad = 1.0
        recursion(self)


class nanoNeuron:
    def __init__(self, n, fun="relu"):
        self.weights = [nano(random.gauss(mu=0, sigma=0.5)) for _ in range(n)]
        self.bias = nano(random.gauss(mu=0, sigma=1))
        self.act = fun

    def __repr__(self):
        return f"nanoNeuron(weights={self.weights}, )"

    def __call__(self, n):
        if len(self.weights) != len(n):
            raise IndexError("Input size not matching")
        out = sum((w * x for w, x in zip(self.weights, n)), self.bias)
        return (
            out.relu()
            if self.act == "relu"
            else out.sigmoid()
            if self.act == "sigmoid"
            else out
        )

    def zero_grad(self):
        for w in self.weights + [self.bias]:
            w.grad = 0


class nanoLayer:
    def __init__(self, nin, nout, fun="relu"):
        self.neurons = [nanoNeuron(nin, fun=fun) for _ in range(nout)]
        self.act = fun

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def __repr__(self):
        return f"nanoLayer {len(self.neurons[0].weights)} -> {len(self.neurons)} with function {self.act}"

    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()

    def params(self):
        return self.neurons


class nanoModel:
    def __init__(self, nin, layers):
        sizes = [nin] + layers
        self.layers = [nanoLayer(sizes[k], sizes[k + 1]) for k in range(len(sizes) - 2)]
        self.layers.append(
            nanoLayer(sizes[len(sizes) - 2], sizes[len(sizes) - 1], fun="sigmoid")
        )

    def __repr__(self):
        out = "nanoModel: \n"
        for layer in self.layers:
            out += str(layer) + "\n"
        return out

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def train(self, x, y, lr, lossfunc="mse"):
        pred = self(x)
        L = MSE(pred, y) if lossfunc == "mse" else nano(0)
        self.zero_grad()
        L.backward()


def MSE(p: list[nano], y: list[nano]) -> "nano":
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y))


class nanoArray:
    def __init__(self, size, default):
        default = float(default)
        if isinstance(size, int):
            size = (1, size)
            self.dims = 1
        elif isinstance(size, tuple):
            self.dims = len(size) - 1 * (size[0] == 1)
        if len(size) not in [1, 2]:
            raise IndexError("Illegal size in nanoArray __init__")
        self.data = [default for _ in range(size[0] * size[1])]
        self.shape = size

    def __getitem__(self, index):
        if isinstance(index, int) and self.dims == 1:
            return self.data[index]
        elif isinstance(index, int) and self.dims == 2:
            return self.data[self.shape[1] * (index) : self.shape[1] * (index + 1)]
        elif isinstance(index, tuple) and len(index) == 2 and self.dims == 2:
            return self.data[self.shape[1] * index[0] + index[1]]
        raise IndexError("Illegal index in __getitem__")

    def __setitem__(self, index, val):
        if self.dims == 1:
            if isinstance(index, int):
                self.data[index] = float(val)
                return
            raise IndexError("Illegal index in __setitem__ for onedimensional array")
        if self.dims == 2:
            if isinstance(index, tuple) and len(index) == 2:
                self.data[self.shape[1] * index[0] + index[1]] = float(val)
                return
            elif isinstance(index, int) and len(val) == self.shape[1]:
                self.data[self.shape[1] * index : self.shape[1] * (index + 1)] = [
                    float(x) for x in val
                ]
                return
            raise ValueError("Illegal value in __setitem__ for twodimensional array")
        raise IndexError("Illegal dimensions in __setitem__")

    def __repr__(self):
        s = f"nanoArray: {self.shape}\n"
        if self.dims == 1:
            return s + str(self.data)
        s += "["
        for k in range(self.shape[0]):
            s += str(self.data[self.shape[1] * k : self.shape[1] * (k + 1)]) + "\n"
        return s + "]"

    def __len__(self):
        return self.shape[0] if self.dims == 2 else self.shape[1]

    def __add__(self, other):  
        if isinstance(other, (int, float)):
            other = nanoArray(self.shape, other)
        if self.shape!=other.shape:
            raise Exception("Sizes not matching in __add__")
        out = nanoArray(self.shape, 0)
        for k in range(self.shape[0]):
            out.data[k * self.shape[1] : (k + 1) * (self.shape[1])] = [
                self.data[k * self.shape[1] + l] + other.data[k * self.shape[1] + l]
                for l in range(self.shape[1])
            ]
        return out

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = nanoArray(self.shape, other)
        if self.shape != other.shape:
            raise Exception("Sizes not matching in __mul__")
        out = nanoArray(self.shape, 0.0)
        for k in range(self.shape[0]):
            out.data[k * (self.shape[1]) : k * (self.shape[1] + 1)] = [
                self.data[k * self.shape[1] + l] * other.data[k * self.shape[1] + l]
                for l in range(self.shape[1])
            ]
        return out
        


def ones(size) -> "nanoArray":
    return nanoArray(size, 1.0)


def zeros(size) -> "nanoArray":
    return nanoArray(size, 0.0)


def unit(size: int) -> "nanoArray":
    out = nanoArray((size, size), 0.0)
    for k in range(size):
        out[k, k] = 1.0
    return out
