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
            n.zero_grad


class nanoModel:
    def __init__(self, nin, layers):
        sizes=[nin]+layers
        self.layers=[nanoLayer(sizes[k], sizes[k+1]) for k in range(len(sizes)-1)]
        self.layers.append(nanoLayer(sizes[len(sizes)-2], sizes[len(sizes)-2], fun="sigmoid"))
    def __repr__(self):
        return f"nanoModel {self.layers}"
    def __call__(self, x):
        for layer in self.layers:
            x=layer(x)
        return x
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad
