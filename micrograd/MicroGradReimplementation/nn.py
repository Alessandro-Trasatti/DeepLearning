import random
from engine import Value

class Neuron:
    """nin (__int__): number of inputs of the neuron
    """
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # Creates random weights
        self.b = Value(random.uniform(-1, 1)) # Creates a random bias
        
    def __call__(self, x): # Given a neuron n, we can call this function by n(x)
        # tanh(w * x + b), where w * x is the canonical scalar product
        assert len(x) == len(self.w), f"x should be a vector of length {len(self.w)}, but got a vector of length {len(x)}."
        # w * x + b
        # act = sum((wi * xi for wi, xi in zip(self.w, x)), start=0.0) + self.b # A better way is to start from self.b straight away
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b) # This is a Value, since both self.b and self.w are Values
        out = act.tanh()
        return out
    
    def parameters(self): # Collects all the parameters of a single neuron -> Ueful for risk minimization!
        return self.w + [self.b] # Concatenation, not addition!

class Layer:
    """
    nin (__int__): number of inputs to each single neuron
    nout (__int__): number of neurons in the layer
    """
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    """
    nin (__int__): number of inputs to each single neuron
    nouts(__list[__int__]__): list of values of size of each layer.
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # Concatenation, not addition!
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]