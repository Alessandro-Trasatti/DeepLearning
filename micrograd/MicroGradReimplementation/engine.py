import math

class Value():
    """Every Value() is specified by:
    1. data (_int_ or _float_): the value associated with the object;
    2. _children (_tuple_): It keeps track of from which Value objects the current Value object is generated, creating
                      like this a graph. In other words, it keeps track of the children nodes.
                      By default it is an empty tuple.
                      Important for the forward pass.
    3. _op (_string_): It stores the operation that led to the current Value starting from its children (or previous nodes).
    """
    def __init__(self, data, _children = (), _op = '', label = ''):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0 # By defualt, at the beginning, we suppose that any variable has no effect on the root.
        self._backward = lambda: None # By defualt, it doesn't do anything. This is the case for a leaf node, i.e. a node without predecessors. Note that for every type of operation (+, *, tankh, etc...) we need to specify how do go backward, i.e. how to do the derivative
        self.label = label
        self._op = _op
        
    def __repr__(self) -> str:
        return f"Value({self.data}, {self.label})"
    
    # operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')
        
        def _backward():
            # In the following, the += is necessary (and not just the =) as I might be doing x+x, and the derivative would be 2, not 1.
            self.grad += 1.0 * out.grad # 1.0 is the (partial) local derivative. In other words if f(x, y) = x + y => df/dx = 1 6 by the chain rule dL/dx = dL/d(x + y)*d(x+y)/dx. In our case dL/d(x+y) = out.grad()
            other.grad += 1.0 * out.grad # Here it is the partial derivative with respect to y...
        
        out._backward = _backward # recall that out.grad is a function...
        
        return out
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')
        
        def _backward():
            # f(x,y) = x * y => df/dx = y & df/dy = x. If f(x) = x**2 => df/dx = 2x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        
        return out
    
    # In case you call 2 * a, where a is a Value, without __rmul__, it would throw an error.
    def __rmul__(self, other): # other * self
        return self * other        
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
            
        out._backward = _backward

        return out
    
    def __truediv__(self, other): # selft / other
        return self * other**-1
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return self - other
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(data=t, _children=(self,), _op='tanh')
        
        def _backward():
            # f(x) = tanh(x) => df/dx = 1 - tanh(x)**2
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        
        return out
    
    def exp(self):
        e = math.exp(self.data)
        out = Value(data=e, _children=(self,), _op='tanh')
        
        def _backward():
            # f(x) = tanh(x) => df/dx = 1 - tanh(x)**2
            self.grad += out.data * out.grad
        
        out._backward = _backward
        
        return out
    
    def backward(self): # Backpropagation
        
        # Must be global: build topo is called iteratively on the children and we need to keep track of the visited sets as well as the topological sort
        visited = set()
        topo = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)            # Mark the node as visited
                for child in v._prev:     # Explore each child (dependency)
                    build_topo(child)     # Recursively build the topological order for the child
                topo.append(v)            # Add the node to topo after its children
            return topo

        topo = build_topo(self) # Does the topological sort...
        self.grad = 1.0 # Initializes the gradient to 1 for backpropagation from self...
        # Actually do the backpropagation
        for node in reversed(topo): # We need to go on reverse to do backpropagation
            node._backward()
        
    # getters and setters
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        assert isinstance(data, (int, float)), f"data must be an integer or float, but got {data}, which is of type {type(data)}."
        self._data = data