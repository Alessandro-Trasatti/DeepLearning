## Micrograd

It is an *autograd* engine.

### Autograd engines
An autograd engine is a tool commonly used in machine learning frameworks to perform automatic differentiation (more commonly known as **back propagation**). Automatic differentiation is essential in training neural networks because it allows the efficient calculation of gradients (partial derivatives) of a loss function with respect to model parameters. These gradients are then used in optimization algorithms like gradient descent to update the model parameters and minimize the loss.

The core idea behind an autograd engine is to keep track of operations and build a computation graph as data passes through the model. This graph enables reverse-mode differentiation, which is particularly efficient for models with many parameters, like neural networks. Here’s a breakdown of how it works:

1.	*Forward Pass*: As you pass data through the model, the autograd engine records each operation in a computation graph. Each node in this graph represents an operation (e.g., addition, multiplication), and each edge represents the flow of data (tensor values).
2.	*Backward Pass*: When you call for gradients, the engine performs backpropagation using the chain rule, starting from the output and moving backward through the graph. It computes derivatives for each operation, ultimately yielding the gradient of the loss with respect to each parameter.
3.	*Efficiency*: The autograd engine leverages optimization techniques to compute gradients efficiently, especially by using reverse-mode autodiff, which is more efficient than forward-mode autodiff for functions with many inputs and fewer outputs (like neural networks).

##### Examples:
- *PyTorch* includes an autograd engine <code>torch.autograd</code> 
- TensorFlow’s <code>tf.GradientTape</code>