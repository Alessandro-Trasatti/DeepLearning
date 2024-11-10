## Softmax function
The *softmax function*, also known as softargmax or normalized exponential function, converts a vector of $K$ real numbers into a probability distribution of $K$ possible outcomes.

It is a generalization of the logistic function to multiple dimensions, and is used in multinomial logistic regression. The softmax function is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes.

### Definition
The softmax function takes as input a vector $\mathbf{z}$ of $K$ real numbers (in the context of neural networks, the entries of $\mathbf{z}$ are called *logits*), and normalizes it into a probability distribution consisting of $K$ probabilities proportional to the exponentials of the input numbers. That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to $1$; but after applying softmax, each component will be in the interval $(0,1)$, and the components will add up to $1$, so that they can be interpreted as probabilities. Furthermore, the larger input components will correspond to larger probabilities.

Formally, the standard (unit) softmax function 

$$
\sigma \colon \mathbb {R}^{K}\to (0,1)^{K},
$$

where $K\geq 1$, takes a vector $\mathbf {z} =(z_{1},\dotsc ,z_{K})\in \mathbb {R} ^{K}$ and computes each component of vector $\sigma (\mathbf {z} )\in (0,1)^{K}$ with

$$
\sigma (\mathbf {z} )_{i}={\frac {e^{z_{i}}}{\sum _{j=1}^{K}e^{z_{j}}}}
$$