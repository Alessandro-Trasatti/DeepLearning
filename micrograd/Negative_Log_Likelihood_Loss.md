## Negative Log-Likelihood Loss
The *Negative Log-Likelihood (NLL) loss*, commonly called *log loss*, is a function often used in classification problems, especially when training neural networks and other probabilistic models. Here’s a breakdown of what it does and why it’s so popular in this context:

### 1. What Does NLL Measure?
NLL measures how well a model’s predicted probabilities align with the actual classes in the data. Lower values indicate a closer match between predictions and true labels, while higher values suggest poor alignment. In other words, NLL penalizes incorrect or uncertain predictions, encouraging the model to output high probabilities for the correct class.

### 2. Understanding NLL Loss Formula
For a classification problem with $C$ possible classes and an input $x$, the model will output a probability distribution  $p(y = c \mid x)$  for each class  $c$  (where  $c = 1, 2, \ldots, C$).

Given the true label  $y$ , the NLL loss for a single data point is:
$$
\text{NLL} = -\log(p(y \mid x))
$$
This means:
- If  $p(y \mid x)$  is close to $1$ (meaning the model is confident in the correct class), $\log(p(y \mid x))$ is close to zero, so NLL is low.
- If  $p(y \mid x)$  is close to $0$ (low confidence or incorrect prediction), $\log(p(y \mid x))$ is very negative, and the NLL loss is high.

In practice, we calculate the average NLL over a batch or dataset of  $N$  examples:
$$
\text{NLL}_{\text{total}} = -\frac{1}{N}\sum_{i = 1}^N\log(p(y_i \mid x_i))
$$
Here,  $y_i$  is the actual label for the  $i$-th example, and  $p(y_i \mid x_i)$  is the predicted probability for that label.
