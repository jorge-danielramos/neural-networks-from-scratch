import numpy as np
from nn import Module


class SGD:
    """Stochastic Gradient Descent optimizer: param <- param - lr * grad."""

    def __init__(self, model: Module, lr: float = 0.01):
        """
        Args:
            model: Module whose `.parameters()` yields (param, grad) pairs,
                where `param` and `grad` are numpy arrays of identical shape.
            lr: learning rate, scalar step size applied to every gradient.
        """
        self.model = model
        self.lr = lr

    def step(self) -> None:
        """Updates every parameter in-place using its current gradient."""
        ##CODE
        # TODO: for every (param, grad) pair in self.model.parameters(),
        # apply the SGD update param <- param - lr * grad (in-place).
        for param, grad in self.model.parameters():
            param -= self.lr * grad
        ##END

    def zero_grad(self) -> None:
        """Resets every parameter's gradient buffer to zero, in-place."""
        for _, grad in self.model.parameters():
            grad.fill(0)
