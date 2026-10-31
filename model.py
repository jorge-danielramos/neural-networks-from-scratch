import numpy as np
import nn


class MLPClassifier_1(nn.Module):
    """Single-layer perceptron: Z = W @ X + b (no hidden layer)."""

    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: number of input features D_in.
            out_features: number of output features D_out.
        """
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: input, shape (batch_size, in_features).

        Returns:
            Z: output, shape (batch_size, out_features).
        """
        ##CODE
        # TODO: run X through self.layer and return the result.
        Z = self.layer.forward(X)
        return Z
        ##END

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        Args:
            dZ: gradient of the loss w.r.t. the output Z,
                shape (batch_size, out_features).

        Returns:
            dX: gradient of the loss w.r.t. the input X,
                shape (batch_size, in_features).
        """
        ##CODE
        # TODO: backpropagate dZ through self.layer and return dX.
        dX = self.layer.backward(dZ)
        return dX
        ##END


class MLPClassifier_2(nn.Module):
    """Two-layer MLP: Linear -> activation -> Linear."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: type[nn.Module] = nn.Sigmoid,
    ):
        """
        Args:
            in_features: number of input features D_in.
            hidden_features: number of hidden units D_hidden.
            out_features: number of output features D_out.
            activation: `nn.Module` subclass used as the hidden activation
                (e.g. `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`).
        """
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.act1 = activation()
        self.layer2 = nn.Linear(hidden_features, out_features)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: input, shape (batch_size, in_features).

        Returns:
            Z2: output, shape (batch_size, out_features).
        """
        ##CODE
        # TODO: chain self.layer1 -> self.act1 -> self.layer2 and return Z2.
        Z1 = self.layer1.forward(X)
        H = self.act1.forward(Z1)
        Z2 = self.layer2.forward(H)
        return Z2

        ##END

    def backward(self, dZ2: np.ndarray) -> np.ndarray:
        """
        Args:
            dZ2: gradient of the loss w.r.t. the output Z2,
                shape (batch_size, out_features).

        Returns:
            dX: gradient of the loss w.r.t. the input X,
                shape (batch_size, in_features).
        """
        ##CODE
        # TODO: backpropagate dZ2 through layer2 -> act1 -> layer1, in that
        # order, and return dX.
        dH = self.layer2.backward(dZ2)
        dZ1 = self.act1.backward(dH)
        dX = self.layer1.backward(dZ1)
        return dX
        ##END
