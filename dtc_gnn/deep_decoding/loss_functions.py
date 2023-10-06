import torch


class LogCoshLoss(torch.nn.Module):
    """
    The logarithm of the hyperbolic cosine loss function.
    """
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    @staticmethod
    def forward(y_pred, y_true):
        return torch.mean(
            torch.log(torch.cosh(y_pred - y_true))
        )
