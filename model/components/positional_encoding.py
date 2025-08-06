from torch import nn

__all__ = ["nnPositionalEncoding", "SpodPositionalEncoding"]


class nnPositionalEncoding(nn.Module):
    def __init__(self):
        super(nnPositionalEncoding, self).__init__()


class SpodPositionalEncoding(nnPositionalEncoding):
    def __init__(self):
        super(SpodPositionalEncoding, self).__init__()
