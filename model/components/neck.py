from torch import nn

__all__ = ["nnNeck", "SpodNeck"]


class nnNeck(nn.Module):
    def __init__(self):
        super(nnNeck, self).__init__()


class SpodNeck(nnNeck):
    def __init__(self):
        super(SpodNeck, self).__init__()
