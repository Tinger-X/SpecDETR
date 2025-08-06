from torch import nn

__all__ = ["nnDecoder", "SpodDecoder"]


class nnDecoder(nn.Module):
    def __init__(self):
        super(nnDecoder, self).__init__()


class SpodDecoder(nnDecoder):
    def __init__(self):
        super(SpodDecoder, self).__init__()
