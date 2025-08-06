from torch import nn

__all__ = ["Cost", "FocalLossCost", "BBoxL1Cost", "IoUCost", "IoULossCost"]


class Cost(nn.Module):
    def __init__(self, weight: float):
        super(Cost, self).__init__()
        self.weight = weight


class FocalLossCost(Cost):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            eps: float = 1e-12,
            binary_input: bool = False,
            weight: float = 2.0
    ):
        super(FocalLossCost, self).__init__(weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input


class BBoxL1Cost(Cost):
    def __init__(
            self,
            box_format: str = "xyxy",
            weight: float = 5.0
    ):
        super(BBoxL1Cost, self).__init__(weight)
        self.box_format = box_format


class IoUCost(Cost):
    def __init__(self, iou_mode: str = "giou", weight: float = 2.0):
        super(IoUCost, self).__init__(weight)
        self.iou_mode = iou_mode


class IoULossCost(Cost):
    def __init__(self, iou_mode: str = "giou", weight: float = 1.0):
        super(IoULossCost, self).__init__(weight)
        self.iou_mode = iou_mode
