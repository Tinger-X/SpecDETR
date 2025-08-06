import torch
from torch import nn, Tensor
import torch.nn.functional as F
import functools

__all__ = ["Loss", "FocalLoss", "L1Loss", "GIoULoss"]


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(
        loss: Tensor,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        avg_factor: Optional[float] = None
) -> Tensor:
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == "mean":
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != "none":
            raise ValueError("avg_factor can not be used with reduction='sum'")
    return loss


def weighted_loss(loss_func: Callable) -> Callable:
    @functools.wraps(loss_func)
    def wrapper(
            pred: Tensor,
            target: Tensor,
            weight: Optional[Tensor] = None,
            reduction: str = "mean",
            avg_factor: Optional[int] = None,
            **kwargs
    ) -> Tensor:
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def py_sigmoid_focal_loss(
        pred,
        target,
        weight=None,
        gamma=2.0,
        alpha=0.25,
        reduction="mean",
        avg_factor=None
):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(
        pred,
        target,
        weight=None,
        gamma=2.0,
        alpha=0.25,
        reduction="mean",
        avg_factor=None
):
    if pred.dim() != target.dim():
        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction="none") * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(
        pred,
        target,
        weight=None,
        gamma=2.0,
        alpha=0.25,
        reduction="mean",
        avg_factor=None
):
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma, alpha, None, "none")
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@weighted_loss
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


def fp16_clamp(x, _min=None, _max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(_min, _max).half()
    return x.clamp(_min, _max)


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    enclosed_rb, enclosed_lt = 0, 0
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]
        wh = fp16_clamp(rb - lt, _min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]
        wh = fp16_clamp(rb - lt, _min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, _min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


@weighted_loss
def giou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()


class FocalLoss(Loss):
    def __init__(
            self,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction="mean",
            loss_weight=1.0,
            activated=False
    ):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid focal loss supported now."
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(
            self,
            pred,
            target,
            weight=None,
            avg_factor=None,
            reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        assert self.use_sigmoid is True, "Only sigmoid focal loss supported now."
        reduction = reduction_override if reduction_override else self.reduction
        if self.activated:
            calculate_loss_func = py_focal_loss_with_prob
        else:
            if pred.dim() == target.dim():
                calculate_loss_func = py_sigmoid_focal_loss
            elif torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss

        loss_cls = self.loss_weight * calculate_loss_func(
            pred, target, weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor
        )
        return loss_cls


class L1Loss(Loss):
    def __init__(
            self,
            reduction: str = "mean",
            loss_weight: float = 5.0
    ):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
            self,
            pred: Tensor,
            target: Tensor,
            weight: Optional[Tensor] = None,
            avg_factor: Optional[int] = None,
            reduction_override: Optional[str] = None
    ) -> Tensor:
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


class GIoULoss(Loss):

    def __init__(
            self,
            eps: float = 1e-6,
            reduction: str = "mean",
            loss_weight: float = 2.0
    ):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
            self,
            pred: Tensor,
            target: Tensor,
            weight: Optional[Tensor] = None,
            avg_factor: Optional[int] = None,
            reduction_override: Optional[str] = None,
            **kwargs
    ) -> Tensor:
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss
