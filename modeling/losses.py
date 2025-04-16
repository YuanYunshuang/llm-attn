import torch
from torch import nn
import torch.nn.functional as F


def build_loss(type, **kwargs):
    return globals()[type](**kwargs)


class BaseLoss(nn.Module):
    def __init__(self,
                 reduction: str = 'mean',
                 activation: str = 'none',
                 loss_weight: float = 1.0):
        """
        :param reduction: (optional) the method to reduce the loss.
        :param activation: options are "none", "mean" and "sum".
        :param loss_weight: (optional) the weight of loss.
        """
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activation = activation

    @property
    def name(self):
        return self.__class__.__name__

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor,
                weight: torch.Tensor=None,
                avg_factor: int=None,
                reduction_override: str=None,
                *args, **kwargs) -> torch.Tensor:
        """

        :param preds: prediction tensor.
        :param targets: target tensor.
        :param weight: The weight of loss for each
                prediction. Defaults to None.
        :param avg_factor: Average factor that is used to average
                the loss. Defaults to None.
        :param reduction_override: The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        :param args: additional arguments.
        :param kwargs:
        :return: weighted loss.
        """
        loss = self.loss(preds, targets, *args, **kwargs)
        # if weight is specified, apply element-wise weight
        if weight is not None:
            loss = loss * weight
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                # Avoid causing ZeroDivisionError when avg_factor is 0.0,
                # i.e., all labels of an image belong to ignore index.
                eps = torch.finfo(torch.float32).eps
                loss = loss.sum() / (avg_factor + eps)
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return self.loss_weight * loss


def quality_focal_loss(pred: torch.Tensor,
                       target: tuple([torch.Tensor]),
                       beta: float = 2.0) -> torch.Tensor:
    r"""
    Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    :param pred: Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
    :param target: Target category label with shape (N,)
            and target quality label with shape (N,).
    :param beta: The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
    :return: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


def quality_focal_loss_with_prob(pred: torch.Tensor,
                       target: tuple([torch.Tensor]),
                       beta: float = 2.0) -> torch.Tensor:
    r"""
    Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    :param pred: Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
    :param target: Target category label with shape (N,)
            and target quality label with shape (N,).
    :param beta: The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
    :return: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


class QualityFocalLoss(BaseLoss):
    def __init__(self,
                 use_sigmoid: bool=True,
                 beta: float=2.0,
                 activated: bool=False,
                 **kwargs):
        r"""
        Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
        Learning Qualified and Distributed Bounding Boxes for Dense Object
        Detection <https://arxiv.org/abs/2006.04388>`_.

        :param use_sigmoid: Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        :param beta: The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        :param activated: (optional) Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
        :param kwargs:
        """
        super(QualityFocalLoss, self).__init__(**kwargs)
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.activated = activated

    def loss(self, pred: torch.Tensor, target: torch.Tensor):
        """Forward function.

        :param pred: Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
        :param target: Target category label with shape
                (N,) and target quality label with shape (N,).
        :return: loss result.
        """
        if self.use_sigmoid:
            if self.activated:
                loss_cls = quality_focal_loss_with_prob(pred, target, self.beta)
            else:
                loss_cls = quality_focal_loss(pred, target, self.beta)
        else:
            raise NotImplementedError
        return loss_cls


class GaussianFocalLoss(BaseLoss):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.
    """

    def __init__(self,
                 alpha: float=2.0,
                 gamma: float=4.0,
                 reduction: str='mean',
                 loss_weight: float=1.0):
        """

        :param alpha: Power of prediction.
        :param gamma: Power of target for negative samples.
        :param reduction: Options are "none", "mean" and "sum".
        :param loss_weight: Loss weight of current loss.
        """
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def loss(self, pred: torch.Tensor, target: torch.Tensor):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
        distribution.

        :param pred: The prediction.
        :param target: The learning target of the prediction
                in gaussian distribution.
        :return: loss result.
        """
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(self.alpha) * neg_weights
        return pos_loss + neg_loss


def py_focal_loss_with_prob(pred: torch.Tensor,
                            target: torch.Tensor,
                            gamma: float=2.0,
                            alpha: float=0.25):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    :param pred: The prediction probability with shape (N, C),
            C is the number of classes.
    :param target: The learning label of the prediction.
    :param gamma: The gamma for calculating the modulating
            factor. Defaults to 2.0.
    :param alpha: A balanced form for Focal Loss.
            Defaults to 0.25.
    :return: loss result.
    """
    num_classes = pred.size(1)
    assert pred.ndim == target.ndim or pred.ndim == target.ndim + 1
    if pred.ndim == target.ndim + 1:
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    return loss


def py_sigmoid_focal_loss(pred: torch.Tensor,
                            target: torch.Tensor,
                            gamma: float=2.0,
                            alpha: float=0.25):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    :param pred: The prediction probability with shape (N, C),
            C is the number of classes.
    :param target: The learning label of the prediction.
    :param gamma: The gamma for calculating the modulating
            factor. Defaults to 2.0.
    :param alpha: A balanced form for Focal Loss.
            Defaults to 0.25.
    :return: loss result.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    return loss


def py_softmax_focal_loss(pred: torch.Tensor,
                            target: torch.Tensor,
                            gamma: float=2.0,
                            alpha: float=0.25):

    target = target.type_as(pred)
    CE_loss = F.cross_entropy(pred, target, reduction='none')  # -log(pt)
    pt = torch.exp(-CE_loss)

    focal_weight = alpha * (1 - pt).pow(gamma)
    loss = CE_loss * focal_weight
    return loss


class FocalLoss(BaseLoss):

    def __init__(self,
                 use_sigmoid: bool=True,
                 gamma: float=2.0,
                 alpha: float=0.25,
                 activated: bool=False,
                 bg_idx: int=None,
                 **kwargs):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        :param use_sigmoid: Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
        :param gamma: The gamma for calculating the modulating
                factor. Defaults to 2.0.
        :param alpha: A balanced form for Focal Loss.
                Defaults to 0.25.
        :param activated:  Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        :param bg_idx: background class index.
        :param kwargs:
        """
        super(FocalLoss, self).__init__(**kwargs)
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.activated = activated
        self.bg_idx = bg_idx
        if use_sigmoid:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'

    def loss(self, pred: torch.Tensor, target: torch.Tensor, *args, **kwargs):
        """
        :param pred: prediction.
        :param target: ground truth targets.
        :param args:
        :param kwargs:
        :return:
        """
        num_classes = pred.size(1)
        if self.use_sigmoid and isinstance(target, torch.cuda.FloatTensor):
            target = torch.stack([1 - target, target], dim=-1)
        else:
            if target.ndim == 1:
                target = F.one_hot(target.long(), num_classes=num_classes + 1)
            elif target.shape[-1] == num_classes - 1:
                target = torch.cat([1 - target.sum(dim=-1, keepdim=True), target], dim=-1)
            else:
                assert target.shape[-1] == num_classes
        if self.bg_idx is None:
            target = target[:, :num_classes]
        else:
            target = target[:, [c for c in range(num_classes + 1) if c != self.bg_idx]]

        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                calculate_loss_func = py_sigmoid_focal_loss
        else:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                calculate_loss_func = py_softmax_focal_loss

        loss_cls = calculate_loss_func(
            pred,
            target,
            gamma=self.gamma,
            alpha=self.alpha)

        return loss_cls


class BinaryFocalLoss(BaseLoss):

    def __init__(self,
                 use_sigmoid: bool=True,
                 gamma: float=2.0,
                 alpha: float=0.25,
                 activated: bool=False,
                 bg_idx: int=None,
                 **kwargs):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        :param use_sigmoid: Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
        :param gamma: The gamma for calculating the modulating
                factor. Defaults to 2.0.
        :param alpha: A balanced form for Focal Loss.
                Defaults to 0.25.
        :param activated:  Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        :param bg_idx: background class index.
        :param kwargs:
        """
        super(BinaryFocalLoss, self).__init__(**kwargs)
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.activated = activated
        self.bg_idx = bg_idx
        if use_sigmoid:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'

    def loss(self, pred: torch.Tensor, target: torch.Tensor, *args, **kwargs):
        """
        :param pred: prediction.
        :param target: ground truth targets.
        :param args:
        :param kwargs:
        :return:
        """
        loss_cls = py_sigmoid_focal_loss(
            pred,
            target,
            gamma=self.gamma,
            alpha=self.alpha)

        return loss_cls


