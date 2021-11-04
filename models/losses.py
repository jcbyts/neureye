"""
Custom loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoissonNLLDatFilterLoss(nn.PoissonNLLLoss):
    r"""Negative log likelihood loss with Poisson distribution of target.

    WITH SUPPORT FOR DATAFILTERS

    The loss can be described as:

    .. math::
        \text{target} \sim \mathrm{Poisson}(\text{input})

        \text{loss}(\text{input}, \text{target}) = \text{input} - \text{target} * \log(\text{input})
                                    + \log(\text{target!})

    The last term can be omitted or approximated with Stirling formula. The
    approximation is used for target values more than 1. For targets less or
    equal to 1 zeros are added to the loss.

    Args:
        log_input (bool, optional): if ``True`` the loss is computed as
            :math:`\exp(\text{input}) - \text{target}*\text{input}`, if ``False`` the loss is
            :math:`\text{input} - \text{target}*\log(\text{input}+\text{eps})`.
        full (bool, optional): whether to compute full loss, i. e. to add the
            Stirling approximation term

            .. math::
                \text{target}*\log(\text{target}) - \text{target} + 0.5 * \log(2\pi\text{target}).
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input = False`. Default: 1e-8
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> loss = nn.PoissonNLLLoss()
        >>> log_input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> output = loss(log_input, target)
        >>> output.backward()

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`,
          the same shape as the input
    """
    __constants__ = ['log_input', 'full', 'eps', 'reduction']
    log_input: bool
    full: bool
    eps: float

    def __init__(self, log_input: bool = True, full: bool = False, size_average=None,
                 eps: float = 1e-8, reduce=None, reduction: str = 'mean') -> None:
        super(PoissonNLLDatFilterLoss, self).__init__(size_average, reduce, reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, log_input: torch.Tensor, target: torch.Tensor, datafilters=None) -> torch.Tensor:
        if datafilters is not None:
            loss = F.poisson_nll_loss(log_input, target, log_input=self.log_input, full=self.full,
                                  eps=self.eps, reduce=None, reduction='none')
            df = datafilters * loss
            loss = torch.sum(df)
            if self.reduction == 'mean':
                loss = loss / torch.sum(datafilters)
        else:
            loss = F.poisson_nll_loss(log_input, target, log_input=self.log_input, full=self.full,
                                  eps=self.eps, reduction=self.reduction)

        return loss