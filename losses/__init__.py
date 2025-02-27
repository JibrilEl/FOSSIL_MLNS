from .LossFunction import LossFunction
from .ReadoutLoss import ReadoutLoss
from .FOSSILLoss import FOSSILLoss
from .FOSSILLossV2 import FOSSILLossV2


def build_loss(loss_fn='fgwd', **kwargs):
    if loss_fn == 'readout':
        return ReadoutLoss(**kwargs)
    elif loss_fn == 'fgwd':
        return FOSSILLoss(**kwargs)
    elif loss_fn == 'fgwd_v2':
        return FOSSILLossV2(**kwargs)
    else:
        raise Exception("wrong loss_function")
