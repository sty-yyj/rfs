from .convnet import convnet4
from .resnet import resnet12
from .resnet import seresnet12
from .wresnet import wrn_28_10

from .resnet_new import resnet50
from .modules import DynamicConv


model_pool = [
    'convnet4',
    'resnet12',
    'seresnet12',
    'wrn_28_10',
]

model_dict = {
    'wrn_28_10': wrn_28_10,
    'convnet4': convnet4,
    'resnet12': resnet12,
    'seresnet12': seresnet12,
    'resnet50': resnet50,
}

from .util import special_softmax
from .attention_module import make_model
from .context_block import ContextBlock

__all__ = ['make_model', 'special_softmax', 'ContextBlock', 'DynamicConv']
