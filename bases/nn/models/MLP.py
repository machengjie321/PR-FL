from torch import nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits

from bases.nn.conv2d import DenseConv2d
from bases.nn.linear import DenseLinear
from bases.nn.models.base_model import BaseModel
from bases.nn.sequential import DenseSequential

__all__ = ["MLP"]


class MLP(BaseModel):
    def __init__(self, dict_module: dict = None):
        if dict_module is None:
            dict_module = dict()
            features = DenseSequential(DenseLinear(784, 200, a=0),
                                         nn.ReLU(inplace=True),
                                         DenseLinear(200, 510, a=0),
                                         nn.ReLU(inplace=True),)
            classifier = DenseSequential(DenseLinear(200, 10, a=0),)

            dict_module["features"] = features
            dict_module["classifier"] = classifier

        super(MLP, self).__init__(binary_cross_entropy_with_logits, dict_module)

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        prunable_nums = [ly_id for ly_id, ly in enumerate(self.param_layers) if not isinstance(ly, nn.BatchNorm2d)]
        self.prunable_layers = [self.param_layers[ly_id] for ly_id in prunable_nums]
        self.prunable_layer_prefixes = [self.param_layer_prefixes[ly_id] for ly_id in prunable_nums]



    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.classifier(outputs)
        return outputs

    def to_sparse(self):
        new_features = [ft.to_sparse() if isinstance(ft, DenseConv2d) else ft for ft in self.features]
        new_module_dict = {"features": nn.Sequential(*new_features), "classifier": self.classifier.to_sparse()}
        return self.__class__(new_module_dict)
