import torch
from torch import nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import torch.nn.functional as F
from bases.nn.conv2d import DenseConv2d
from bases.nn.linear import DenseLinear
from bases.nn.models.base_model import BaseModel
from bases.nn.sequential import DenseSequential
from bases.nn.models.utils import is_conv, is_fc
from torch.distributions import LogNormal, Dirichlet
import torch
import torch.nn as nn

import math
import torch
import torch.nn.functional as F
from torch.distributions import LogNormal, Dirichlet
from torch.distributions import kl_divergence

class ProdLDA(BaseModel):
    def __init__(self, dict_module: dict = None,vocab_size = None, hidden_size = None, num_topics = None,
                 dropout = None, use_lognormal=False):
        self.use_lognormal =use_lognormal
        if dict_module is None:
            dict_module = dict()
            Encoder = DenseSequential(DenseLinear(vocab_size,hidden_size),
                                     nn.Softplus(),
                                     DenseLinear(hidden_size,hidden_size),
                                     nn.Softplus(),
                                     nn.Dropout(dropout))
            if use_lognormal:
                HiddenToLogNormal_1 =   DenseSequential(DenseLinear(hidden_size,num_topics),
                                                    nn.BatchNorm1d(num_topics)


                                                )
                HiddenToLogNormal_2 =   DenseSequential(
                                                    DenseLinear(hidden_size, num_topics),
                                                    nn.BatchNorm1d(num_topics)
                                                )
            else:
                HiddenToLogNormal_1 = DenseSequential(DenseLinear(hidden_size,num_topics),
                                                      nn.BatchNorm1d(num_topics),

                                                )

            Decoder = DenseSequential(nn.Dropout(dropout),
                                      DenseLinear(num_topics,vocab_size),
                                      nn.BatchNorm1d(vocab_size))
            if use_lognormal:
                dict_module["Encoder"] = Encoder
                dict_module["HiddenToLogNormal_1"] = HiddenToLogNormal_1
                dict_module['HiddenToLogNormal_2'] = HiddenToLogNormal_2
                dict_module["Decoder"] = Decoder
            else:
                dict_module["Encoder"] = Encoder
                dict_module["HiddenToLogNormal_1"] = HiddenToLogNormal_1
                dict_module["Decoder"] = Decoder

        super(ProdLDA, self).__init__(binary_cross_entropy_with_logits, dict_module)

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)#修改了param_layers和param_layer_predixes
        self.prunable_layers = self.param_layers
        self.prunable_layer_prefixes = self.param_layer_prefixes

    def forward(self, inputs):
        outputs = self.Encoder(inputs)
        if self.use_lognormal:
            mu = self.HiddenToLogNormal_1(outputs)
            lv = self.HiddenToLogNormal_1(outputs)
            posterior = LogNormal(mu, (0.5 * lv).exp())
            outputs = posterior.rsample().to(inputs.device)
        else:
            outputs = self.HiddenToLogNormal_1(outputs).exp().cpu()
            posterior = Dirichlet(outputs)
            outputs = posterior.mean.to(inputs.device)
        outputs = outputs / outputs.sum(1, keepdim=True)
        outputs=F.log_softmax(self.Decoder(outputs), dim=1)
        return outputs,posterior

    @staticmethod
    def recon_loss(targets, outputs):
        nll = - torch.sum(targets * outputs)
        return nll
    @staticmethod
    def standard_prior_like(posterior):
        if isinstance(posterior, LogNormal):
            loc = torch.zeros_like(posterior.loc)
            scale = torch.ones_like(posterior.scale)
            prior = LogNormal(loc, scale)
        elif isinstance(posterior, Dirichlet):
            alphas = torch.ones_like(posterior.concentration)
            prior = Dirichlet(alphas)
        return prior

    def get_loss(self,inputs, model, device):
        inputs = inputs.to(device)
        outputs, posterior = model(inputs)
        prior = self.standard_prior_like(posterior)
        nll = self.recon_loss(inputs, outputs)
        kld = torch.sum(kl_divergence(posterior, prior).to(device))
        return nll, kld

    def _evaluate(self,config,data_source,model, device):
        model.eval()
        total_nll = 0.0
        total_kld = 0.0
        total_words = 0
        size = data_source.size
        for i in range(0, data_source.size, config.CLIENT_BATCH_SIZE):
            batch_size = min(data_source.size - i, config.CLIENT_BATCH_SIZE)
            data = data_source.get_batch(batch_size, i)
            nll, kld = self.get_loss(data, model, device)
            total_nll += nll.item() / size
            total_kld += kld.item() / size
            total_words += data.sum()
        loss = total_nll+total_kld
        ppl = math.exp(total_nll * size / total_words)
        return (total_nll, total_kld, ppl, loss)

    def _train(self,data_source, model,optimizer,device):
        self.train()
        nll, kld = self.get_loss(data_source, model, device)
        optimizer.zero_grad()
        loss = nll + kld
        loss.backward()
        optimizer.step()
        return(nll, kld, loss)

    def get_savepath(self,args):
        dataset = args.data.rstrip('/').split('/')[-1]
        path = '/mnt/sda1/mcj/PruneFL-master/PruneFL-master/bases/nn/models/saves/hid{0:d}.tpc{1:d}{2}.{3}.pt'.format(
            args.hidden_size, args.num_topics,
            '.wd{:.0e}'.format(args.wd) if args.wd > 0 else '',
            dataset)
        return path

    def print_top_words(self,beta, idx2word, n_words=10):
        print('-' * 30 + ' Topics ' + '-' * 30)
        for i in range(len(beta)):
            line = ' '.join(
                [idx2word[j] for j in beta[i].argsort()[:-n_words - 1:-1]])
            print(line)

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder

        return self.Decoder[1].weight.cpu().detach().numpy().T



    def to_sparse(self):
        new_features = [ft.to_sparse() if isinstance(ft, DenseConv2d) else ft for ft in self.features]
        new_module_dict = {"features": nn.Sequential(*new_features), "classifier": self.classifier.to_sparse()}
        return self.__class__(new_module_dict)

    def remove_empty_channels(self):
        list_in_out = []
        is_transition = False
        prev_is_transition = False
        for idx, (layer, next_layer) in enumerate(zip(self.prunable_layers, self.prunable_layers[1:] + [None])):
            # works for both conv and fc
            if is_conv(layer) and is_fc(next_layer):
                is_transition = True

            num_out, num_in = layer.weight.size()[:2]

            if idx == 0 or prev_is_transition:
                list_remain_in = "all"
            else:
                list_remain_in = set()
                for in_id in range(num_in):
                    mask_slice = layer.mask.index_select(dim=1, index=torch.tensor([in_id]))
                    if not torch.equal(mask_slice, torch.zeros_like(mask_slice)):
                        list_remain_in.add(in_id)
                if len(list_remain_in) == layer.weight.size()[1]:
                    list_remain_in = "all"

            if next_layer is None or is_transition:
                list_remain_out = "all"
            else:
                list_remain_out = set()
                for out_id in range(num_out):
                    mask_slice = layer.mask.index_select(dim=0, index=torch.tensor([out_id]))
                    if not torch.equal(mask_slice, torch.zeros_like(mask_slice)):
                        list_remain_out.add(out_id)
                if len(list_remain_out) == layer.weight.size()[0]:
                    list_remain_out = "all"

            list_in_out.append((list_remain_in, list_remain_out))

            if prev_is_transition:
                prev_is_transition = False
            if is_transition:
                prev_is_transition = True
                is_transition = False

        for ((in_indices, out_indices),
             (in_indices_next, out_indices_next),
             layer,
             next_layer) in zip(list_in_out[:-1], list_in_out[1:], self.prunable_layers[:-1],
                                self.prunable_layers[1:]):

            if out_indices == "all" or in_indices_next == "all":
                merged_indices = "all"
            else:
                merged_indices = list(out_indices.intersection(in_indices_next))

            if merged_indices != "all":
                layer.weight = nn.Parameter(layer.weight.index_select(dim=0, index=torch.tensor(merged_indices)))
                layer.mask = layer.mask.index_select(dim=0, index=torch.tensor(merged_indices))
                len_merged_indices = len(merged_indices)
                if layer.bias is not None:
                    layer.bias = nn.Parameter(layer.bias[merged_indices])
                if is_conv(layer):
                    layer.out_channels = len_merged_indices
                elif is_fc(layer):
                    layer.out_features = len_merged_indices

                next_layer.weight = nn.Parameter(
                    next_layer.weight.index_select(dim=1, index=torch.tensor(merged_indices)))
                next_layer.mask = next_layer.mask.index_select(dim=1, index=torch.tensor(merged_indices))
                if is_conv(next_layer):
                    next_layer.in_channels = len_merged_indices
                elif is_fc(next_layer):
                    next_layer.in_features = len_merged_indices