
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

import configs.News_20
from bases.nn.conv2d import DenseConv2d
from bases.nn.linear import DenseLinear
from bases.nn.models.base_model import BaseModel
from bases.nn.sequential import DenseSequential
from bases.nn.models.utils import is_conv, is_fc
from collections import OrderedDict
import torch.nn as nn
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import LogNormal, Dirichlet
from torch.distributions import kl_divergence
from torch.utils.data import DataLoader

class ProdLDA(BaseModel):
    def __init__(self, dict_module: dict = None,vocab_size = None, hidden_size = None, num_topics = None,
                 dropout = None, use_lognormal=False,device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),train_data=None):
        learn_priors = True
        topic_prior_mean = 0.0
        topic_prior_variance = None
        self.input_size = vocab_size
        self.n_components = num_topics
        self.hidden_sizes = hidden_size
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.final_topic_word = None
        self.final_topic_document = None
        self.best_loss_train = None
        self.num_samples = num_topics
        self.use_lognormal =use_lognormal
        self.num_topics = num_topics
        self.device = device
        self.train_data = train_data
        if dict_module is None:
            dict_module = dict()
            input_layer = DenseSequential(DenseLinear(vocab_size,hidden_size[0]),
                                            nn.Softplus())



            hiddens = DenseSequential(OrderedDict([
                ('l_{}'.format(i), DenseSequential(DenseLinear(h_in, h_out), nn.Softplus()))
                for i, (h_in, h_out) in enumerate(zip(hidden_size[:-1], hidden_size[1:]))]))

            f_drop = nn.Dropout(p=self.dropout)
            f_mu = DenseSequential(DenseLinear(hidden_size[-1],num_topics),
                                                nn.BatchNorm1d(num_topics,affine=False)
                                                )


            f_sigma = DenseSequential(DenseLinear(hidden_size[-1],num_topics),
                                                nn.BatchNorm1d(num_topics,affine=False)
                                                )




            beta_batchnorm = nn.BatchNorm1d(vocab_size, affine=False)

            # dropout on theta
            drop_theta = nn.Dropout(p=self.dropout)




            dict_module["input_layer"] = input_layer
            dict_module["hiddens"] = hiddens

            dict_module['f_drop'] = f_drop
            dict_module["f_mu"] = f_mu
            dict_module["f_sigma"] = f_sigma
            dict_module['beta_batchnorm'] = beta_batchnorm
            dict_module['drop_theta'] = drop_theta


        super(ProdLDA, self).__init__(binary_cross_entropy_with_logits, dict_module)
        self.beta = torch.Tensor(num_topics, vocab_size)
        if torch.cuda.is_available():
            self.beta = self.beta.to(device)

        self.beta = nn.Parameter(self.beta)

        nn.init.xavier_uniform_(self.beta)
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * num_topics)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.to(device)
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        if topic_prior_variance is None:
            topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * num_topics)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.to(device)
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)#修改了param_layers和param_layer_predixes
        self.prunable_layers = self.param_layers
        self.prunable_layer_prefixes = self.param_layer_prefixes

    def forward(self, inputs):

        outputs = self.input_layer(inputs)
        outputs = self.hiddens(outputs)

        outputs = self.f_drop(outputs)
        posterior_mu = self.f_mu(outputs)
        posterior_log_sigma = self.f_sigma(outputs)


        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        topic_doc = theta
        theta = self.drop_theta(theta)
        # in: batch_size x input_size x n_components
        word_dist = F.softmax(
            self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
        topic_word = self.beta
        # word_dist: batch_size x input_size
        self.topic_word_matrix = self.beta
        self.final_topic_word = topic_word
        self.final_topic_document = topic_doc




        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, topic_word,topic_doc

    def get_theta(self, x):
        with torch.no_grad():
            # batch_size x n_components
            prior_mean, prior_var, posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, topic_word,topic_doc = self.forward(x)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta

    def _evaluate(self, model,  loader):

        """Train epoch."""
        model.eval()
        val_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            # batch_size x vocab_size
            x = batch_samples['X']


            x = x.to(self.device)
            # forward pass
            model.zero_grad()
            prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
            word_dists, topic_word, topic_document = model(x)

            loss = self._loss(x, word_dists, prior_mean, prior_var,
                              posterior_mean, posterior_var, posterior_log_var)

            # compute train loss
            samples_processed += x.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def _train(self, x, model, optimizer, device):
        model.train()

        x = x.to(device)
        train_loss = 0
        samples_processed = 0
        topic_doc_list = []

        prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
        word_dists, topic_words, topic_document = model(x)

        topic_doc_list.extend(topic_document)
        optimizer.zero_grad()

        # backward pass
        loss = self._loss(x, word_dists, prior_mean, prior_var,
                              posterior_mean, posterior_var, posterior_log_var)
        loss.backward()
        optimizer.step()

        # compute train loss
        samples_processed += np.size(x,0)
        train_loss += loss.item()

        train_loss /= samples_processed

        self.best_loss_train = train_loss


        return samples_processed, train_loss, topic_words, topic_doc_list



    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (var_division + diff_term - self.num_topics + logvar_det_division)
        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        loss = KL + RL

        return loss.sum()

    def predict(self, dataset,model):
        """Predict input."""
        model.eval()

        loader = DataLoader(dataset, batch_size=configs.News_20.CLIENT_BATCH_SIZE, shuffle=False,
                            num_workers=0)

        topic_document_mat = []
        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                x = batch_samples['X']
                x = x.reshape(x.shape[0], -1)

                x = x.to(self.device)
                # forward pass
                model.zero_grad()
                _, _, _, _, _, _, _, topic_document = model(x)
                topic_document_mat.append(topic_document)

        results = self.get_info()
        # results['test-topic-document-matrix2'] = np.vstack(
        #    np.asarray([i.cpu().detach().numpy() for i in topic_document_mat])).T
        results['test-topic-document-matrix'] = np.asarray(self.get_thetas(dataset)).T

        return results



    def get_topic_word_mat(self):
        top_wor = self.final_topic_word.cpu().detach().numpy()
        return top_wor

    def get_topic_document_mat(self):
        top_doc = self.final_topic_document
        top_doc_arr = np.array([i.cpu().detach().numpy() for i in top_doc])
        return top_doc_arr

    def get_topics(self, k=10):
        """
        Retrieve topic words.

        Args
            k : (int) number of words to return per topic, default 10.
        """
        assert k <= self.input_size, "k must be <= input size."
        component_dists = self.beta
        from collections import defaultdict
        topics = defaultdict(list)
        topics_list = []
        if self.num_topics is not None:
            for i in range(self.num_topics):
                _, idxs = torch.topk(component_dists[i], k)
                component_words = [self.train_data.idx2token[idx]
                                   for idx in idxs.cpu().numpy()]
                topics[i] = component_words
                topics_list.append(component_words)

        return topics_list

    def get_info(self):
        info = {}
        topic_word = self.get_topics()
        topic_word_dist = self.get_topic_word_mat()
        # topic_document_dist = self.get_topic_document_mat()
        info['topics'] = topic_word

        # info['topic-document-matrix2'] = topic_document_dist.T
        info['topic-document-matrix'] = np.asarray(self.get_thetas(self.train_data)).T

        info['topic-word-matrix'] = topic_word_dist
        return info

    def get_thetas(self, dataset):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter num_samples.
        :param dataset: a PyTorch Dataset containing the documents
        """
        self.eval()

        loader = DataLoader(
            dataset, batch_size=configs.News_20.CLIENT_BATCH_SIZE, shuffle=False, num_workers=0)
        final_thetas = []
        for sample_index in range(self.num_samples):
            with torch.no_grad():
                collect_theta = []
                for batch_samples in loader:
                    # batch_size x vocab_size
                    x = batch_samples['X']
                    x = x.reshape(x.shape[0], -1)
                    x = x.to(self.device)
                    # forward pass
                    self.zero_grad()
                    collect_theta.extend(self.get_theta(x).cpu().numpy().tolist())

                final_thetas.append(np.array(collect_theta))
        return np.sum(final_thetas, axis=0) / self.num_samples

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