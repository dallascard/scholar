import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform


class Scholar(object):
    """
    Scholar: a neural model for documents with metadata
    """

    def __init__(self, config, alpha=1.0, learning_rate=0.001, init_embeddings=None, update_embeddings=True,
                 init_bg=None, update_background=True, update_beta=True, adam_beta1=0.99, adam_beta2=0.999):

        self.network_architecture = config
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.update_embeddings = update_embeddings
        self.update_background = update_background
        self.update_beta = update_beta

        # create priors on the hidden state
        self.h_dim = (config["n_topics"])

        # interpret alpha as either a (symmetric) scalar prior or a vector prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.h_dim)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.h_dim

        # create the pyTorch model
        self._model = torchScholar(config, self.alpha, update_embeddings, init_emb=init_embeddings, bg_init=init_bg)

        # set the criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # create the optimizer
        grad_params = filter(lambda p: p.requires_grad, self._model.parameters())
        self.optimizer = optim.Adam(grad_params, lr=learning_rate, betas=(adam_beta1, adam_beta2))

    def fit(self, X, Y, C, eta_bn_prop=1.0):
        """
        Fit the model to a minibatch of data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param C: np.array of covariates [batch size x n_covariates]
        :return: overall loss for minibatch; per-instance label predictions
        """
        X = Variable(torch.Tensor(X))
        if Y is not None:
            Y = Variable(torch.Tensor(Y))
        if C is not None:
            C = Variable(torch.Tensor(C))
        self.optimizer.zero_grad()
        _, X_recon, Y_probs, loss = self._model(X, Y, C, eta_bn_prop=eta_bn_prop)
        if Y_probs is not None:
            Y_probs = Y_probs.data.numpy()
        loss.backward()
        self.optimizer.step()
        return loss.data.numpy(), Y_probs

    def predict(self, X, C, eta_bn_prop=0.0):
        """
        Predict labels for a minibatch of data
        """
        # input a vector of all zeros in place of the labels that the model has been trained on
        batch_size = self.get_batch_size(X)
        Y = np.zeros((batch_size, self.network_architecture['n_labels'])).astype('float32')
        X = Variable(torch.Tensor(X))
        Y = Variable(torch.Tensor(Y))
        if C is not None:
            C = Variable(torch.Tensor(C))
        theta, _, Y_recon, _ = self._model(X, Y, C, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
        return theta, Y_recon.data.numpy()

    def predict_from_topics(self, theta, C, eta_bn_prop=0.0):
        """
        Predict label probabilities from each topiuc
        """
        theta = Variable(torch.Tensor(theta))
        if C is not None:
            C = Variable(torch.Tensor(C))
        probs = self._model.predict_from_theta(theta, C)
        return probs

    def get_losses(self, X, Y, C, eta_bn_prop=0.0):
        """
        Compute and return the loss values for all instances in X, Y, C
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if C is not None and batch_size == 1:
            C = np.expand_dims(C, axis=0)
        X = Variable(torch.Tensor(X))
        if Y is not None:
            Y = Variable(torch.Tensor(Y))
        if C is not None:
            C = Variable(torch.Tensor(C))
        _, _, Y_recon, losses = self._model(X, Y, C, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
        return losses.data.numpy()

    def compute_theta(self, X, Y, C, eta_bn_prop=0.0):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, C
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if C is not None and batch_size == 1:
            C = np.expand_dims(C, axis=0)
        X = Variable(torch.Tensor(X))
        if Y is not None:
            Y = Variable(torch.Tensor(Y))
        if C is not None:
            C = Variable(torch.Tensor(C))
        theta, _, _, _ = self._model(X, Y, C, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
        return theta.data.numpy()

    def get_weights(self):
        emb = self._model.beta_layer.weight.data.numpy().T
        return emb

    def get_bg(self):
        bg = self._model.beta_layer.bias.data.numpy()
        return bg

    def get_covar_weights(self):
        emb = self._model.beta_c_layer.weight.data.numpy().T
        return emb

    def get_batch_size(self, X):
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


class torchScholar(nn.Module):

    """
    The pyTorch core of Scholar
    """

    def __init__(self, config, alpha, update_embeddings=True, init_emb=None, bg_init=None):
        super(torchScholar, self).__init__()

        vocab_size = config['dv']
        self.dh = config['n_topics']
        self.n_labels = config['n_labels']
        self.n_covariates = config['n_covariates']
        self.words_emb_dim = config['embedding_dim']
        self.label_emb_dim = config['label_emb_dim']
        self.covar_emb_dim = config['covar_emb_dim']
        emb_size = self.words_emb_dim
        self.classifier_layers = config['classifier_layers']

        self.embeddings_x_layer = nn.Linear(vocab_size, self.words_emb_dim, bias=False)
        if self.n_covariates > 0:
            if self.covar_emb_dim > 0:
                self.covar_emb_layer = nn.Linear(self.n_covariates, self.covar_emb_dim)
                emb_size += self.covar_emb_dim
            elif self.covar_emb_dim < 0:
                emb_size += self.n_covariates
        if self.n_labels > 0:
            if self.label_emb_dim > 0:
                self.label_emb_layer = nn.Linear(self.n_labels, self.label_emb_dim)
                emb_size += self.label_emb_dim
            elif self.label_emb_dim < 0:
                emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb))
        else:
            xavier_uniform(self.embeddings_x_layer.weight)

        self.mean_layer = nn.Linear(emb_size, self.dh)
        self.logvar_layer = nn.Linear(emb_size, self.dh)

        self.mean_bn_layer = nn.BatchNorm1d(self.dh, eps=0.001, momentum=0.001)
        self.logvar_bn_layer = nn.BatchNorm1d(self.dh, eps=0.001, momentum=0.001)

        self.z_dropout_layer = nn.Dropout(p=0.2)

        self.beta_layer = nn.Linear(self.dh, vocab_size)
        xavier_uniform(self.beta_layer.weight)
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        if self.n_covariates > 0:
            self.beta_c_layer = nn.Linear(self.n_covariates, vocab_size, bias=False)

        if self.n_labels > 0:
            if self.classifier_layers == 0:
                self.classifier_layer_0 = nn.Linear(self.dh, self.n_labels)
            elif self.classifier_layers == 1:
                self.classifier_layer_0 = nn.Linear(self.dh, self.dh)
                self.classifier_layer_1 = nn.Linear(self.dh, self.n_labels)
            else:
                self.classifier_layer_0 = nn.Linear(self.dh, self.dh)
                self.classifier_layer_1 = nn.Linear(self.dh, self.dh)
                self.classifier_layer_2 = nn.Linear(self.dh, self.n_labels)

        self.eta_bn_layer = nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001)

        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (((1.0 / alpha) * (1 - (2.0 / self.dh))).T + (1.0 / (self.dh * self.dh)) * np.sum(1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.dh))
        prior_var = np.array(prior_var).reshape((1, self.dh))
        self.prior_mean = torch.from_numpy(prior_mean)
        self.prior_var = torch.from_numpy(prior_var)
        self.prior_logvar = self.prior_var.log()

    def forward(self, X, Y, C, compute_loss=True, do_average=True, eta_bn_prop=1.0, var_scale=1.0):

        en0_x = self.embeddings_x_layer(X)
        encoder_parts = [en0_x]

        if self.n_covariates > 0:
            if self.covar_emb_dim > 0:
                c_emb = self.covar_emb_layer(C)
                en0_c = c_emb
                encoder_parts.append(en0_c)
            elif self.covar_emb_dim < 0:
                c_emb = C
                encoder_parts.append(C)
            else:
                c_emb = C

        if self.n_labels > 0:
            if self.label_emb_dim > 0:
                y_emb = self.label_emb_layer(Y)
                en0_y = y_emb
                encoder_parts.append(en0_y)
            elif self.label_emb_dim < 0:
                encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1)
        else:
            en0 = en0_x

        encoder_output = F.softplus(en0)
        encoder_output_do = self.encoder_dropout_layer(encoder_output)

        posterior_mean = self.mean_layer(encoder_output_do)
        posterior_logvar = self.logvar_layer(encoder_output_do)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        posterior_var = posterior_logvar_bn.exp()

        eps = Variable(X.data.new().resize_as_(posterior_mean_bn.data).normal_())

        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_do = self.z_dropout_layer(z)

        theta = F.softmax(z_do, dim=1)

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta)

        # add deviations for covariates (and interactions)
        if self.n_covariates > 0:
            eta = eta + self.beta_c_layer(c_emb)
            # TODO: implement interactions

        eta_bn = self.eta_bn_layer(eta)

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn

        # predict labels
        Y_recon = None
        if self.n_labels > 0:
            if self.n_covariates > 0:
                classifier_input = torch.concat([theta, c_emb], dim=1)
            else:
                classifier_input = theta
            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_2(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        if compute_loss:
            return theta, X_recon, Y_recon, self._loss(X, Y, X_recon, Y_recon, posterior_mean_bn, posterior_logvar_bn, posterior_var, do_average)
        else:
            return theta, X_recon, Y_recon

    def _loss(self, X, Y, X_recon, Y_recon, posterior_mean, posterior_logvar, posterior_var, do_average=True):
        # negative log likelihood
        NL = -(X * (X_recon+1e-10).log()).sum(1)
        if self.n_labels > 0:
            NL += -(Y * (Y_recon+1e-10).log()).sum(1)

        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.dh)

        # loss
        loss = (NL + KLD)

        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if do_average:
            return loss.mean()
        else:
            return loss

    def predict_from_theta(self, theta, C):
        """
        Predict labels from a distribution over topics
        """
        if self.n_covariates > 0:
            if self.covar_emb_dim > 0:
                c_emb = self.covar_emb_layer(C)
            elif self.covar_emb_dim < 0:
                c_emb = C
            else:
                c_emb = C

        Y_recon = None
        if self.n_labels > 0:
            if self.n_covariates > 0:
                classifier_input = torch.concat([theta, c_emb], dim=1)
            else:
                classifier_input = theta
            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_1(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        return Y_recon