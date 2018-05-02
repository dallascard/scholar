import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
slim = tf.contrib.slim


def xavier_init(fan_in, fan_out, constant=1):
    """
    Helper function ot initialize weights
    """
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class Scholar(object):
    """
    Scholar: a neural model for documents with metadata
    """

    def __init__(self, network_architecture, alpha=1.0,
                 learning_rate=0.001, batch_size=100, init_embeddings=None, update_embeddings=True,
                 init_bg=None, update_background=True, init_beta=None, update_beta=True,
                 threads=4, regularize=False, optimizer='adam',
                 adam_beta1=0.99, seed=None):
        """
        :param network_architecture: a dictionary of model configuration parameters (see run_scholar_tf.py)
        :param alpha: hyperparameter for Dirichlet prior on documents (scalar or np.array)
        :param learning_rate:
        :param batch_size: default batch size
        :param init_embeddings: np.array of word vectors to be used in the encoder (optional)
        :param update_embeddings: if False, do not update the word embeddings used in the encoder
        :param init_bg: vector of weights to iniatialize the background term (optional)
        :param update_background: if False, do not update the weights of the background term
        :param init_beta: initial topic-word weights (optional)
        :param update_beta: if False, do not update topic-word weights
        :param threads: limit computation to this many threads (seems to be doubled in practice)
        :param regularize: if True, apply adaptive L2 regularizatoin
        :param optimizer: optimizer to use [adam|sgd|adagrad]
        :param adam_beta1: beta1 parameter for Adam optimizer
        :param seed: random seed (optional)
        """

        if seed is not None:
            tf.set_random_seed(seed)

        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        n_topics = network_architecture['n_topics']
        n_labels = network_architecture['n_labels']
        n_covariates = network_architecture['n_covariates']
        covar_emb_dim = network_architecture['covar_emb_dim']
        use_covar_interactions = network_architecture['use_covar_interactions']
        dv = network_architecture['dv']

        self.regularize = regularize

        # create placeholders for covariates l2 penalties
        self.beta_c_length = 0      # size of embedded covariates
        self.beta_ci_length = 0     # size of embedded covariates * topics
        if n_covariates > 0:
            if covar_emb_dim > 0:
                self.beta_c_length = covar_emb_dim
            else:
                self.beta_c_length = n_covariates
        if use_covar_interactions:
            self.beta_ci_length = self.beta_c_length * n_topics

        self.l2_strengths = tf.placeholder(tf.float32, [n_topics, dv], name="l2_strengths")
        self.l2_strengths_c = tf.placeholder(tf.float32, [self.beta_c_length, dv], name="l2_strengths_c")
        self.l2_strengths_ci = tf.placeholder(tf.float32, [self.beta_ci_length, dv], name="l2_strengths_ci")

        # create placeholders for runtime options
        self.batch_size = tf.placeholder_with_default(batch_size, [], name='batch_size')
        self.var_scale = tf.placeholder_with_default(1.0, [], name='var_scale')        # set to 0 to use posterior mean
        self.bg_scale = tf.placeholder_with_default(1.0, [], name='bg_scale')          # set to 0 to not use background
        self.is_training = tf.placeholder_with_default(True, [], name='is_training')   # placeholder for batchnorm
        self.eta_bn_prop = tf.placeholder_with_default(1.0, [], name='eta_bn_prop')    # used to anneal away from bn
        self.kld_weight = tf.placeholder_with_default(1.0, [], name='kld_weight')      # optional KLD weight param

        self.update_embeddings = update_embeddings
        self.update_background = update_background
        self.update_beta = update_beta
        self.optimizer_type = optimizer

        # create a placeholder for train / test inputs
        self.x = tf.placeholder(tf.float32, [None, dv], name='input')  # batch size x vocab matrix of word counts
        if n_labels > 0:
            self.y = tf.placeholder(tf.float32, [None, n_labels], name='input_y')
        else:
            self.y = tf.placeholder(tf.float32, [], name='input_y')
        if n_covariates > 0:
            self.c = tf.placeholder(tf.float32, [None, n_covariates], name='input_c')
        else:
            self.c = tf.placeholder(tf.float32, [], name='input_c')

        # create a placeholder for dropout strength
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # create placeholders to allow injecting a specific value of hidden variables
        self.theta_input = tf.placeholder(tf.float32, [None, n_topics], name='theta_input')
        # set self.use_theta_input to 1 to override sampled theta and generate from self.theta_input
        self.use_theta_input = tf.placeholder_with_default(0.0, [], name='use_theta_input')

        # create priors on the hidden state
        self.h_dim = (network_architecture["n_topics"])

        # interpret alpha as either a (symmetric) scalar prior or a vector prior
        if np.array(alpha).size == 1:
            self.alpha = alpha * np.ones((1, self.h_dim)).astype(np.float32)
        else:
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.h_dim

        # compute prior mean and variance of Laplace approximation to Dirichlet
        self.prior_mean = tf.constant((np.log(self.alpha).T - np.mean(np.log(self.alpha), 1)).T)
        if self.h_dim > 1:
            self.prior_var = tf.constant((((1.0/self.alpha) * (1 - (2.0/self.h_dim))).T + (1.0/(self.h_dim*self.h_dim)) * np.sum(1.0/self.alpha, 1)).T)
        else:
            self.prior_var = tf.constant(1.0/self.alpha)
        self.prior_logvar = tf.log(self.prior_var)

        # create the network
        self._create_network()
        with tf.name_scope('loss'):
            self._create_loss_optimizer()

        init = tf.global_variables_initializer()

        # create a session
        config = tf.ConfigProto(intra_op_parallelism_threads=threads,
                                inter_op_parallelism_threads=threads)
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(init)

        # initialize background
        if init_bg is not None:
            self.sess.run(self.network_weights['background'].assign(init_bg))

        # initialize topic-word weights
        if init_beta is not None:
            self.sess.run(self.network_weights['beta'].assign(init_beta))

        # initialize word embeddings
        if init_embeddings is not None:
            self.sess.run(self.network_weights['embeddings'].assign(init_embeddings))

    def _create_network(self):
        encoder_layers = self.network_architecture['encoder_layers']
        dh = self.network_architecture['n_topics']
        n_labels = self.network_architecture['n_labels']
        n_covariates = self.network_architecture['n_covariates']
        words_emb_dim = self.network_architecture['embedding_dim']
        label_emb_dim = self.network_architecture['label_emb_dim']
        covar_emb_dim = self.network_architecture['covar_emb_dim']
        emb_size = words_emb_dim
        use_covar_interactions = self.network_architecture['use_covar_interactions']
        classifier_layers = self.network_architecture['classifier_layers']

        self.network_weights = self._initialize_weights()

        # create the first layer of the encoder
        encoder_parts = []
        # convert word indices to embeddings
        en0_x = tf.matmul(self.x, self.network_weights['embeddings'])
        encoder_parts.append(en0_x)

        # add label if we have them
        if n_labels > 0:
            if label_emb_dim > 0:
                # use the label embedding if we're projecting them down
                y_emb = tf.matmul(self.y, self.network_weights['label_embeddings'])
                en0_y = y_emb
                emb_size += int(label_emb_dim)
                encoder_parts.append(en0_y)
            elif label_emb_dim < 0:
                # if label_emb_dim < 0 (default), just feed in label vectors as is
                emb_size += n_labels
                encoder_parts.append(self.y)

        # do the same for covariates
        if n_covariates > 0:
            if covar_emb_dim > 0:
                c_emb = tf.matmul(self.c, self.network_weights['covariate_embeddings'])
                en0_c = c_emb
                emb_size += covar_emb_dim
                encoder_parts.append(en0_c)
            elif covar_emb_dim < 0:
                # if covar_emb_dim < 0 (default), just feed in covariate vectors as is
                c_emb = self.c
                emb_size += n_covariates
                encoder_parts.append(c_emb)
            else:
                # if covar_emb_dim == 0, do not give the covariate vectors to the encoder
                c_emb = self.c

        # combine everything to produce the output of layer 0
        if len(encoder_parts) > 1:
            en0 = tf.concat(encoder_parts, axis=1)
        else:
            en0 = en0_x

        # optionally add more encoder layers
        if encoder_layers == 0:
            # technically this will involve two layers, but they're both linear, so it's basically the same as one
            encoder_output = en0
        elif encoder_layers == 1:
            encoder_output = tf.nn.softplus(en0, name='softplus0')
        else:
            en0_softmax = tf.nn.softplus(en0, name='softplus0')
            en1 = slim.layers.linear(en0_softmax, emb_size, scope='en1')
            encoder_output = tf.nn.softplus(en1, name='softplus1')

        # optionally add an encoder shortcut
        if self.network_architecture['encoder_shortcut']:
            encoder_output = tf.add(encoder_output, slim.layers.linear(self.x, emb_size))

        # apply dropout to encoder output
        encoder_output_do = slim.layers.dropout(encoder_output, self.keep_prob, scope='en_dropped')

        # apply linear transformations to encoder output for mean and log of diagonal of covariance matrix
        self.posterior_mean = slim.layers.linear(encoder_output_do, dh, scope='FC_mean')
        self.posterior_logvar = slim.layers.linear(encoder_output_do, dh, scope='FC_logvar')

        # apply batchnorm to these vectors
        self.posterior_mean_bn = slim.layers.batch_norm(self.posterior_mean, scope='BN_mean', is_training=self.is_training)
        self.posterior_logvar_bn = slim.layers.batch_norm(self.posterior_logvar, scope='BN_logvar', is_training=self.is_training)

        with tf.name_scope('h_scope'):
            # sample from symmetric Gaussian noise
            eps = tf.random_normal((self.batch_size, dh), 0, 1, dtype=tf.float32)
            # use the reparameterization trick to get a sample from the latent variable posterior
            self.z = tf.add(self.posterior_mean_bn, tf.multiply(self.var_scale, tf.multiply(tf.sqrt(tf.exp(self.posterior_logvar_bn)), eps)))
            self.posterior_var = tf.exp(self.posterior_logvar_bn)

        # apply dropout to the (unnormalized) latent representation
        z_do = slim.layers.dropout(self.z, self.keep_prob, scope='p_dropped')

        # transform z to the simplex using a softmax
        theta_sample = slim.layers.softmax(z_do)

        # use manually-set generator output for generation; during training use_theta_input should equal 0
        self.theta = tf.add(tf.multiply((1.0 - self.use_theta_input), theta_sample), tf.multiply(self.use_theta_input, self.theta_input))

        # combine latent representation with topics and background
        eta = tf.add(tf.matmul(self.theta, self.network_weights['beta']), tf.multiply(self.bg_scale, self.network_weights['background']))

        # add deviations for covariates (and interactions)
        if n_covariates > 0:
            eta = tf.add(eta, tf.matmul(c_emb, self.network_weights['beta_c']))
            if use_covar_interactions:
                gen_output_rsh = tf.reshape(self.theta, [self.batch_size, dh, 1])
                c_emb_rsh = array_ops.reshape(c_emb, [self.batch_size, 1, self.beta_c_length])
                covar_interactions = tf.reshape(gen_output_rsh * c_emb_rsh, [self.batch_size, self.beta_ci_length])
                eta = tf.add(eta, tf.matmul(covar_interactions, self.network_weights['beta_ci']))

        # add batchnorm to eta
        eta_bn = slim.layers.batch_norm(eta, scope='BN_decoder', is_training=self.is_training)

        # reconstruct both with and without batchnorm on eta
        self.x_recon = tf.nn.softmax(eta_bn)
        self.x_recon_no_bn = tf.nn.softmax(eta)

        # predict labels using theta and (optionally) covariates
        if n_labels > 0:
            if n_covariates > 0 and self.network_architecture['covars_in_classifier']:
                classifier_input = tf.concat([self.theta, c_emb], axis=1)
            else:
                classifier_input = self.theta
            if classifier_layers == 0:
                decoded_y = slim.layers.linear(classifier_input, n_labels, scope='y_decoder')
            elif classifier_layers == 1:
                cls0 = slim.layers.linear(classifier_input, dh, scope='cls0')
                cls0_sp = tf.nn.softplus(cls0, name='cls0_softplus')
                decoded_y = slim.layers.linear(cls0_sp, n_labels, scope='y_decoder')
            else:
                cls0 = slim.layers.linear(classifier_input, dh, scope='cls0')
                cls0_sp = tf.nn.softplus(cls0, name='cls0_softplus')
                cls1 = slim.layers.linear(cls0_sp, dh, scope='cls1')
                cls1_sp = tf.nn.softplus(cls1, name='cls1_softplus')
                decoded_y = slim.layers.linear(cls1_sp, n_labels, scope='y_decoder')
            self.y_recon = tf.nn.softmax(decoded_y, name='y_recon')
            self.pred_y = tf.argmax(self.y_recon, axis=1, name='pred_y')

    def _initialize_weights(self):
        all_weights = dict()

        dh = self.network_architecture['n_topics']
        dv = self.network_architecture['dv']
        embedding_dim = self.network_architecture['embedding_dim']
        n_labels = self.network_architecture['n_labels']
        label_emb_dim = self.network_architecture['label_emb_dim']
        n_covariates = self.network_architecture['n_covariates']
        covar_emb_dim = self.network_architecture['covar_emb_dim']

        # background log-frequency of terms (overwrite with pre-specified initialization later))
        all_weights['background'] = tf.Variable(tf.zeros(dv, dtype=tf.float32), trainable=self.update_background)

        # initial layer of word embeddings (overwrite with pre-specified initialization later))
        all_weights['embeddings'] = tf.Variable(xavier_init(dv, embedding_dim), trainable=self.update_embeddings)

        # topic deviations (overwrite with pre-specified initialization later))
        all_weights['beta'] = tf.Variable(xavier_init(dh, dv), trainable=self.update_beta)

        # create embeddings for labels
        if n_labels > 0:
            if label_emb_dim > 0:
                all_weights['label_embeddings'] = tf.Variable(xavier_init(n_labels, label_emb_dim), trainable=True)

        if n_covariates > 0:
            if covar_emb_dim > 0:
                all_weights['covariate_embeddings'] = tf.Variable(xavier_init(n_covariates, covar_emb_dim), trainable=True)

        all_weights['beta_c'] = tf.Variable(xavier_init(self.beta_c_length, dv))
        all_weights['beta_ci'] = tf.Variable(xavier_init(self.beta_ci_length, dv))

        return all_weights

    def _create_loss_optimizer(self):

        # Compute an interpolation between reconstruction with and without batchnorm on eta.
        # This is done to allow annealing away from using batchnorm on eta over the course of training
        x_recon = tf.add(tf.add(tf.multiply(self.eta_bn_prop, self.x_recon), tf.multiply((1.0 - self.eta_bn_prop), self.x_recon_no_bn)), 1e-10)

        # compute the negative log loss
        self.NL_x = -tf.reduce_sum(self.x * tf.log(x_recon), 1)

        if self.network_architecture['n_labels'] > 0:
            # loss for categortical labels
            # TODO: add losses for other types of labels
            NL_y = -tf.reduce_sum(self.y * tf.log(self.y_recon+1e-10), 1)  # test

            self.classifier_loss = tf.reduce_mean(NL_y)

            self.NL = tf.add(self.NL_x, NL_y)
        else:
            self.NL = self.NL_x

        # compute terms for the KL divergence between prior and variational posterior
        var_division = self.posterior_var / self.prior_var
        diff = self.posterior_mean_bn - self.prior_mean
        diff_term = diff * diff / self.prior_var
        logvar_division = self.prior_logvar - self.posterior_logvar_bn

        self.KLD = 0.5 * (tf.reduce_sum(var_division + diff_term + logvar_division, 1) - self.h_dim)

        self.losses = tf.add(self.NL, tf.multiply(self.kld_weight, self.KLD))
        self.loss = tf.reduce_mean(self.losses)

        # add in regularization terms
        if self.regularize:
            self.loss = tf.add(self.loss, tf.reduce_sum(tf.multiply(self.l2_strengths, tf.square(self.network_weights['beta']))))
            if self.network_architecture['n_covariates']:
                self.loss = tf.add(self.loss, tf.reduce_sum(tf.multiply(self.l2_strengths_c, tf.square(self.network_weights['beta_c']))))
                if self.network_architecture['use_covar_interactions']:
                    self.loss = tf.add(self.loss, tf.reduce_sum(tf.multiply(self.l2_strengths_ci, tf.square(self.network_weights['beta_ci']))))

        # explicitly add batchnorm terms to parameters to be updated so as to save the global means
        update_ops = []
        update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='BN_mean'))
        update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='BN_logvar'))
        update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='BN_decoder'))

        # choose an optimizer
        with tf.control_dependencies(update_ops):
            if self.optimizer_type == 'adam':
                print("Using Adam")
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.adam_beta1).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                print("Using adagrad")
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            else:
                print("Using SGD")
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def fit(self, X, Y, C, l2_strengths, l2_strengths_c, l2_strengths_ci, eta_bn_prop=1.0, kld_weight=1.0, keep_prob=0.8):
        """
        Fit the model to data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param C: np.array of covariates [batch size x n_covariates]
        :param l2_strengths: np.array of l2 weights on beta (updated in run_scholar_tf.py)
        :param l2_strengths_c: np.array of l2 weights on beta_c (updated in run_scholar_tf.py)
        :param l2_strengths_ci: np.array of l2 weights on beta_ci (updated in run_scholar_tf.py)
        :param eta_bn_prop: in [0, 1] controlling the interpolation between using batch norm on the final layer and not
        :param kld_weight: weighting factor for KLD term (default=1.0)
        :param keep_prob: probability of not zeroing a weight in dropout
        :return: overall loss for minibatch; loss from the classifier; per-instance predictions
        """
        batch_size = self.get_batch_size(X)
        theta_input = np.zeros([batch_size, self.network_architecture['n_topics']]).astype('float32')
        if Y is not None:
            opt, loss, classifier_loss, pred = self.sess.run((self.optimizer, self.loss, self.classifier_loss, self.pred_y), feed_dict={self.x: X, self.y: Y, self.c: C, self.keep_prob: .8, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.eta_bn_prop: eta_bn_prop, self.kld_weight: kld_weight, self.theta_input: theta_input})
        else:
            opt, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: X, self.y: Y, self.c: C, self.keep_prob: keep_prob, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.eta_bn_prop: eta_bn_prop, self.kld_weight: kld_weight, self.theta_input: theta_input})
            classifier_loss = 0
            pred = -1
        return loss, classifier_loss, pred

    def predict(self, X, C, eta_bn_prop=0.0):
        """
        Predict document representations (theta) and labels (Y) given input (X) and covariates (C)
        """
        # set all regularization strenghts to be zero, since we don't care about topic reconstruction here
        l2_strengths = np.zeros(self.network_weights['beta'].shape)
        l2_strengths_c = np.zeros(self.network_weights['beta_c'].shape)
        l2_strengths_ci = np.zeros(self.network_weights['beta_ci'].shape)
        # input a vector of all zeros in place of the labels that the model has been trained on
        Y = np.zeros((1, self.network_architecture['n_labels'])).astype('float32')
        batch_size = self.get_batch_size(X)
        theta_input = np.zeros([batch_size, self.network_architecture['n_topics']]).astype('float32')

        theta, pred = self.sess.run((self.theta, self.y_recon), feed_dict={self.x: X, self.y: Y, self.c: C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.batch_size: 1, self.var_scale: 0.0, self.is_training: False, self.theta_input: theta_input, self.eta_bn_prop: eta_bn_prop})
        return theta, pred

    def predict_from_topics(self, theta, C=None):
        """
        Predict the probability of labels given a distribution over topics (theta), and covariates (C)
        """
        l2_strengths = np.zeros(self.network_weights['beta'].shape)
        l2_strengths_c = np.zeros(self.network_weights['beta_c'].shape)
        l2_strengths_ci = np.zeros(self.network_weights['beta_ci'].shape)
        X = np.zeros([1, self.network_architecture['dv']]).astype('float32')
        Y = np.zeros([1, self.network_architecture['n_labels']]).astype('float32')
        probs = self.sess.run(self.y_recon, feed_dict={self.x: X, self.y: Y, self.c: C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.batch_size: 1, self.is_training: False, self.theta_input: theta, self.use_theta_input: 1.0})
        return probs

    def get_losses(self, X, Y, C, eta_bn_prop=0.0):
        """
        Compute and return the loss values for all instances in X, Y, C
        """
        l2_strengths = np.zeros(self.network_weights['beta'].shape)
        l2_strengths_c = np.zeros(self.network_weights['beta_c'].shape)
        l2_strengths_ci = np.zeros(self.network_weights['beta_ci'].shape)
        # make inputs 2-dimensional
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if C is not None and batch_size == 1:
            C = np.expand_dims(C, axis=0)
        theta_input = np.zeros([batch_size, self.network_architecture['n_topics']]).astype('float32')
        losses = self.sess.run(self.losses, feed_dict={self.x: X, self.y: Y, self.c: C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.batch_size: batch_size, self.var_scale: 0.0, self.is_training: False, self.theta_input: theta_input, self.eta_bn_prop: eta_bn_prop})
        return losses

    def compute_theta(self, X, Y, C):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, C
        """
        l2_strengths = np.zeros(self.network_weights['beta'].shape)
        l2_strengths_c = np.zeros(self.network_weights['beta_c'].shape)
        l2_strengths_ci = np.zeros(self.network_weights['beta_ci'].shape)

        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if C is not None and batch_size == 1:
            C = np.expand_dims(C, axis=0)
        theta_input = np.zeros([batch_size, self.network_architecture['n_topics']]).astype('float32')

        theta = self.sess.run(self.theta, feed_dict={self.x: X, self.y: Y, self.c: C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.batch_size: batch_size, self.is_training: False, self.theta_input: theta_input})
        return theta

    def get_weights(self):
        """
        Return the current values of the topic-vocabulary weights
        """
        decoder_weight = self.network_weights['beta']
        emb = self.sess.run(decoder_weight)
        return emb

    def get_bg(self):
        """
        Return the current values of the background term
        """
        decoder_weight = self.network_weights['background']
        bg = self.sess.run(decoder_weight)
        return bg

    def get_covar_weights(self):
        """
        Return the current values of the per-covariate vocabulary deviations
        """
        decoder_weight = self.network_weights['beta_c']
        emb = self.sess.run(decoder_weight)
        return emb

    def get_covar_inter_weights(self):
        """
        Return the current values of the interactions terms between topics and covariates
        """
        decoder_weight = self.network_weights['beta_ci']
        emb = self.sess.run(decoder_weight)
        return emb

    def get_label_embeddings(self):
        """
        Return the embeddings of labels used by the encoder
        """
        param = self.network_weights['label_embeddings']
        emb = self.sess.run(param)
        return emb

    def get_covar_embeddings(self):
        """
        Return the embeddings of covariates used by the encoder and decoder
        """
        param = self.network_weights['covariate_embeddings']
        emb = self.sess.run(param)
        return emb

    def get_batch_size(self, X):
        """
        Determine the number of instances in a given minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size