import os
import sys
from optparse import OptionParser

import gensim
import numpy as np
import pandas as pd

import file_handling as fh
from scholar_pytorch import Scholar


def main():
    usage = "%prog input_dir train_prefix"
    parser = OptionParser(usage=usage)
    parser.add_option('-a', dest='alpha', default=1.0,
                      help='Hyperparameter for logistic normal prior: default=%default')
    parser.add_option('-k', dest='n_topics', default=20,
                      help='Size of latent representation (~num topics): default=%default')
    parser.add_option('-b', dest='batch_size', default=200,
                      help='Size of minibatches: default=%default')
    parser.add_option('-l', dest='learning_rate', default=0.002,
                      help='Initial learning rate: default=%default')
    parser.add_option('-m', dest='momentum', default=0.99,
                      help='beta1 for Adam: default=%default')
    parser.add_option('-e', dest='epochs', default=200,
                      help='Number of epochs: default=%default')
    parser.add_option('--emb_dim', dest='embedding_dim', default=300,
                      help='Dimension of input embeddings: default=%default')
    parser.add_option('--labels', dest='label_name', default=None,
                      help='Read labels from input_dir/[train|test]_prefix.label_name.csv: default=%default')
    parser.add_option('--covars', dest='covar_names', default=None,
                      help='Read covars from files with these names (comma-separated): default=%default')
    parser.add_option('--label_emb_dim', dest='label_emb_dim', default=-1,
                      help='Class embedding dimension [0 = identity]: default=%default')
    parser.add_option('--covar_emb_dim', dest='covar_emb_dim', default=-1,
                      help='Covariate embedding dimension [0 = identity]: default=%default')
    parser.add_option('--min_covar_count', dest='min_covar_count', default=None,
                      help='Drop binary covariates that occur less than this in training: default=%default')
    parser.add_option('--c_layers', dest='classifier_layers', default=1,
                      help='Number of layers in (generative) classifier [0|1|2]: default=%default')
    parser.add_option('-t', dest='test_prefix', default=None,
                      help='Prefix of test set: default=%default')
    parser.add_option('-o', dest='output_dir', default='output',
                      help='Output directory: default=%default')
    parser.add_option('--w2v', dest='word2vec_file', default=None,
                      help='Use this word2vec .bin file to initialize and fix embeddings: default=%default')
    parser.add_option('--update_bg', action="store_true", dest="update_bg", default=False,
                      help='Update background parameters: default=%default')
    parser.add_option('--no_bg', action="store_true", dest="no_bg", default=False,
                      help='Do not use background freq: default=%default')
    parser.add_option('--no_bn_anneal', action="store_true", dest="no_bn_anneal", default=False,
                      help='Do not anneal away from batchnorm: default=%default')
    parser.add_option('--dev_folds', dest='dev_folds', default=0,
                      help='Number of dev folds: default=%default')
    parser.add_option('--dev_fold', dest='dev_fold', default=0,
                      help='Fold to use as dev (if dev_folds > 0): default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]
    train_prefix = args[1]

    alpha = float(options.alpha)
    n_topics = int(options.n_topics)
    batch_size = int(options.batch_size)
    learning_rate = float(options.learning_rate)
    adam_beta1 = float(options.momentum)
    n_epochs = int(options.epochs)
    embedding_dim = int(options.embedding_dim)
    label_file_name = options.label_name
    covar_file_names = options.covar_names
    label_emb_dim = int(options.label_emb_dim)
    covar_emb_dim = int(options.covar_emb_dim)
    min_covar_count = options.min_covar_count
    classifier_layers = int(options.classifier_layers)
    test_prefix = options.test_prefix
    output_dir = options.output_dir
    word2vec_file = options.word2vec_file
    update_background = options.update_bg
    no_bg = options.no_bg
    bn_anneal = not options.no_bn_anneal
    dev_folds = int(options.dev_folds)
    dev_fold = int(options.dev_fold)
    rng = np.random.RandomState(np.random.randint(0, 100000))

    # load the training data
    train_X, vocab, train_labels, label_names, label_type, train_covariates, covariate_names, covariates_type = load_data(input_dir, train_prefix, label_file_name, covar_file_names)
    n_train, dv = train_X.shape

    if train_labels is not None:
        _, n_labels = train_labels.shape
    else:
        n_labels = 0

    if train_covariates is not None:
        _, n_covariates = train_covariates.shape
        if min_covar_count is not None and int(min_covar_count) > 0:
            print("Removing rare covariates")
            covar_sums = train_covariates.sum(axis=0).reshape((n_covariates, ))
            covariate_selector = covar_sums > int(min_covar_count)
            train_covariates = train_covariates[:, covariate_selector]
            covariate_names = [name for i, name in enumerate(covariate_names) if covariate_selector[i]]
            n_covariates = len(covariate_names)
    else:
        n_covariates = 0

    # split into train and dev
    if dev_folds > 0:
        n_dev = int(n_train / dev_folds)
        indices = np.array(range(n_train), dtype=int)
        rng.shuffle(indices)
        if dev_fold < dev_folds - 1:
            dev_indices = indices[n_dev * dev_fold: n_dev * (dev_fold +1)]
        else:
            dev_indices = indices[n_dev * dev_fold:]
        train_indices = list(set(indices) - set(dev_indices))
        dev_X = train_X[dev_indices, :]
        train_X = train_X[train_indices, :]
        if train_labels is not None:
            dev_labels = train_labels[dev_indices, :]
            train_labels = train_labels[train_indices, :]
        else:
            dev_labels = None
        if train_covariates is not None:
            dev_covariates = train_covariates[dev_indices, :]
            train_covariates = train_covariates[train_indices, :]
        else:
            dev_covariates = None
        n_train = len(train_indices)
    else:
        dev_X = None
        dev_labels = None
        dev_covariates = None
        n_dev = 0

    # load the test data
    if test_prefix is not None:
        test_X, _, test_labels, _, _, test_covariates, _, _ = load_data(input_dir, test_prefix, label_file_name, covar_file_names, vocab=vocab)
        n_test, _ = test_X.shape
        if test_labels is not None:
            _, n_labels_test = test_labels.shape
            assert n_labels_test == n_labels
        if test_covariates is not None:
            if min_covar_count is not None and int(min_covar_count) > 0:
                test_covariates = test_covariates[:, covariate_selector]
            _, n_covariates_test = test_covariates.shape
            assert n_covariates_test == n_covariates

    else:
        test_X = None
        n_test = 0
        test_labels = None
        test_covariates = None

    # initialize the background using overall word frequencies
    init_bg = get_init_bg(train_X)
    if no_bg:
        init_bg = np.zeros_like(init_bg)

    # combine the network configuration parameters into a dictionary
    network_architecture = make_network(dv, embedding_dim, n_topics, label_type, n_labels, label_emb_dim,
                                        covariates_type, n_covariates, covar_emb_dim, classifier_layers)  # make_network()

    print("Network architecture:")
    for key, val in network_architecture.items():
        print(key + ':', val)

    # load pretrained word vectors
    if word2vec_file is not None:
        vocab_size = len(
            vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        embeddings = np.array(rng.rand(vocab_size, 300) * 0.25 - 0.5, dtype=np.float32)
        count = 0
        print("Loading word vectors")
        pretrained = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)

        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[index, :] = pretrained[word]

        print("Found embeddings for %d words" % count)
        update_embeddings = False
    else:
        embeddings = None
        update_embeddings = True

    # create the model
    model = Scholar(network_architecture, alpha=alpha, learning_rate=learning_rate, init_embeddings=embeddings, update_embeddings=update_embeddings, init_bg=init_bg, update_background=update_background, adam_beta1=adam_beta1)

    # train the model
    print("Optimizing full model")
    model = train(model, network_architecture, train_X, train_labels, train_covariates, training_epochs=n_epochs, batch_size=batch_size, rng=rng, X_dev=dev_X, Y_dev=dev_labels, C_dev=dev_covariates, bn_anneal=bn_anneal)

    # make output directory
    fh.makedirs(output_dir)

    # print background
    bg = model.get_bg()
    if not no_bg:
        print_top_bg(bg, vocab)

    # print topics
    emb = model.get_weights()
    print("Topics:")
    maw, sparsity = print_top_words(emb, vocab)
    print("sparsity in topics = %0.4f" % sparsity)
    save_weights(output_dir, emb, bg, vocab, sparsity_threshold=1e-5)

    fh.write_list_to_text(['{:.4f}'.format(maw)], os.path.join(output_dir, 'maw.txt'))
    fh.write_list_to_text(['{:.4f}'.format(sparsity)], os.path.join(output_dir, 'sparsity.txt'))

    if n_covariates > 0:
        beta_c = model.get_covar_weights()
        print("Covariate deviations:")
        if covar_emb_dim == 0:
            maw, sparsity = print_top_words(beta_c, vocab, covariate_names)
        else:
            maw, sparsity = print_top_words(beta_c, vocab)
        print("sparsity in covariates = %0.4f" % sparsity)
        if output_dir is not None:
            np.savez(os.path.join(output_dir, 'beta_c.npz'), beta=beta_c, names=covariate_names)

    # Evaluate perplexity on dev and test dataa
    if dev_X is not None:
        perplexity = evaluate_perplexity(model, dev_X, dev_labels, dev_covariates, eta_bn_prop=0.0)
        print("Dev perplexity = %0.4f" % perplexity)
        fh.write_list_to_text([str(perplexity)], os.path.join(output_dir, 'perplexity.dev.txt'))

    if test_X is not None:
        perplexity = evaluate_perplexity(model, test_X, test_labels, test_covariates, eta_bn_prop=0.0)
        print("Test perplexity = %0.4f" % perplexity)
        fh.write_list_to_text([str(perplexity)], os.path.join(output_dir, 'perplexity.test.txt'))

    # evaluate accuracy on predicting categorical covariates
    if n_covariates > 0 and covariates_type == 'categorical':
        print("Predicting categorical covariates")
        predictions = infer_categorical_covariate(model, network_architecture, train_X, train_labels)
        accuracy = float(np.sum(predictions == np.argmax(train_covariates, axis=1)) / float(len(train_covariates)))
        print("Train accuracy on covariates = %0.4f" % accuracy)
        if output_dir is not None:
            fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.train.txt'))

        if dev_X is not None:
            predictions = infer_categorical_covariate(model, network_architecture, dev_X, dev_labels)
            accuracy = float(np.sum(predictions == np.argmax(dev_covariates, axis=1)) / float(len(dev_covariates)))
            print("Dev accuracy on covariates = %0.4f" % accuracy)
            if output_dir is not None:
                fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.dev.txt'))

        if test_X is not None:
            predictions = infer_categorical_covariate(model, network_architecture, test_X, test_labels)
            accuracy = float(np.sum(predictions == np.argmax(test_covariates, axis=1)) / float(len(test_covariates)))
            print("Test accuracy on covariates = %0.4f" % accuracy)
            if output_dir is not None:
                fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.test.txt'))

    # evaluate accuracy on predicting labels
    if n_labels > 0:
        print("Predicting labels")
        predict_labels_and_evaluate(model, train_X, train_labels, train_covariates, output_dir, subset='train')

        if dev_X is not None:
            predict_labels_and_evaluate(model, dev_X, dev_labels, dev_covariates, output_dir, subset='dev')

        if test_X is not None:
            predict_labels_and_evaluate(model, test_X, test_labels, test_covariates, output_dir, subset='test')

    # Print associations between topics and labels
    if n_labels > 0 and n_labels < 7:
        print("Label probabilities based on topics")
        print("Labels:", ' '.join([name for name in label_names]))
        for k in range(n_topics):
            Z = np.zeros([1, n_topics]).astype('float32')
            Z[0, k] = 1.0
            Y = None
            if n_covariates > 0:
                C = np.zeros([1, n_covariates]).astype('float32')
            else:
                C = None
            probs = model.predict_from_topics(Z, C)
            output = str(k) + ': '
            for i in range(n_labels):
                output += '%.4f ' % probs[0, i]
            print(output)

        if n_covariates > 0:
            all_probs = np.zeros([n_covariates, n_topics])
            for k in range(n_topics):
                Z = np.zeros([1, n_topics]).astype('float32')
                Z[0, k] = 1.0
                Y = None
                for c in range(n_covariates):
                    C = np.zeros([1, n_covariates]).astype('float32')
                    C[0, c] = 1.0
                    probs = model.predict_from_topics(Z, C)
                    all_probs[c, k] = probs[0, 0]
            np.savez(os.path.join(output_dir, 'covar_topic_probs.npz'), probs=all_probs)

    # save document representations
    print("Getting topic proportions")
    theta = model.compute_theta(train_X, train_labels, train_covariates)
    print("Saving topic proportions")
    np.savez(os.path.join(output_dir, 'theta.train.npz'), theta=theta)

    if dev_X is not None:
        dev_Y = np.zeros_like(dev_labels)
        print("Getting topic proportions for dev data")
        theta = model.compute_theta(dev_X, dev_Y, dev_covariates)
        print("Saving topic proportions")
        np.savez(os.path.join(output_dir, 'theta.dev.npz'), theta=theta)

    if n_test > 0:
        test_Y = np.zeros_like(test_labels)
        print("Getting topic proportions for test data")
        theta = model.compute_theta(test_X, test_Y, test_covariates)
        print("Saving topic proportions")
        np.savez(os.path.join(output_dir, 'theta.test.npz'), theta=theta)


def load_data(input_dir, input_prefix, label_file_name=None, covar_file_names=None, vocab=None):
    print("Loading data")
    temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    X = np.array(temp, dtype='float32')
    if vocab is None:
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
    n_items, vocab_size = X.shape
    assert vocab_size == len(vocab)
    print("Loaded %d documents with %d features" % (n_items, vocab_size))

    # filter out empty documents
    non_empty_sel = X.sum(axis=1) > 0
    print("Found %d non-empty documents" % np.sum(non_empty_sel))
    X = X[non_empty_sel, :]
    n_items, vocab_size = X.shape

    if label_file_name is not None:
        label_file = os.path.join(input_dir, input_prefix + '.' + label_file_name + '.csv')
        if os.path.exists(label_file):
            print("Loading labels from", label_file)
            temp = pd.read_csv(label_file, header=0, index_col=0)
            label_names = temp.columns
            labels = np.array(temp.values)
            labels = labels[non_empty_sel, :]
            n, n_labels = labels.shape
            assert n == n_items
            print("%d labels" % n_labels)
        else:
            print("Label file not found:", label_file)
            sys.exit()
        if (np.sum(labels, axis=1) == 1).all() and (np.sum(labels == 0) + np.sum(labels == 1) == labels.size):
            label_type = 'categorical'
        elif np.sum(labels == 0) + np.sum(labels == 1) == labels.size:
            label_type = 'bernoulli'
        else:
            label_type = 'real'
        print("Found labels of type %s" % label_type)

    else:
        labels = None
        label_names = None
        label_type = None

    if covar_file_names is not None:
        covariate_list = []
        covariate_names_list = []
        covar_file_names = covar_file_names.split(',')
        for covar_file_name in covar_file_names:
            covariates_file = os.path.join(input_dir, input_prefix + '.' + covar_file_name + '.csv')
            if os.path.exists(covariates_file):
                print("Loading covariates from", covariates_file)
                temp = pd.read_csv(covariates_file, header=0, index_col=0)
                covariate_names = temp.columns
                covariates = np.array(temp.values, dtype=np.float32)
                covariates = covariates[non_empty_sel, :]
                n, n_covariates = covariates.shape
                assert n == n_items
                covariate_list.append(covariates)
                covariate_names_list.extend(covariate_names)
            else:
                print("Covariates file not found:", covariates_file)
                sys.exit()
        covariates = np.hstack(covariate_list)
        covariate_names = covariate_names_list
        n, n_covariates = covariates.shape

        if (np.sum(covariates, axis=1) == 1).all() and (np.sum(covariates == 0) + np.sum(covariates == 1) == covariates.size):
            covariates_type = 'categorical'
        else:
            covariates_type = 'other'

        print("Found covariates of type %s" % covariates_type)

        assert n == n_items
        print("%d covariates" % n_covariates)
    else:
        covariates = None
        covariate_names = None
        covariates_type = None

    counts_sum = X.sum(axis=0)
    order = list(np.argsort(counts_sum).tolist())
    order.reverse()
    print("Most common words: ", ' '.join([vocab[i] for i in order[:10]]))

    return X, vocab, labels, label_names, label_type, covariates, covariate_names, covariates_type


def get_init_bg(data):
    """
    Compute the log background frequency of all words
    """
    sums = np.sum(data, axis=0)+1
    print("Computing background frequencies")
    print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg


def create_minibatch(X, Y, C, batch_size=200, rng=None):
    """
    Split data into minibatches
    """
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            ixs = rng.randint(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)
        if Y is not None and C is not None:
            yield X[ixs, :].astype('float32'), Y[ixs, :].astype('float32'), C[ixs, :].astype('float32')
        elif Y is not None:
            yield X[ixs, :].astype('float32'), Y[ixs, :].astype('float32'), None
        elif C is not None:
            yield X[ixs, :].astype('float32'), None, C[ixs, :].astype('float32')
        else:
            yield X[ixs, :].astype('float32'), None, None


def make_network(dv, embedding_dim=300, n_topics=50, label_type=None, n_labels=0, label_emb_dim=0, covariate_type=None, n_covariates=0, covar_emb_dim=0, classifier_layers=1):
    """
    Assemble the network configuration parameters into a dictionary
    """
    network_architecture = \
        dict(embedding_dim=embedding_dim,
             n_topics=n_topics,
             dv=dv,
             label_type=label_type,
             n_labels=n_labels,
             label_emb_dim=label_emb_dim,
             covariate_type=covariate_type,
             n_covariates=n_covariates,
             covar_emb_dim=covar_emb_dim,
             classifier_layers=classifier_layers
             )
    return network_architecture


def train(model, network_architecture, X, Y, C, batch_size=200, training_epochs=100, display_step=5, X_dev=None, Y_dev=None, C_dev=None, bn_anneal=True, init_eta_bn_prop=1.0, rng=None):
    """
    Train the model
    """
    n_train, dv = X.shape
    mb_gen = create_minibatch(X, Y, C, batch_size=batch_size, rng=rng)
    total_batch = int(n_train / batch_size)
    batches = 0
    eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_cls_cost = 0.
        accuracy = 0.
        mse = 0.
        # Loop over all batches
        for i in range(total_batch):
            # get a minibatch
            batch_xs, batch_ys, batch_cs = next(mb_gen)
            # do one minibatch update
            cost, recon_y = model.fit(batch_xs, batch_ys, batch_cs, eta_bn_prop=eta_bn_prop)
            # compute accuracy on minibatch
            if network_architecture['n_labels'] > 0:
                accuracy += np.sum(np.argmax(recon_y, axis=1) == np.argmax(batch_ys, axis=1)) / float(n_train)

            # Compute average loss
            avg_cost += float(cost) / n_train * batch_size
            batches += 1
            if np.isnan(avg_cost):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                sys.exit()

        # Display logs per epoch step
        if epoch % display_step == 0 and epoch > 0:
            if network_architecture['n_labels'] > 0:
                print("Epoch:", '%d' % epoch, "; cost =", "{:.9f}".format(avg_cost), "; clscost =", "{:.9f}".format(avg_cls_cost), "; training accuracy (noisy) =", "{:.9f}".format(accuracy))
            else:
                print("Epoch:", '%d' % epoch, "cost=", "{:.9f}".format(avg_cost))

            if X_dev is not None:
                # switch to eval mode for intermediate evaluation
                model.eval()
                dev_perplexity = evaluate_perplexity(model, X_dev, Y_dev, C_dev, eta_bn_prop=eta_bn_prop)
                n_dev, _ = X_dev.shape
                if network_architecture['n_labels'] > 0:
                    dev_predictions = predict_labels(model, X_dev, C_dev, eta_bn_prop=eta_bn_prop)
                    dev_accuracy = float(np.sum(dev_predictions == np.argmax(Y_dev, axis=1))) / float(n_dev)
                    print("Epoch: %d; Dev perplexity = %0.4f; Dev accuracy = %0.4f" % (epoch, dev_perplexity, dev_accuracy))
                else:
                    print("Epoch: %d; Dev perplexity = %0.4f" % (epoch, dev_perplexity))
                # switch back to training mode
                model.train()

        # anneal eta_bn_prop from 1.0 to 0.0 over training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(0.75 * training_epochs)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

    # finish training
    model.eval()
    return model


def infer_categorical_covariate(model, network_architecture, X, Y, eta_bn_prop=0.0):
    """
    Predict the value of categorical covariates for each instances based on log probability of words
    """
    n_items, vocab_size = X.shape
    n_covariates = network_architecture['n_covariates']
    n_labels = network_architecture['n_labels']
    predictions = np.zeros(n_items, dtype=int)

    if n_covariates == 1:
        for i in range(n_items):
            C_i = np.zeros((2, 1)).astype('float32')
            C_i[1, 0] = 1.0
            X_i = np.zeros((2, vocab_size)).astype('float32')
            X_i[:, :] = X[i, :]
            if Y is not None:
                Y_i = np.zeros((2, n_labels)).astype('float32')
                Y_i[:, :] = Y[i, :]
            else:
                Y_i = None
            losses = model.get_losses(X_i, Y_i, C_i, eta_bn_prop=eta_bn_prop)
            pred = np.argmin(losses)
            predictions[i] = pred

    else:
        for i in range(n_items):
            C_i = np.eye(n_covariates).astype('float32')
            X_i = np.zeros((n_covariates, vocab_size)).astype('float32')
            X_i[:, :] = X[i, :]
            if Y is not None:
                Y_i = np.zeros((n_covariates, n_labels)).astype('float32')
                Y_i[:, :] = Y[i, :]
            else:
                Y_i = None
            losses = model.get_losses(X_i, Y_i, C_i, eta_bn_prop=eta_bn_prop)
            pred = np.argmin(losses)
            predictions[i] = pred

    return predictions


def predict_labels(model, X, C, eta_bn_prop=0.0):
    """
    Predict a label for each instance using the classifier part of the network
    """
    Z, Y_recon = model.predict(X, C, eta_bn_prop=eta_bn_prop)
    predictions = np.argmax(Y_recon, axis=1)
    return predictions


def print_top_words(beta, feature_names, topic_names=None, n_top_words=8, sparsity_threshold=1e-5, values=False):
    """
    Display the highest and lowest weighted words in each topic, along with mean ave weight and sparisty
    """
    sparsity_vals = []
    maw_vals = []
    for i in range(len(beta)):
        # sort the beta weights
        order = list(np.argsort(beta[i]))
        order.reverse()
        output = ''
        # get the top words
        for j in range(n_top_words):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        order.reverse()
        output += ' / '
        # get the bottom words
        for j in range(n_top_words):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        # compute sparsity
        sparsity = float(np.sum(np.abs(beta[i]) < sparsity_threshold) / float(len(beta[i])))
        maw = np.mean(np.abs(beta[i]))
        sparsity_vals.append(sparsity)
        maw_vals.append(maw)
        output += ': MAW=%0.4f' % maw + '; sparsity=%0.4f' % sparsity

        # print the topic summary
        if topic_names is not None:
            output = topic_names[i] + ': ' + output
        else:
            output = str(i) + ': ' + output
        print(output)

    # return mean average weight and sparsity
    return np.mean(maw_vals), np.mean(sparsity_vals)


def print_top_bg(bg, feature_names, n_top_words=10):
    """
    Print the most highly weighted words in the background log frequency
    """
    print('Background frequencies of top words:')
    print(" ".join([feature_names[j]
                    for j in bg.argsort()[:-n_top_words - 1:-1]]))
    temp = bg.copy()
    temp.sort()
    print(np.exp(temp[:-n_top_words-1:-1]))


def evaluate_perplexity(model, X, Y, C, eta_bn_prop=0.0):
    """
    Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    """
    # compute perplexity for all documents in a single batch
    doc_sums = np.array(X.sum(axis=1), dtype=float)
    X = X.astype('float32')
    if Y is not None:
        Y = Y.astype('float32')
    if C is not None:
        C = C.astype('float32')
    losses = model.get_losses(X, Y, C, eta_bn_prop=eta_bn_prop)
    perplexity = np.exp(np.mean(losses / doc_sums))

    return perplexity


def save_weights(output_dir, beta, bg, feature_names, sparsity_threshold=1e-5):
    """
    Save model weights to npz files (also the top words in each topic
    """
    np.savez(os.path.join(output_dir, 'beta.npz'), beta=beta)
    if bg is not None:
        np.savez(os.path.join(output_dir, 'bg.npz'), bg=bg)
    fh.write_to_json(feature_names, os.path.join(output_dir, 'vocab.json'), sort_keys=False)

    topics_file = os.path.join(output_dir, 'topics.txt')
    lines = []
    for i in range(len(beta)):
        order = list(np.argsort(beta[i]))
        order.reverse()
        pos_words = [feature_names[j] for j in order[:100] if beta[i][j] > sparsity_threshold]
        output = ' '.join(pos_words)
        lines.append(output)

    fh.write_list_to_text(lines, topics_file)


def predict_labels_and_evaluate(model, X, Y, C, output_dir=None, subset='train'):
    """
    Predict labels for all instances using the classifier network and evaluate the accuracy
    """
    predictions = predict_labels(model, X, C)
    accuracy = float(np.sum(predictions == np.argmax(Y, axis=1)) / float(len(Y)))
    print(subset, "accuracy on labels = %0.4f" % accuracy)
    if output_dir is not None:
        fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.' + subset + '.txt'))


if __name__ == '__main__':
    main()
