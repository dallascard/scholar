import os
import sys
from optparse import OptionParser

import gensim
import numpy as np
import pandas as pd
import tensorflow as tf

import file_handling as fh
from scholar_tf import Scholar


def main():
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-a', dest='alpha', default=1.0,
                      help='Hyperparameter for logistic normal prior: default=%default')
    parser.add_option('-k', dest='n_topics', default=20,
                      help='Size of latent representation (~num topics): default=%default')
    parser.add_option('--batch-size', dest='batch_size', default=200,
                      help='Size of minibatches: default=%default')
    parser.add_option('-l', dest='learning_rate', default=0.002,
                      help='Initial learning rate: default=%default')
    parser.add_option('-m', dest='momentum', default=0.99,
                      help='beta1 for Adam: default=%default')
    parser.add_option('--epochs', dest='epochs', default=250,
                      help='Number of epochs: default=%default')
    parser.add_option('--train-prefix', type=str, default='train',
                      help='Prefix of train set: default=%default')
    parser.add_option('--test-prefix', dest='test_prefix', default=None,
                      help='Prefix of test set: default=%default')
    parser.add_option('--en-layers', dest='encoder_layers', default=1,
                      help='Number of encoder layers [0|1|2]: default=%default')
    parser.add_option('--emb-dim', dest='embedding_dim', default=300,
                      help='Dimension of input embeddings: default=%default')
    parser.add_option('--en-short', action="store_true", dest="encoder_shortcuts", default=False,
                      help='Use shortcut connections on encoder: default=%default')
    parser.add_option('--labels', dest='label_name', default=None,
                      help='Read labels from input_dir/[train|test]_prefix.label_name.csv: default=%default')
    parser.add_option('--topic-covars', dest='covar_names', default=None,
                      help='Read covars from files with these names (comma-separated): default=%default')
    parser.add_option('--label-emb-dim', dest='label_emb_dim', default=-1,
                      help="Class embedding dimension [-1 = identity; 0 = don't encode]: default=%default")
    parser.add_option('--covar-emb-dim', dest='covar_emb_dim', default=-1,
                      help="Covariate embedding dimension [-1 = identity; 0 = don't encode]: default=%default")
    parser.add_option('--min-covar-count', dest='min_covar_count', default=None,
                      help='Drop binary covariates that occur less than this in training: default=%default')
    parser.add_option('--interactions', action="store_true", dest="covar_interactions", default=False,
                      help='Use covariate interactions in model: default=%default')
    parser.add_option('--infer-covars', action="store_true", dest="infer_covars", default=False,
                      help='Infer categorical covariate values after fitting model (slow): default=%default')
    parser.add_option('--c-layers', dest='classifier_layers', default=1,
                      help='Number of layers in (generative) classifier [0|1|2]: default=%default')
    parser.add_option('--exclude-covars', action="store_true", dest="exclude_covars", default=False,
                      help='Exclude covariates from the classifier: default=%default')
    parser.add_option('-r', action="store_true", dest="regularize", default=False,
                      help='Apply adaptive regularization for sparsity in topics: default=%default')
    parser.add_option('-o', dest='output_dir', default='output',
                      help='Output directory: default=%default')
    parser.add_option('--w2v', dest='word2vec_file', default=None,
                      help='Use this word2vec .bin file to initialize and fix embeddings: default=%default')
    parser.add_option('--vocab-size', dest='vocab_size', default=None,
                      help='Filter the vocabulary keeping the most common n words: default=%default')
    parser.add_option('--update-bg', action="store_true", dest="update_bg", default=False,
                      help='Update background parameters: default=%default')
    parser.add_option('--no-bg', action="store_true", dest="no_bg", default=False,
                      help='Do not use background freq: default=%default')
    parser.add_option('--no-bn-anneal', action="store_true", dest="no_bn_anneal", default=False,
                      help='Do not anneal away from batchnorm: default=%default')
    parser.add_option('--samples', dest='test_samples', default=20,
                      help='Number of samples to use in computing test perplexity: default=%default')
    parser.add_option('--dev-folds', dest='dev_folds', default=0,
                      help='Number of dev folds: default=%default')
    parser.add_option('--dev-fold', dest='dev_fold', default=0,
                      help='Fold to use as dev (if dev_folds > 0): default=%default')
    parser.add_option('--opt', dest='optimizer', default='adam',
                      help='Optimization algorithm to use [adam|adagrad|sgd]: default=%default')
    parser.add_option('--threads', dest='threads', default=8,
                      help='Use this to limit the number of CPUs: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]

    train_prefix = options.train_prefix
    alpha = float(options.alpha)
    n_topics = int(options.n_topics)
    batch_size = int(options.batch_size)
    learning_rate = float(options.learning_rate)
    adam_beta1 = float(options.momentum)
    n_epochs = int(options.epochs)
    encoder_layers = int(options.encoder_layers)
    embedding_dim = int(options.embedding_dim)
    encoder_shortcuts = options.encoder_shortcuts
    label_file_name = options.label_name
    covar_file_names = options.covar_names
    use_covar_interactions = options.covar_interactions
    infer_covars = options.infer_covars
    label_emb_dim = int(options.label_emb_dim)
    covar_emb_dim = int(options.covar_emb_dim)
    min_covar_count = options.min_covar_count
    classifier_layers = int(options.classifier_layers)
    covars_in_classifier = not options.exclude_covars
    auto_regularize = options.regularize
    test_prefix = options.test_prefix
    output_dir = options.output_dir
    word2vec_file = options.word2vec_file
    vocab_size = options.vocab_size
    update_background = options.update_bg
    no_bg = options.no_bg
    bn_anneal = not options.no_bn_anneal
    test_samples = int(options.test_samples)
    dev_folds = int(options.dev_folds)
    dev_fold = int(options.dev_fold)
    optimizer = options.optimizer
    seed = options.seed
    threads = int(options.threads)
    if seed is not None:
        seed = int(seed)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(np.random.randint(0, 100000))

    # load the training data
    train_X, vocab, train_labels, label_names, label_type, train_covariates, covariate_names, covariates_type, col_sel = load_data(input_dir, train_prefix, label_file_name, covar_file_names, vocab_size=vocab_size)
    n_train, dv = train_X.shape

    if train_labels is not None:
        _, n_labels = train_labels.shape
    else:
        n_labels = 0

    if train_covariates is not None:
        _, n_covariates = train_covariates.shape
        # filter on covariate frequency, if desired
        if min_covar_count is not None and int(min_covar_count) > 0:
            print("Removing rare covariates")
            covar_sums = train_covariates.sum(axis=0).reshape((n_covariates, ))
            covariate_selector = covar_sums > int(min_covar_count)
            train_covariates = train_covariates[:, covariate_selector]
            covariate_names = [name for i, name in enumerate(covariate_names) if covariate_selector[i]]
            n_covariates = len(covariate_names)
    else:
        n_covariates = 0

    # split training data into train and dev
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

    # load test data using the same vocabulary
    if test_prefix is not None:
        test_X, _, test_labels, _, _, test_covariates, _, _, _ = load_data(input_dir, test_prefix, label_file_name, covar_file_names, vocab=vocab, col_sel=col_sel)
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

    # initialize the background using the overall frequency of terms
    init_bg = get_init_bg(train_X)
    init_beta = None
    update_beta = True
    if no_bg:
        if n_topics == 1:
            init_beta = init_bg.copy()
            init_beta = init_beta.reshape([1, len(vocab)])
            update_beta = False
        init_bg = np.zeros_like(init_bg)

    # create the network configuration
    network_architecture = make_network(dv, encoder_layers, embedding_dim,
                                        n_topics, encoder_shortcuts, label_type, n_labels, label_emb_dim,
                                        covariates_type, n_covariates, covar_emb_dim, use_covar_interactions,
                                        classifier_layers, covars_in_classifier)  # make_network()

    print("Network architecture:")
    for key, val in network_architecture.items():
        print(key + ':', val)

    # load pretrained word vectors
    if word2vec_file is not None:
        vocab_size = len(
            vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        embeddings = np.array(rng.rand(vocab_size, embedding_dim) * 0.25 - 0.5, dtype=np.float32)
        count = 0
        print("Loading word vectors")
        if word2vec_file[-3:] == 'bin':
            pretrained = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
        else:
            pretrained = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[index, :] = pretrained[word]

        print("Found embeddings for %d words" % count)
        update_embeddings = False
    else:
        embeddings = None
        update_embeddings = True

    tf.reset_default_graph()

    # create the model
    model = Scholar(network_architecture, alpha=alpha, learning_rate=learning_rate, batch_size=batch_size, init_embeddings=embeddings, update_embeddings=update_embeddings, init_bg=init_bg, update_background=update_background, init_beta=init_beta, update_beta=update_beta, threads=threads, regularize=auto_regularize, optimizer=optimizer, adam_beta1=adam_beta1, seed=seed)

    # train the model
    print("Optimizing full model")
    model = train(model, network_architecture, train_X, train_labels, train_covariates, regularize=auto_regularize, training_epochs=n_epochs, batch_size=batch_size, rng=rng, X_dev=dev_X, Y_dev=dev_labels, C_dev=dev_covariates, bn_anneal=bn_anneal)

    # create output directory
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
        if covar_emb_dim > 0:
            maw, sparsity = print_top_words(beta_c, vocab)
        else:
            maw, sparsity = print_top_words(beta_c, vocab, covariate_names)
        print("sparsity in covariates = %0.4f" % sparsity)
        if output_dir is not None:
            np.savez(os.path.join(output_dir, 'beta_c.npz'), beta=beta_c, names=covariate_names)
        if use_covar_interactions:
            print("Covariate interactions")
            beta_ci = model.get_covar_inter_weights()
            print(beta_ci.shape)
            if covariate_names is not None:
                names = [str(k) + ':' + c for k in range(n_topics) for c in covariate_names]
            else:
                names = None
            maw, sparsity = print_top_words(beta_ci, vocab, names)
            if output_dir is not None:
                np.savez(os.path.join(output_dir, 'beta_ci.npz'), beta=beta_ci, names=names)
            print("sparsity in covariate interactions = %0.4f" % sparsity)
            print("Combined covariates and interactions:")

        if covar_emb_dim > 0:
            print_covariate_embeddings(model, covariate_names, output_dir)

    # Evaluate perplexity on dev and test dataa
    if dev_X is not None:
        perplexity = evaluate_perplexity(model, dev_X, dev_labels, dev_covariates, eta_bn_prop=0.0, n_samples=test_samples)
        print("Dev perplexity = %0.4f" % perplexity)
        fh.write_list_to_text([str(perplexity)], os.path.join(output_dir, 'perplexity.dev.txt'))

    if test_X is not None:
        perplexity = evaluate_perplexity(model, test_X, test_labels, test_covariates, eta_bn_prop=0.0, n_samples=test_samples)
        print("Test perplexity = %0.4f" % perplexity)
        fh.write_list_to_text([str(perplexity)], os.path.join(output_dir, 'perplexity.test.txt'))

    # evaluate accuracy on covariates (if categorical)
    if n_covariates > 0 and covariates_type == 'categorical' and infer_covars:
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

    # evaluate accuracy on labels
    if n_labels > 0:
        print("Predicting labels")
        predict_labels_and_evaluate(model, train_X, train_labels, train_covariates, output_dir, subset='train')

        if dev_X is not None:
            predict_labels_and_evaluate(model, dev_X, dev_labels, dev_covariates, output_dir, subset='dev')

        if test_X is not None:
            predict_labels_and_evaluate(model, test_X, test_labels, test_covariates, output_dir, subset='test')

    # Print associations between topics and labels
    if n_labels > 0:
        all_probs = np.zeros([n_topics, n_labels])
        if n_labels < 15:
            print("Label probabilities based on topics")
            print("Labels:", ' '.join([name for name in label_names]))
        for k in range(n_topics):
            Z = np.zeros([1, n_topics]).astype('float32')
            Z[0, k] = 1.0
            if n_covariates > 0:
                C = np.zeros([1, n_covariates]).astype('float32')
            else:
                C = None
            probs = model.predict_from_topics(Z, C)
            all_probs[k, :] = probs
            if n_labels < 15:
                output = str(k) + ': '
                for i in range(n_labels):
                    output += '%.4f ' % probs[0, i]
                print(output)
        np.savez(os.path.join(output_dir, 'topic_label_probs.npz'), probs=all_probs)

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
    theta = model.compute_theta(train_X, train_labels, train_covariates)
    np.savez(os.path.join(output_dir, 'theta.train.npz'), theta=theta)

    if dev_X is not None:
        dev_Y = np.zeros_like(dev_labels)
        theta = model.compute_theta(dev_X, dev_Y, dev_covariates)
        np.savez(os.path.join(output_dir, 'theta.dev.npz'), theta=theta)

    if n_test > 0:
        test_Y = np.zeros_like(test_labels)
        theta = model.compute_theta(test_X, test_Y, test_covariates)
        np.savez(os.path.join(output_dir, 'theta.test.npz'), theta=theta)


def load_data(input_dir, input_prefix, label_file_name=None, covar_file_names=None, vocab_size=None, vocab=None, col_sel=None):
    print("Loading data")
    temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    n_items, temp_size = temp.shape
    print("Loaded %d documents with %d features" % (n_items, temp_size))

    if vocab is None:
        col_sel = None
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
        # filter vocabulary by word frequency
        if vocab_size is not None:
            print("Filtering vocabulary to the most common %d terms" % int(vocab_size))
            col_sums = np.array(temp.sum(axis=0)).reshape((len(vocab), ))
            order = list(np.argsort(col_sums))
            order.reverse()
            col_sel = np.array(np.zeros(len(vocab)), dtype=bool)
            for i in range(int(vocab_size)):
                col_sel[order[i]] = True
            temp = temp[:, col_sel]
            vocab = [word for i, word in enumerate(vocab) if col_sel[i]]

    elif col_sel is not None:
        print("Using given vocabulary")
        temp = temp[:, col_sel]

    X = np.array(temp, dtype='float32')
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

    return X, vocab, labels, label_names, label_type, covariates, covariate_names, covariates_type, col_sel


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


def make_network(dv, encoder_layers=2, embedding_dim=300, n_topics=50, encoder_shortcut=False, label_type=None, n_labels=0, label_emb_dim=0, covariate_type=None, n_covariates=0, covar_emb_dim=0, use_covar_interactions=False, classifier_layers=1, covars_in_classifier=True):
    """
    Combine the network configuration parameters into a dictionary
    """
    tf.reset_default_graph()
    network_architecture = \
        dict(encoder_layers=encoder_layers,
             encoder_shortcut=encoder_shortcut,
             embedding_dim=embedding_dim,
             n_topics=n_topics,
             dv=dv,
             label_type=label_type,
             n_labels=n_labels,
             label_emb_dim=label_emb_dim,
             covariate_type=covariate_type,
             n_covariates=n_covariates,
             covar_emb_dim=covar_emb_dim,
             use_covar_interactions=use_covar_interactions,
             classifier_layers=classifier_layers,
             covars_in_classifier=covars_in_classifier
             )
    return network_architecture


def train(model, network_architecture, X, Y, C, batch_size=200, training_epochs=100, display_step=5, min_weights_sq=1e-7, regularize=False, X_dev=None, Y_dev=None, C_dev=None, bn_anneal=True, init_eta_bn_prop=1.0, rng=None):

    n_train, dv = X.shape
    mb_gen = create_minibatch(X, Y, C, batch_size=batch_size, rng=rng)

    dv = network_architecture['dv']
    n_topics = network_architecture['n_topics']

    total_batch = int(n_train / batch_size)

    # create np arrays to store regularization strengths, which we'll update outside of the tensorflow model
    if regularize:
        l2_strengths = 0.5 * np.ones([n_topics, dv]) / float(n_train)
        l2_strengths_c = 0.5 * np.ones([model.beta_c_length, dv]) / float(n_train)
        l2_strengths_ci = 0.5 * np.ones([model.beta_ci_length, dv]) / float(n_train)
    else:
        l2_strengths = np.zeros([n_topics, dv])
        l2_strengths_c = np.zeros([model.beta_c_length, dv])
        l2_strengths_ci = np.zeros([model.beta_ci_length, dv])

    batches = 0

    eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon
    kld_weight = 1.0  # could use this to anneal KLD, but not currently doing so

    # Training cycle
    for epoch in range(training_epochs):
        avg_loss = 0.
        avg_cls_loss = 0.
        accuracy = 0.
        # Loop over all batches
        for i in range(total_batch):
            # get a minibatch
            batch_xs, batch_ys, batch_cs = next(mb_gen)
            # do one update, passing in the data, regularization strengths, and bn
            loss, cls_loss, pred = model.fit(batch_xs, batch_ys, batch_cs, l2_strengths=l2_strengths, l2_strengths_c=l2_strengths_c, l2_strengths_ci=l2_strengths_ci, eta_bn_prop=eta_bn_prop, kld_weight=kld_weight)
            # compute accuracy on minibatch
            if network_architecture['n_labels'] > 0:
                accuracy += np.sum(pred == np.argmax(batch_ys, axis=1)) / float(n_train)
            # Compute average loss
            avg_loss += loss / n_train * batch_size
            avg_cls_loss += cls_loss / n_train * batch_size
            batches += 1
            if np.isnan(avg_loss):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()

        # update weight prior variances using current weight values
        if regularize:
            weights = model.get_weights()
            weights_sq = weights ** 2
            # avoid infinite regularization
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l2_strengths = 0.5 / weights_sq / float(n_train)

            if network_architecture['n_covariates'] > 0:
                weights = model.get_covar_weights()
                weights_sq = weights ** 2
                # avoid infinite regularization
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l2_strengths_c = 0.5 / weights_sq / float(n_train)
                if network_architecture['use_covar_interactions']:
                    weights = model.get_covar_inter_weights()
                    weights_sq = weights ** 2
                    # avoid infinite regularization
                    weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                    l2_strengths_ci = 0.5 / weights_sq / float(n_train)

        # Display logs per epoch step
        if epoch % display_step == 0 and epoch > 0:
            if network_architecture['n_labels'] > 0:
                print("Epoch:", '%d' % epoch, "; loss =", "{:.9f}".format(avg_loss), "; clsloss =", "{:.9f}".format(avg_cls_loss), "; training accuracy (noisy) =", "{:.9f}".format(accuracy))
            else:
                print("Epoch:", '%d' % epoch, "loss=", "{:.9f}".format(avg_loss))

            if X_dev is not None:
                dev_perplexity = evaluate_perplexity(model, X_dev, Y_dev, C_dev, eta_bn_prop=eta_bn_prop)
                n_dev, _ = X_dev.shape
                if network_architecture['n_labels'] > 0:
                    dev_predictions = predict_labels(model, X_dev, C_dev, eta_bn_prop=eta_bn_prop)
                    dev_accuracy = float(np.sum(dev_predictions == np.argmax(Y_dev, axis=1))) / float(n_dev)
                    print("Epoch: %d; Dev perplexity = %0.4f; Dev accuracy = %0.4f" % (epoch, dev_perplexity, dev_accuracy))
                else:
                    print("Epoch: %d; Dev perplexity = %0.4f" % (epoch, dev_perplexity))

        # anneal eta_bn_prop from 1 to 0 over the course of training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(training_epochs*0.75)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

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
        # process instances one by one
        for i in range(n_items):
            # create a matrix of all possible covariate values and evaluate all as a minibatch
            C_i = np.eye(n_covariates).astype('float32')
            X_i = np.zeros((n_covariates, vocab_size)).astype('float32')
            X_i[:, :] = X[i, :]
            if Y is not None:
                Y_i = np.zeros((n_covariates, n_labels)).astype('float32')
                Y_i[:, :] = Y[i, :]
            else:
                Y_i = None
            # check the log-loss for each possible value of C and take the best
            losses = model.get_losses(X_i, Y_i, C_i, eta_bn_prop=eta_bn_prop)
            pred = np.argmin(losses)
            predictions[i] = pred

    return predictions


def predict_labels(model, X, C, eta_bn_prop=0.0):
    """
    Predict a label for each instance using the classifier part of the network
    """
    n_items, vocab_size = X.shape
    predictions = np.zeros(n_items, dtype=int)

    # predict items one by one
    for i in range(n_items):
        X_i = np.expand_dims(X[i, :], axis=0)
        # optionally provide covariates
        if C is not None:
            C_i = np.expand_dims(C[i, :], axis=0)
        else:
            C_i = None

        # predict probabilities
        z, y_recon = model.predict(X_i, C_i, eta_bn_prop=eta_bn_prop)

        # take the label with the maximum predicted probability
        pred = np.argmax(y_recon)
        predictions[i] = pred

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


def evaluate_perplexity(model, X, Y, C, eta_bn_prop=1.0, n_samples=0):
    """
    Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    """
    # count the number of words in each document
    doc_sums = np.array(X.sum(axis=1), dtype=float)
    X = X.astype('float32')
    if Y is not None:
        Y = Y.astype('float32')
    if C is not None:
        C = C.astype('float32')
    # get the losses for all instances
    losses = model.get_losses(X, Y, C, eta_bn_prop=eta_bn_prop, n_samples=n_samples)
    # compute perplexity for all documents in a single batch
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
        pos_words = [feature_names[j] for j in order[:40] if beta[i][j] > sparsity_threshold]
        output = ' '.join(pos_words)
        lines.append(output)

    fh.write_list_to_text(lines, topics_file)


def print_label_embeddings(model, class_names):
    """
    Display label embeddings
    """
    label_embeddings = model.get_label_embeddings()
    n_labels, _ = label_embeddings.shape
    dists = np.zeros([n_labels, n_labels])
    for i in range(n_labels):
        for j in range(n_labels):
            emb_i = label_embeddings[i, :]
            emb_j = label_embeddings[j, :]
            dists[i, j] = np.dot(emb_i, emb_j) / np.sqrt(np.dot(emb_i, emb_i)) / np.sqrt(np.dot(emb_j, emb_j))
    for i in range(n_labels):
        order = list(np.argsort(dists[i, :]))
        order.reverse()
        output = class_names[i] + ': '
        for j in range(4):
            output += class_names[order[j]] + ' '
        print(output)


def print_covariate_embeddings(model, covariate_names, output_dir):
    """
    Display covariate embeddings
    """
    covar_embeddings = model.get_covar_embeddings()
    n_covariates , emb_dim = covar_embeddings.shape
    dists = np.zeros([n_covariates, n_covariates])
    for i in range(n_covariates):
        for j in range(n_covariates):
            emb_i = covar_embeddings[i, :]
            emb_j = covar_embeddings[j, :]
            dists[i, j] = np.dot(emb_i, emb_j) / np.sqrt(np.dot(emb_i, emb_i)) / np.sqrt(np.dot(emb_j, emb_j))
    for i in range(n_covariates):
        order = list(np.argsort(dists[i, :]))
        order.reverse()
        output = covariate_names[i] + ': '
        for j in range(4):
            output += covariate_names[order[j]] + ' '
        print(output)
    if n_covariates < 30 and emb_dim < 10:
        print(covar_embeddings)
    np.savez(os.path.join(output_dir, 'covar_emb.npz'), W=covar_embeddings, names=covariate_names)


def predict_labels_and_evaluate(model, X, Y, C, output_dir=None, subset='train'):
    """
    Predict labels for all instances using the classifier network and evaluate the accuracy
    """
    n_items, vocab_size = X.shape
    predictions = predict_labels(model, X, C)
    accuracy = float(np.sum(predictions == np.argmax(Y, axis=1)) / float(n_items))

    print(subset, "accuracy on labels = %0.4f" % accuracy)
    # save the results to file
    if output_dir is not None:
        fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.' + subset + '.txt'))


if __name__ == '__main__':
    main()
