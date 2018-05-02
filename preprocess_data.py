import os
import re
import sys
import string
from optparse import OptionParser
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat

import file_handling as fh

"""
Convert a dataset into the required format (as well as formats required by other tools).
Input format is one line per item.
Each line should be a json object.
At a minimum, each json object should have a "text" field, with the document text.
Any other field can be used as a label (specified with the --label option).
An optional "metadata" field can contain a dictionary with additional fields which will be used.
If training and test data are to be processed separately, the same input directory should be used
Run "python preprocess_data -h" for more options.
If an 'id' field is provided, this will be used as an identifier in the dataframes, otherwise index will be used 
"""

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


def main():
    usage = "%prog train.jsonlist output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default=None,
                      help='field to use as label: default=%default')
    parser.add_option('--test', dest='test', default=None,
                      help='Test data (test.jsonlist): default=%default')
    parser.add_option('--train_prefix', dest='train_prefix', default='train',
                      help='Output prefix for training data: default=%default')
    parser.add_option('--test_prefix', dest='test_prefix', default='test',
                      help='Output prefix for test data: default=%default')
    parser.add_option('--stopwords', dest='stopwords', default='snowball',
                      help='List of stopwords to exclude [None|mallet|snowball]: default=%default')
    parser.add_option('--min_doc_count', dest='min_doc_count', default=0,
                      help='Exclude words that occur in less than this number of documents')
    parser.add_option('--max_doc_freq', dest='max_doc_freq', default=1.0,
                      help='Exclude words that occur in more than this proportion of documents')
    parser.add_option('--keep_num', action="store_true", dest="keep_num", default=False,
                      help='Keep tokens made of only numbers: default=%default')
    parser.add_option('--keep_alphanum', action="store_true", dest="keep_alphanum", default=False,
                      help="Keep tokens made of a mixture of letters and numbers: default=%default")
    parser.add_option('--strip_html', action="store_true", dest="strip_html", default=False,
                      help='Strip HTML tags: default=%default')
    parser.add_option('--no_lower', action="store_true", dest="no_lower", default=False,
                      help='Do not lowercase text: default=%default')
    parser.add_option('--min_length', dest='min_length', default=3,
                      help='Minimum token length: default=%default')
    parser.add_option('--vocab_size', dest='vocab_size', default=None,
                      help='Size of the vocabulary (by most common, following above exclusions): default=%default')
    parser.add_option('--bigrams', action="store_true", dest="bigrams", default=False,
                      help='Output bigrams instead of unigrams: default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random integer seed (only relevant for choosing test set): default=%default')

    (options, args) = parser.parse_args()

    train_infile = args[0]
    output_dir = args[1]

    test_infile = options.test
    train_prefix = options.train_prefix
    test_prefix = options.test_prefix
    label_name = options.label
    min_doc_count = int(options.min_doc_count)
    max_doc_freq = float(options.max_doc_freq)
    vocab_size = options.vocab_size
    stopwords = options.stopwords
    if stopwords == 'None':
        stopwords = None
    keep_num = options.keep_num
    keep_alphanum = options.keep_alphanum
    strip_html = options.strip_html
    lower = not options.no_lower
    min_length = int(options.min_length)
    bigrams = options.bigrams
    seed = options.seed
    if seed is not None:
        np.random.seed(int(seed))

    if not os.path.exists(output_dir):
        sys.exit("Error: output directory does not exist")

    preprocess_data(train_infile, test_infile, output_dir, train_prefix, test_prefix, min_doc_count, max_doc_freq, vocab_size, stopwords, keep_num, keep_alphanum, strip_html, lower, min_length, label_name=label_name, bigrams=bigrams)


def preprocess_data(train_infile, test_infile, output_dir, train_prefix, test_prefix, min_doc_count=0, max_doc_freq=1.0, vocab_size=None, stopwords=None, keep_num=False, keep_alphanum=False, strip_html=False, lower=True, min_length=3, label_name=None, output_plaintext=False, bigrams=False):

    if stopwords == 'mallet':
        print("Using Mallet stopwords")
        stopword_list = fh.read_text(os.path.join('stopwords', 'mallet_stopwords.txt'))
    elif stopwords == 'snowball':
        print("Using snowball stopwords")
        stopword_list = fh.read_text(os.path.join('stopwords', 'snowball_stopwords.txt'))
    elif stopwords is not None:
        print("Using custom stopwords")
        stopword_list = fh.read_text(os.path.join('stopwords', stopwords + '_stopwords.txt'))
    else:
        stopword_list = []
    stopword_set = {s.strip() for s in stopword_list}

    print("Reading data files")
    train_items = fh.read_jsonlist(train_infile)
    n_train = len(train_items)

    if test_infile is not None:
        test_items = fh.read_jsonlist(test_infile)
        n_test = len(test_items)
    else:
        test_items = []
        n_test = 0

    all_items = train_items + test_items
    n_items = n_train + n_test

    # determine labels and metadata from data
    metadata_keys = set()
    print("Dealing with labels and metadata")
    # find all the metadata keys present
    for i, item in enumerate(all_items):
        if 'text' not in item:
            print("Text field not found for item %d" % i)
            sys.exit()
        if 'metadata' in item:
            for key in item['metadata'].keys():
                metadata_keys.add(key)

    # only keep the ones that are present everywhere
    if len(metadata_keys) > 0:
        for i, item in enumerate(all_items):
            if 'metadata' not in item:
                print('metadata not found for item %d' % i)
            for key in metadata_keys:
                if key not in item['metadata']:
                    print('dropping metadata field %s (not found for item %d)' % (key, i))
                    metadata_keys.remove(key)

    metadata_keys = list(metadata_keys)
    metadata_keys.sort()
    if len(metadata_keys) > 0:
        print("Metadata keys:", metadata_keys)

    label_set = set()
    for i, item in enumerate(all_items):
        if label_name is not None:
            label_set.add(item[label_name])

    label_list = list(label_set)
    label_list.sort()
    n_labels = len(label_list)
    if label_name is not None:
        print("Using label %s with %d classes" % (label_name, n_labels))

    # make vocabulary
    train_parsed = []
    test_parsed = []

    print("Parsing %d documents" % n_items)
    word_counts = Counter()
    doc_counts = Counter()
    count = 0
    for i, item in enumerate(all_items):
        if i % 1000 == 0 and count > 0:
            print(i)

        text = item['text']
        tokens = tokenize(text, strip_html=strip_html, lower=lower, keep_numbers=keep_num, keep_alphanum=keep_alphanum, min_length=min_length, stopwords=stopword_set, bigrams=bigrams)

        # store the parsed documents
        if i < n_train:
            train_parsed.append(tokens)
        else:
            test_parsed.append(tokens)

        # keep track fo the number of documents with each word
        word_counts.update(tokens)
        doc_counts.update(set(tokens))

    print("Size of full vocabulary=%d" % len(word_counts))

    print("Selecting the vocabulary")
    most_common = doc_counts.most_common()
    words, doc_counts = zip(*most_common)
    doc_freqs = np.array(doc_counts) / float(n_items)
    vocab = [word for i, word in enumerate(words) if doc_counts[i] >= min_doc_count and doc_freqs[i] <= max_doc_freq]
    most_common = [word for i, word in enumerate(words) if doc_freqs[i] > max_doc_freq]
    print("Excluding most common:", most_common)

    print("Vocab size after filtering = %d" % len(vocab))
    if vocab_size is not None:
        if len(vocab) > int(vocab_size):
            vocab = vocab[:int(vocab_size)]

    vocab_size = len(vocab)
    print("Final vocab size = %d" % vocab_size)

    print("Most common words remaining:", ' '.join(vocab[:10]))
    vocab.sort()

    fh.write_to_json(vocab, os.path.join(output_dir, train_prefix + '.vocab.json'))

    train_X_sage, tr_aspect, tr_no_aspect, tr_widx, vocab_for_sage = process_subset(train_items, train_parsed, label_name, label_list, vocab, metadata_keys, output_dir, train_prefix, output_plaintext=output_plaintext)
    if n_test > 0:
        test_X_sage, te_aspect, te_no_aspect, _, _= process_subset(test_items, test_parsed, label_name, label_list, vocab, metadata_keys, output_dir, test_prefix, output_plaintext=output_plaintext)

    train_sum = np.array(train_X_sage.sum(axis=0))
    print("%d words missing from training data" % np.sum(train_sum == 0))

    if n_test > 0:
        test_sum = np.array(test_X_sage.sum(axis=0))
        print("%d words missing from test data" % np.sum(test_sum == 0))

    sage_output = {'tr_data': train_X_sage, 'tr_aspect': tr_aspect, 'widx': tr_widx, 'vocab': vocab_for_sage}
    if n_test > 0:
        sage_output['te_data'] = test_X_sage
        sage_output['te_aspect'] = te_aspect
    savemat(os.path.join(output_dir, 'sage_labeled.mat'), sage_output)
    sage_output['tr_aspect'] = tr_no_aspect
    if n_test > 0:
        sage_output['te_aspect'] = te_no_aspect
    savemat(os.path.join(output_dir, 'sage_unlabeled.mat'), sage_output)


def process_subset(items, parsed, label_name, label_list, vocab, metadata_keys, output_dir, output_prefix, output_plaintext=False):
    n_items = len(items)
    n_labels = len(label_list)
    vocab_size = len(vocab)
    vocab_index = dict(zip(vocab, range(vocab_size)))

    ids = []
    for i, item in enumerate(items):
        if 'id' in item:
            ids.append(item['id'])
    if len(ids) != n_items:
        ids = [str(i) for i in range(n_items)]

    # create a label index using string representations
    label_list_strings = [str(label) for label in label_list]
    label_index = dict(zip(label_list_strings, range(n_labels)))

    # convert labels to a data frame
    labels_df = None
    label_vector_df = None
    if n_labels > 0:
        labels_df = pd.DataFrame(np.zeros([n_items, n_labels], dtype=int), index=ids, columns=label_list_strings)
        label_vector_df = pd.DataFrame(np.zeros(n_items, dtype=int), index=ids, columns=[label_name])
    else:
        print("No labels found")

    # convert metadata to a dataframe
    metadata_df = None
    if len(metadata_keys) > 0:
        metadata_df = pd.DataFrame(index=ids, columns=metadata_keys)

    for i, item in enumerate(items):
        id = ids[i]
        if label_name is not None:
            label = item[label_name]
            labels_df.loc[id][str(label)] = 1
            label_vector_df.loc[id] = label_index[str(label)]

        for key in metadata_keys:
            metadata_df.loc[id][key] = item['metadata'][key]

    # save labels
    if labels_df is not None:
        labels_df.to_csv(os.path.join(output_dir, output_prefix + '.' + label_name + '.csv'))
        label_vector_df.to_csv(os.path.join(output_dir, output_prefix + '.label_vector.csv'))

    # save metadata
    if metadata_df is not None:
        metadata_df.to_csv(os.path.join(output_dir, output_prefix + '.covariates.csv'))

    X = np.zeros([n_items, vocab_size], dtype=int)

    dat_strings = []
    dat_labels = []
    mallet_strings = []
    fast_text_lines = []

    counter = Counter()
    word_counter = Counter()
    doc_lines = []
    print("Converting to count representations")
    for i, words in enumerate(parsed):
        # get the vocab indices of words that are in the vocabulary
        indices = [vocab_index[word] for word in words if word in vocab_index]
        word_subset = [word for word in words if word in vocab_index]

        #if output_plaintext:
        #    doc_lines.append(' '.join(word_subset))

        counter.clear()
        counter.update(indices)
        word_counter.clear()
        word_counter.update(word_subset)

        if len(counter.keys()) > 0:
            # udpate the counts
            mallet_strings.append(str(i) + '\t' + 'en' + '\t' + ' '.join(word_subset))

            dat_string = str(int(len(counter))) + ' '
            dat_string += ' '.join([str(k) + ':' + str(int(v)) for k, v in zip(list(counter.keys()), list(counter.values()))])
            dat_strings.append(dat_string)

            if label_name is not None:
                label = items[i][label_name]
                dat_labels.append(str(label_index[str(label)]))

            values = list(counter.values())
            X[np.ones(len(counter.keys()), dtype=int) * i, list(counter.keys())] += values

    # convert to a sparse representation
    sparse_X = sparse.csr_matrix(X)
    fh.save_sparse(sparse_X, os.path.join(output_dir, output_prefix + '.npz'))

    print(sparse_X.shape)
    print(len(dat_strings))

    fh.write_to_json(ids, os.path.join(output_dir, output_prefix + '.ids.json'))

    # save output for Mallet
    fh.write_list_to_text(mallet_strings, os.path.join(output_dir, output_prefix + '.mallet.txt'))

    # save output for David Blei's LDA/SLDA code
    fh.write_list_to_text(dat_strings, os.path.join(output_dir, output_prefix + '.data.dat'))
    if len(dat_labels) > 0:
        fh.write_list_to_text(dat_labels, os.path.join(output_dir, output_prefix + '.' + label_name + '.dat'))

    #if output_plaintext:
    #    fh.write_list_to_text(doc_lines, os.path.join(output_dir, output_prefix + '.plaintext.txt'))

    # save output for Jacob Eisenstein's SAGE code:
    sparse_X_sage = sparse.csr_matrix(X, dtype=float)
    vocab_for_sage = np.zeros((vocab_size,), dtype=np.object)
    vocab_for_sage[:] = vocab
    if n_labels > 0:
        # convert array to vector of labels for SAGE
        sage_aspect = np.argmax(np.array(labels_df.values, dtype=float), axis=1) + 1
    else:
        sage_aspect = np.ones([n_items, 1], dtype=float)
    sage_no_aspect = np.array([n_items, 1], dtype=float)
    widx = np.arange(vocab_size, dtype=float) + 1

    return sparse_X_sage, sage_aspect, sage_no_aspect, widx, vocab_for_sage


def tokenize(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False, keep_numbers=False, keep_alphanum=False, min_length=3, stopwords=None, bigrams=False):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)
    tokens = text.split()
    if bigrams:
        if stopwords is None:
            tokens = [tokens[i] + '_' + tokens[i+1] for i in range(len(tokens)-1)]
        else:
            tokens = [tokens[i] + '_' + tokens[i+1] for i in range(len(tokens)-1) if tokens[i] not in stopwords and tokens[i+1] not in stopwords]
    elif stopwords is not None:
        tokens = [t for t in tokens if t not in stopwords]

    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:
        tokens = [t for t in tokens if alpha.match(t)]
    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t for t in tokens if alpha_or_num.match(t)]

    # drop short tokens
    if min_length > 0:
        tokens = [t for t in tokens if len(t) >= min_length]

    return tokens


def clean_text(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
    # remove html tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # replace angle brackets
        text = re.sub(r'<', '(', text)
        text = re.sub(r'>', ')', text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r'\S+@\S+', ' ', text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r'\s@\S+', ' ', text)
    # replace underscores with spaces
    text = re.sub(r'_', ' ', text)
    # break off single quotes at the ends of words
    text = re.sub(r'\s\'', ' ', text)
    text = re.sub(r'\'\s', ' ', text)
    # remove periods
    text = re.sub(r'\.', '', text)
    # replace all other punctuation (except single quotes) with spaces
    text = replace.sub(' ', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == '__main__':
    main()
