from optparse import OptionParser

import numpy as np

import file_handling as fh

# compute topic coherence in terms of NPMI with respect to a reference corpus

def main():
    usage = "%prog topics.txt ref_counts.npz ref_vocab.json"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='n_vals', default='10',
                      help='Number of words to consider (comma-separated): default=%default')
    parser.add_option('-c', dest='cols', default=0,
                      help='Columns to skip (for Mallet output): default=%default')
    parser.add_option('-o', dest='output_file', default=None,
                      help='Output file: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    topics_file = args[0]
    ref_counts_file = args[1]
    ref_vocab_file = args[2]
    n_vals = options.n_vals
    n_vals = [int(n) for n in n_vals.split(',')]
    cols_to_skip = int(options.cols)
    output_file = options.output_file

    load_and_compute_npmi(topics_file, ref_vocab_file, ref_counts_file, n_vals, cols_to_skip, output_file=output_file)


def load_and_compute_npmi(topics_file, ref_vocab_file, ref_counts_file, n_vals, cols_to_skip=0, output_file=None):
    print("Loading reference counts")
    ref_vocab = fh.read_json(ref_vocab_file)
    ref_counts = fh.load_sparse(ref_counts_file).tocsc()
    compute_npmi(topics_file, ref_vocab, ref_counts, n_vals, cols_to_skip, output_file)


def compute_npmi(topics_file, ref_vocab, ref_counts, n_vals, cols_to_skip=0, output_file=None):
    print("Loading topics")
    topics = fh.read_text(topics_file)

    mean_vals = []
    for n in n_vals:
        mean_npmi = compute_npmi_at_n(topics, ref_vocab, ref_counts, n, cols_to_skip=cols_to_skip)
        mean_vals.append(mean_npmi)

    if output_file is not None:
        lines = [str(n) + ' ' + str(v) for n, v in zip(n_vals, mean_vals)]
        fh.write_list_to_text(lines, output_file)


def compute_npmi_at_n(topics, ref_vocab, ref_counts, n=10, cols_to_skip=0):

    vocab_index = dict(zip(ref_vocab, range(len(ref_vocab))))
    n_docs, _ = ref_counts.shape

    npmi_means = []
    for topic in topics:
        words = topic.split()[cols_to_skip:]
        npmi_vals = []
        for word_i, word1 in enumerate(words[:n]):
            if word1 in vocab_index:
                index1 = vocab_index[word1]
            else:
                index1 = None
            for word2 in words[word_i+1:n]:
                if word2 in vocab_index:
                    index2 = vocab_index[word2]
                else:
                    index2 = None
                if index1 is None or index2 is None:
                    npmi = 0.0
                else:
                    col1 = np.array(ref_counts[:, index1].todense() > 0, dtype=int)
                    col2 = np.array(ref_counts[:, index2].todense() > 0, dtype=int)
                    c1 = col1.sum()
                    c2 = col2.sum()
                    c12 = np.sum(col1 * col2)
                    if c12 == 0:
                        npmi = 0.0
                    else:
                        npmi = (np.log10(n_docs) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(n_docs) - np.log10(c12))
                npmi_vals.append(npmi)
        print(str(np.mean(npmi_vals)) + ': ' + ' '.join(words[:n]))
        npmi_means.append(np.mean(npmi_vals))
    print(np.mean(npmi_means))
    return np.mean(npmi_means)


if __name__ == '__main__':
    main()
