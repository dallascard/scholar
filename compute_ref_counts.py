import re
import os
import codecs
from collections import Counter
from optparse import OptionParser

from scipy import sparse

import file_handling as fh

# Count word occurrence statistics for computing NPMI on a file with one document per line


def main():
    usage = "%prog infile.txt output_dir output_prefix"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='max_lines', default=None,
                      help='Quit after processing this many lines (documents): default=%default')
    #parser.add_option('--lower', action="store_true", dest="lower", default=False,
    #                  help='Lower case words: default=%default')

    (options, args) = parser.parse_args()

    infile = args[0]
    output_dir = args[1]
    output_prefix = args[2]

    max_lines = options.max_lines
    if max_lines is not None:
        max_lines = int(max_lines)

    vocab = []
    vocab_index = {}

    counter = Counter()

    # start by converting each document into a dict of word counts, building a vocab as we go
    rows = []
    cols = []
    values = []
    n_docs = 0
    print("Counting words...")
    with codecs.open(infile, 'r', encoding='utf-8') as f:
        for line_i, line in enumerate(f):
            line = line.strip()
            if len(line) > 0:
                if max_lines is not None and line_i >= max_lines:
                    print("Quitting after processing %d lines" % (line_i+1))
                    break
                if n_docs % 1000 == 0 and n_docs > 0:
                    print(n_docs)
                # split on white space
                words = line.split()
                # filter out everything that's not just letters, and lower case
                words = [word.lower() for word in words if re.match('^[a-zA-Z]*$', word) is not None]
                # look for new words and add them to the vocabulary
                new_words = [word for word in words if word not in vocab_index]
                if len(new_words) > 0:
                    vocab_size = len(vocab)
                    #print("Adding %d words to vocab" % len(new_words))
                    #print("New total should be %d" % (vocab_size + len(new_words)))
                    vocab.extend(new_words)
                    vocab_index.update(dict(zip(new_words, range(vocab_size, vocab_size + len(new_words)))))
                indices = [vocab_index[word] for word in words]
                counter.clear()
                counter.update(indices)
                keys = counter.keys()
                counts = counter.values()
                rows.extend([line_i] * len(keys))
                cols.extend(keys)
                values.extend(counts)
                n_docs += 1

    print("Processed %d documents" % n_docs)
    print("Size of final vocab = %d" % len(vocab))
    print("Saving counts...")

    # now convert these count vectors in to a giant sparse matrix
    counts = sparse.coo_matrix((values, (rows, cols)), shape=(n_docs, len(vocab)))
    fh.save_sparse(counts, os.path.join(output_dir, output_prefix + '.npz'))
    fh.write_to_json(vocab, os.path.join(output_dir, output_prefix + '.vocab.json'))
    print("Done")


if __name__ == '__main__':
    main()
