import os
from optparse import OptionParser

import numpy as np
import pandas as pd

import file_handling as fh

"""
Randomly split a jsonlist (and optionally, a labels and/or covaraites dataframe) into train and test.
"""

def main():
    usage = "%prog input.jsonlist ouput_dir [labels.csv covariates.csv ...]"
    parser = OptionParser(usage=usage)
    parser.add_option('--test_prop', dest='test_prop', default=0.2,
                      help='proportion of documents to use for test data: default=%default')
    parser.add_option('--train', dest='train', default='train',
                      help='output prefix for training data: default=%default')
    parser.add_option('--test', dest='test', default='test',
                      help='output prefix for test data: default=%default')

    (options, args) = parser.parse_args()
    infile = args[0]
    output_dir = args[1]
    if len(args) > 2:
        csv_files = args[2:]
    else:
        csv_files = []

    test_prop = float(options.test_prop)
    train_prefix = options.train
    test_prefix = options.test

    print("Reading", infile)
    items = fh.read_jsonlist(infile)
    n_items = len(items)

    n_test = int(n_items * test_prop)
    print("Creating random test set of %d items" % n_test)
    n_train = n_items - n_test
    train_indices = np.random.choice(np.arange(n_items), size=n_train, replace=False)
    test_indices = list(set(range(n_items)) - set(train_indices))

    train_items = [items[i] for i in train_indices]
    test_items = [items[i] for i in test_indices]

    fh.write_jsonlist(train_items, os.path.join(output_dir, train_prefix + '.jsonlist'))
    fh.write_jsonlist(test_items, os.path.join(output_dir, test_prefix + '.jsonlist'))

    for file in csv_files:
        print(file)
        basename = os.path.basename(file)
        df = pd.read_csv(file, header=0, index_col=0)
        train_df_index = [df.index[i] for i in train_indices]
        train_df = df.loc[train_df_index]
        train_df.to_csv(os.path.join(output_dir, train_prefix + '.' + basename))

        test_df_index = [df.index[i] for i in test_indices]
        test_df = df.loc[test_df_index]
        test_df.to_csv(os.path.join(output_dir, test_prefix + '.' + basename))


if __name__ == '__main__':
    main()
