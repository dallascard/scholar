# Scholar

[Scholar](https://arxiv.org/abs/1705.09296) is a tool for modeling documents with metadata.


## Requirements:

- python3
- pytorch 0.4
- numpy
- scipy
- pandas
- gensim
- torchvision (for IMDB download only)


## Installation:

It is recommended that you install the requirements using Anaconda. Specifically, you can use the following commands to create a new environment, activate it, and install the necessary packages:

`conda create -n scholar python=3`

`source activate scholar`

`conda install pytorch torchvision -c pytorch`

`conda install numpy scipy pandas gensim`

Once the necessary packages are installed, there is no need to compile or install this repo.


## Quick start:

To test out the code, start by downloading the IMDB corpus:

`python download_imdb.py`

Preprocess the data using:

`python preprocess_data.py data/imdb/train.jsonlist data/imdb/processed/ --vocab-size 2000 --label sentiment --test data/imdb/test.jsonlist`

Train a model on this corpus with 10 topics using sentiment as a label on GPU 0:

`python run_scholar.py data/imdb/processed/ -k 10 --test test --labels sentiment --device 0`

## Preprocessing and file formats:

The above command will look for input files with specific names and formats. In order to automatically convert a corpus into the required format, use the `preprocess_data.py` script. The basic usage is:

`python preprocess_data.py train.jsonlist output_dir --vocab-size <vocab_size>`

The required format for `train.jsonlist` is a text file with one line per document, each of which should be a JSON object. At a minimum, each JSON should contain a "text" field, which should be a string containing the document text.

To get a sample dataset to follow along with this documentation, run `python download_imdb.py`. This will create four files in `data/imdb/`, of which you will only need `train.jsonlist` and `test.jsonlist`. The first line of `test.jsonlist` (truncated) should be:

`{"id": 127, "orig": "aclImdb/test/neg/127_3.txt", "sentiment": "neg", "rating": 3, "text": "I love sci-fi and am [...]"}`

If an "id" field is provided, this will be used as the document id (should be unique across train and test). If label information is included as a field in the json object (such as "sentiment"), this can be specified and automatically converted to the requried format.

When preprocessing a corpus, it is recommended that you specify a vocabulary size, which will keep only the most frequent words.

To preprocess the IMDB training data file using the "sentiment" label, with a 2,000 word vocabulary, run:

`python preprocess_data.py data/imdb/train.jsonlist data/imdb/processed/ --vocab-size 2000 --label sentiment --test data/imdb/test.jsonlist`

This will create several files in the processed directory, including:

- `train.npz`, containing the word counts per document
- `train.vocab.json`, containing a list of the words in the vocabulary
- `train.sentiment.csv`, containing the sentiment label for each document in .csv format
- equivalent files for the test corpus
- other files used by other pacakges, such as lda-c, SAGE, and Mallet

If your data does not have labels, you can simply not use the `--label` option.

The `--test` options specifies a second .jsonlist file to process using the same vocabulary as the first file (train.jsonlist). If your whole corpus is in one file (without a split between train and test), you can simply not use the `--test` option.

### Additional preprocesing options

The "train" prefix can be changed with the `--train-prefix` option `preprocess_data.py` or `run_scholar.py`.

By default, `preprocess_data.py` will exclude punctuation, numbers, stopwords, and words less than 3 characters long. To see additional options to modify this behaviour, run

`python preprocess_data.py -h`

### Manual preprocessing and additional covariates

If you want to do your own preprocessing or use additional metadata, you need to create the following files.

- `train.npz`: a sparse matrix of document word counts (n_documents x vocab_size)
- `train.vocab.json`: a list containing the words in the vocabulary in the same order as the columns of the matrix in train.npz
- `train.<label/covar>.csv`: A .csv file for each label or covariate, where <label/covar> is the corresponding name. The first row of the .csv should be a set of column names (one per possible label or covariate value, even in the binary case), and the first column should be a column of document indices. (Follow the above example and look at `data/imdb/processed/train.sentiment.csv` for an example


## Options for running the model:

To run a Scholar model on a preprocessed corpus with metadata, the basic command is:

`python run_scholar.py input_directory`

This will look for the `train.npz` and `train.vocab.json` files in the input directory..

To specify the number of topics, use `-k number_of_topics` (default 10).

To specify the number of epochs, use `--epochs number_of_epochs` (default 200).

To specify a GPU device to use (e.g. 0 or 1), use `--device device_num`.

To evaluate on test data, use `--test test_prefix`, which will look for a file called `test_prefix.npz`

For example, to train a basic model on the IMDB corpus, with no metadata, use:

`python run_scholar.py data/imdb/processed/ -k 10 --test test --device 0`

### Using labels:

To train a classifier which tries to predict labels based on the latent representations, use `--labels label_name`. This will look for a file called `train.label_name.csv` in the input directory. For example, to run on the imdb corpus with a sentiment predictor, use:

`python run_scholar.py data/imdb/processed/ -k 10 --test test --labels sentiment --device 0`

In addition to topics, this will print and save a matix of label probabilities associated with each topic.

### Using covariates:

Alternatively, to treat the labels as observed covariates, and include topic-like deviations for each one, use `--topic-covars covar_name`. For example,

`python run_scholar.py data/imdb/processed/ -k 10 --test test --topic-covars sentiment --device 0`

In addition to topics learned in an unsupervised model, this will print and save a matrix of deviations associated with each covariate.

You can also include interactions between topics and covairates by adding `--interactions`.


### Using word vectors:

To initialize the encoder with pretrained word2vec vectors, download GoogleNews-vectors-negative300.bin.gz from the [word2vec website](https://code.google.com/archive/p/word2vec/) and use `--w2v path/to/file.bin.gz`


### Using regularization:

To regularize the model weights, separate regularization strengths can be specified for the topic weights, the weights for the covariate deviations, and the weights for the interaction terms. These can be specified by ` --l1-topics`, ` --l1-topic-covars`, and `--l1-interactions` respectively, using whatwever regularization strength is desired (e.g. 0.1).

### Additional options:

The default output directory is `output`, but this can be specified using `-o output_dir`

The model can also evaluate on a validation set during training using a random sample of the training data using `--dev-folds X`, where 1/X of the training data will be used for validation.

Finally, it is also possible to include covariates which influence the document representation prior (instead of representing topic-like deviations). This can be done using the `--prior-covars covar_name` option. Note that this feature is not discussed in the accompanying publication (see below).


## Output:

All files will be written to the specified output directory (default=`output`). This includes

- `topics.txt`: the top words in each topic
- `beta.npz`: saved np.array of topic-word weights (open with np.load())
- `beta_c.npz`: saved np.array of covariate-specific deviations (if covariates were provided)
- `beta_ci.npz`: saved np.array of interaction deviations (if covariates were provided)
- `topic_label_probs.npz`: saved np.array of per-topic label probabilities (if labels were provided)
- `theta.train.npz`: saved np.array of document-topic representations for the training instances
- `accuracy.train.txt`: accuracy on labels or categorical covariates on training data (if provided)
- the above two files for test and dev data (instead of train) if test data was provided
- `perplexity.test.txt`: an estimate of the perplexity on the test data (if provided)
- `vocab.json`: the vocabulary in order used in beta.npz and other files above


## TensorFlow vs. PyTorch:

The original implementation of this model was in TensorFlow, and it is the basis of the experiments in the paper. However, the Pytorch implementation is the default and is recommended, as it offers GPU support and some additional options.

For those who want to use it, the Tensorflow version can be run using `python run_scholar_tf.py` with a similar set of options. The requirements for this version are the same, except that tensorflow is required instead of pytorch. The latest version of tensorflow tested was 1.5.1.


## References

If you find this repo useful, please be sure to cite the following publication:

* Dallas Card, Chenhao Tan, and Noah A. Smith. Neural Models for Documents with Metadata. In *Proceedings of ACL* (2018). [[paper](https://www.cs.cmu.edu/~dcard/resources/ACL_2018_paper.pdf)] [[supplementary](https://www.cs.cmu.edu/~dcard/resources/ACL_2018_supplementary.pdf)] [[BibTeX](https://github.com/dallascard/scholar/blob/master/scholar.bib)]

