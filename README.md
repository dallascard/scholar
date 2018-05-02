**SCHOLAR**: **S**parse **C**ontextual **H**idden and **O**bserved **L**anguage **A**utoencode**R**.

Scholar is a tool for modeling documents with metadata. 


### Requirements:

- python3
- numpy
- scipy
- tensorflow
- pandas
- gensim
- scikit-learn


### Installation:

It is recommended that you install the requirements using Anaconda. Specifically, you can use the following commands to create a new environment, activate it, and install the necessary packages:

`conda create -n scholar python=3`

`source activate scholar`

`conda install numpy scipy tensorflow pandas gensim scikit-learn`

Alternatively, you can create a stable environment using:

`conda create -n scholar --file requirements.txt`

and activate it using

`source activate scholar`

Once the necessary packages are installed, there is no need to compile or install this repo.


### Basic usage:

The model is in the file `scholar_tf.py`. To run a basic model with no metadata, use:

`python run_scholar_tf.py input_directory train_file_prefix -k number_of_topics`

Please refer to [tutorial.txt](https://github.com/dallascard/scholar/blob/master/tutorial.txt) for more details and examples.


### Output:

All files will be written to the specified output directory (default=`output`). This includes

- `topics.txt`: the top words in each topic
- `beta.npz`: saved np.array of topic-word weights (open with np.load())
- `beta_c.npz`: saved np.array of covariate-specific deviations (if covariates were provided)
- `topic_label_probs.npz`: saved np.array of per-topic label probabilities (if labels were provided)
- `theta.train.npz`: saved np.array of document-topic representations for the training instances
- `accuracy.train.txt`: accuracy on labels or categorical covariates on training data (if provided)
- the above two files for test and dev data (instead of train) if test data was provided
- `perplexity.test.txt`: an estimate of the perplexity on the test data (if provided)
- `vocab.json`: the vocabulary in order usd in beta.npz and other files above


### TensorFlow vs. PyTorch:

The original implementation of this model was in TensorFlow, and it is the basis of the experiments in the paper. A partial PyTorch implementation has also been provided here. It is functional (and may be faster, and easier to understand than the tensorflow version), but does not yet support regularization or interactions, and will not produce identical results.

The requirements for the PyTorch version are the same as above, except that pytorch (v0.3.1) is required, and tensorflow is not.


### References

If you find this repo useful, please be sure to cite the following publication:

* Dallas Card, Chenhao Tan, and Noah A. Smith. Neural Models for Documents with Metadata. In *Proceedings of ACL* (2018). [[BibTeX](https://github.com/dallascard/scholar/blob/master/scholar.bib)] [preprint] [supplementary]

