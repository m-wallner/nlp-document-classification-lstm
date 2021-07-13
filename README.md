# Natural Language Processing: Document Classification Using LSTM

The goal of this project was to create a complete NLP pipeline for document classification using the pre-trained vectors of a word embedding model combined with an LSTM model.

## Preprocessing
First, the data set is loaded, all documents get tokenized, and a dictionary of vocabularies is created from the tokenized text, with tokens with low frequencies being excluded. Next, a lookup for the embeddings of all the words in the dictionary is created – this is an embedding matrix that maps the ID of each word to the respective pre-trained vector from the embedding model, which is GloVe with a vector length of 300 in this case. Words that are not found in the embedding model are replaced by randomly initialized vectors. The preprocessed and embedded data is then pickled to save time in future runs, and a PyTorch Dataset object is created for the training, validation and test set for optimized data loading during training and inference time.

## Training the model
The words in each tokenized document in a batch are turned into word IDs based on the embedding matrix and the respective pretrained word embeddings are fetched. Next, the LSTM model calculates the hidden states of each given document and uses the last hidden state as document embedding which is sent through a dropout layer with a dropout probability of 0.5, a linear decoder layer, and finally a Softmax layer to predict the probability distribution over the output classes.
This model is then used for further experiments, with each experiment just applying a single change to the baseline architecture: The number of hidden dimensions is increased from 128 to 512, dropout is increased from 0.5 to 0.8, and finally a bidirectional LSTM is used, with each experiment reporting the results in terms of document classification accuracy.
