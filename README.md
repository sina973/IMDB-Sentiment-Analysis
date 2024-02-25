# IMDB-Sentiment-Analysis
Benchmarked the Recurrent Neural Networks models with LSTM architecture and the BERT model for sentiment analysis

# Introduction
Sentiment analysis (also known as opinion mining) is a machine learning tool that analyzes text for polarity, from positive to negative. By training machine learning tools with examples of emotions in text, machines automatically learn how to detect sentiment without human input. Sentiment analysis using machine learning can help any business analyze public opinion, improve customer support, and automate tasks with fast turnarounds. Opinion mining saves not only businesses and companies money but also time. Opinion mining results will give us real actionable insights, helping us make the right decisions. For this project, we have chosen a labeled data set containing 2000 movie reviews divided by 1000 negative and 1000 positive reviews. Our goal is to compare different NLP (Natural Language Processing) techniques for sentiment analysis. Among many deep learning methods, we selected Recurrent Neural Network and BERT for implementation and aimed to compare the accuracy result between these two popular models.

# Dataset
Our dataset is available on the Cornell University website, and you can download it from the link below: <be />
www.cs.cornell.edu/people/pabo/movie-review-data/review polarity.tar.gz This dataset includes 2000 movie reviews from the IMDB website. 1000 are labeled negative comments, and the other half are labeled positive reviews. We used this dataset to compare and implement an NLP task using deep learning models such as RNN and BERT to figure out which one can do a better job when it comes to NLP tasks. In the following subsection, we will explain our basic steps and ideas.

## Data preprocessing
When we talk about data, we usually think of some large datasets with a huge number of rows and columns. While that is a likely scenario, it is not always the case — data could be in so many different forms: Structured Tables, Images, Audio files, Videos, etc. [5]
In any Machine Learning process, data preprocessing is that step in which the data gets transformed or Encoded to bring it to such a state that now the machine can easily parse it. In other words, the features of the data can now be easily interpreted by the algorithm. A dataset can be viewed as a collection of data objects, which are often also called records, points, vectors, patterns, events, cases, samples, observations, or entities. Data objects are described by a number of features, that capture the basic characteristics of an object, such as the mass of a physical object or the time at which an event occurred, etc. Features are often called variables, characteristics, fields, attributes, or dimensions. <br />
By preprocessing data, we make it easier to interpret and use. This process eliminates inconsistencies or duplicates in data, which can otherwise negatively affect a model’s accuracy. Data preprocessing also ensures that there are not any incorrect or missing values due to human error or bugs. In short, employing data preprocessing techniques makes the database more complete and accurate.

### Preprocessing Steps
The most important part of preprocessing our data is to try to turn a document into clean tokens; this step of the preprocessing task is called tokenization. Tokenization is breaking the raw text into small chunks. Tokenization breaks the raw text into words and sentences, which are called tokens. These tokens help in understanding the context or developing the model for the NLP. The tokenization helps in interpreting the meaning of the text by analyzing the sequence of the words. There are different methods and libraries available to perform tokenization. Natural Language Toolkit (NLTK), Gensim, and Keras are some of the libraries that can be used to accomplish the task, for our implementation we used NLTK in all of our steps mentioned below: <br />
- Split tokens by white space
- Remove punctuation from each token
- Remove remaining tokens that are not alphabetic
- Filter stop words
- Filter out short tokens <br />
White space tokenization is the simplest tokenization technique. We used this technique to break our text into meaningful words or tokens to be understandable for our machine. The next step is Removing punctuation or tokens which are not alphabetic. In this step, we are actually removing noise or unhelpful parts of data by removing punctuation and non-alphabetic tokens. Filtering the stop words (such as ”the”) and short tokens (such as ”a”, ”is”, etc.) is our final step of preprocessing the data. This step, like the last step, helps us to clean our data from any noises and reach a cleaner document to feed it to our machines. As our dataset is text, therefore, we could not perform normalization or other techniques for preprocessing datasets with a numerical nature. The above steps are sufficient preprocessing techniques for NLP that we performed on our data.

## Embedding
In natural language processing (NLP), word embedding is a term used for the representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of words such that words that are closer in the vector space are expected to be similar in their meaning. Word embeddings can be obtained using a set of language modeling and feature learning techniques where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves the mathematical embedding from space with many dimensions per word to a continuous vector space with a much lower dimension [4]. The initial embedding techniques dealt only with words. However, an embedding for each word in the set should be generated given a set of words. The simplest method is to encode the sequence of words provided one-hot vectors so that each word will be represented by 1 and other words by 0. While this method would effectively represent words and other simple text-processing tasks, it does not work on more complex ones, such as finding similar words. Basically, word embedding converts a word and identifies the semantics and syntaxes of a word to build a vector representation of this information. Some popular word embedding techniques include Word2Vec, GloVe, ELMo, FastText, etc. [3]

## Word2vec 
The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector. The vectors are chosen carefully such that a simple mathematical function (for example the cosine similarity between the vectors) indicates the level of semantic similarity between the words represented by those vectors [4].

## Dimension reduction
Dimensionality reduction, or dimension reduction, is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data, ideally close to its intrinsic dimension. Working in high-dimensional spaces can be undesirable for many reasons. First, raw data are often sparse as a consequence of the curse of dimensionality. Second, analyzing the data is usually computationally intractable (hard to control or deal with). Dimensionality reduction is common in fields that deal with large numbers of observations and/or large numbers of variables, such as signal processing, speech recognition, neuroinformatics, and bioinformatics [2]. Methods for dimensionality reduction are commonly divided into linear and nonlinear approaches. Approaches can also be divided into feature selection and feature extraction. Dimensionality reduction can be used for noise reduction, data visualization, cluster analysis, or as an intermediate step to facilitate other analyses [2]. When training our RNNs or BERT models, we must perform Word2vec, embedding, and dimension reduction on our data. Having a clean dataset to feed into our machines will result in a better outcome, which is the main goal of all these techniques and steps.

# Methods and definitions
In this section, we will discuss the general definitions and methods we used for our project.

## Train, validation, and test split
Before we decide which algorithm we should use, we should split our data into 2 or sometimes 3 parts: train, test, and validation (Figure 1). Machine learning algorithms first have to be trained on the data distribution available and then validated and tested before they can be deployed to deal with real-world data.

Figure 1: Train, test and validation splitting on a chart

### Training data
This is the part of the dataset on which our machine-learning algorithm is actually trained to build a model. The model tries to learn the dataset and its various characteristics and intricacies.
### Validation data
This is the part of the dataset which is used to validate our various model fits. In simpler words, we use validation data to choose and improve our model hyperparameters. This model does not learn the validation set but uses it to get to a better state of hyperparameters.
### Test data
This part of the data set is used to test our model hypothesis. It is left untouched and unseen until the model and hyperparameters are completely trained. Only after the model is applied to the test data can we get an accurate measure of how it would perform when deployed on real-world data.

## Recurrent neural networks
Recurrent Neural Networks(RNN) are a type of Neural Network where the output from the previous step is fed as an input to the current step. RNN is a type of artificial neural network that uses sequential data or time series data (Figure 2). These deep learning algorithms are commonly used for ordinal or temporal problems, such as language translation, natural language processing (NLP), speech recognition, and image captioning. they are incorporated into popular applications such as Siri, voice search, and Google Translate [6]. <br />
Figure 2: RNN model for Natural Language Processing  
Like feed-forward and convolutional neural networks (CNNs), RNNs utilize training data to learn. They are distinguished by their “memory” as they take information from prior inputs to influence the current input and output. While traditional deep neural networks assume that inputs and outputs are independent of each other, the output of recurrent neural networks depends on the prior elements within the sequence. While future events would also be helpful in determining the output of a given sequence, unidirectional recurrent neural networks cannot account for these events in their predictions. <br />

RNNs are mainly used for: <br />
- Sequence classification: Sentiment Classification and Video Classification
- Sequence Labelling: Part of speech tagging and Named entity recognition
- Sequence Generation: Machine translation and Transliteration <br />
Also, there are different types of RNNs: <br />
- One to one
- One to many
- Many to one
- Many to many
- 
### Common activation function of RNNs
An activation function determines whether a neuron should be activated (Figure 3). The nonlinear functions typically convert the output of a given neuron to a value between 0 and 1 or -1 and 1. Some of the most commonly used functions are defined as follows:
- Sigmoid
- Tanh
- Relu <br />
Figure 3: Relu (left) and Sigmoid (right) are represented with their formula

## Long short-term memory
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning (DL) [7]. Unlike standard feed-forward neural networks, LSTM has feedback connections. It can process not only single data points (such as images) but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented and connected handwriting recognition, speech recognition, and anomaly detection in network traffic or intrusion detection systems (IDSs) [7]. <br />

A common LSTM unit is composed of a cell, an input gate, an output gate, and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell [4]. For our project, we decided to implement LSTM as an architecture for our RNN.

## BERT model
BERT’s key technical innovation is applying the bidirectional training of Transformer, a popular attention model, to language modeling. This is in contrast to previous efforts which looked at a text sequence either from left to right or combined– left-to-right and right-to-left training (Figure 4) [7]. <br />
Figure 4: BERT language model <br />
BERT makes use of a transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, the transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary. As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that it’s non-directional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).

# Implementation
In this part, we talk about the implementation process of our project.

## Preprocessing
The first part of our project is to preprocess the dataset that we have. So we start with preprocessing tasks such as splitting, tokenization, removing stop words, and punctuation. We can use the Natural Language Toolkit (NLTK) for this type of task. The NLTK is a platform used for building Python programs that work with human language data for application in statistical natural language processing (NLP). It contains text-processing libraries for tokenization, parsing, classification, stemming, tagging, and semantic reasoning. It also includes graphical demonstrations and a sample data set and is accompanied by a cookbook and a book that explains the principles behind the underlying language processing tasks that NLTK supports. After implementing the preprocessing program, we can read the data set using the program, and our result would be a clean file named vocab.txt. This is the file we have to use alongside the negative and positive reviews for the other step of the project, which is comparing the BERT and RNN accuracy.

## RNN
Our next step is to implement the RNN train model. For this part of the project, we use Keras. Keras is a high-level, deep-learning API developed by Google for implementing neural networks. It is written in Python and is used to make the implementation of neural networks easy. It also supports multiple back-end neural network computations. Keras allows you to switch between different back-ends. There are different frameworks supported by Keras, but TensorFlow has adopted Keras as its official high-level API. Keras is embedded in TensorFlow and can be used to perform deep learning fast as it provides inbuilt modules for all neural network computations. Another package that we use is Numpy. NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more. After loading documents and loading the generating voacb.txt for our RNN model. We figured out that we have sentences with different lengths. Therefore, we used pas sequence in our implementation. Sequence padding( pad sequences() function) can be used, and the Keras deep learning library can be used to pad variable length sequences. So we can ensure we have an array with equal length for our network. <br />
Recurrent means the output at the current time step becomes the input to the next time step. At each element of the sequence, our model considers not just the current input but what it remembers about the preceding elements. This memory allows the network to learn long-term dependencies in a sequence, which means it can take the entire context into account when making a prediction. At the heart of our RNN model is a layer made of memory cells. One of the most popular cells at this time is Long Short-Term Memory (LSTM) (Figure 5). LSTM maintains a cell state and carries out the process of ensuring that the signal (information in the form of a gradient) is not lost as the sequence is processed. At each step, the LSTM considers the current word, the carry, and the cell state. The LSTM has 3 different gates and weight vectors: there is a “forget” gate for discarding irrelevant information; an “input” gate for handling the current input, and an “output” gate for producing predictions at each time step. Then, the function of each cell element is ultimately decided by the parameters (weights) that are learned during training. <br />
Figure 5: Anatomy of LSTM

## BERT
The next and last part of our project is our BERT program (Figure 6). For this part, we need the tensor flow library like the previous part. The noteworthy point for this part is that we have 20% of our dataset for validation. Moreover, the negative sentences are labeled with number 0, and positive sentences are labeled with number 1. In the next step, we use a sentence to make sure the BERT is working properly. <br />
Figure 6: BERT model layers

# Results and evaluation
Our results show that, however, the RNN model with LSTM architecture can be faster (Table 2), but the BERT model has a more accurate result (Table 1). We believe a better result in an NLP task depends on many factors, such as the type of the data, the size of the data, and also the goal of the project. For us, we understood the BERT model can work more accurately with less loss (0.133) compared to the RNN with LSTM architecture (0.423). However, it might be possible that if we have another step in our project such as clustering or labeling, the result would be different. Additionally, the dataset we used can be considered as an small data set, therefore, if we use these model with a big dataset for NLP task, the result of our work could be different as well(Table 1).


Table 1: Accuracy and loss comparison of our models
Accuracy Loss
RNN with LSTM 84.5% 0.423
BERT 94.7% 0.133%
Table 2: A Table with Models
Model Training Time (Sec.) Number of Epochs
RNN with LSTM 725 5
BERT 18,263 20
<br />

In this part, it is necessary to mention that due to the limited time, only two models could be implemented. Another comparison between the CNN model and RNN could be performed, as well as between BERT and CNN. The dataset is also important for this project. We believe if we had more time, we could work on bigger data sets, such as all IMDB website reviews, rather than only 2000 of them, we could have had a more accurate comparison for NLP and deep learning.

# Conclusion
In this project, we performed sentiment analysis using two different kinds of machine learning techniques, the Recurrent Neural Networks (RNN) model with Long Short Term Memory (LSTM) and BERT (a transformer approach) and compared them in accuracy and time consumption. Sentiment analysis using machine learning can help any business analyze public opinion, improve customer support, and automate tasks with fast turnarounds. For this project, we have chosen a labeled data set containing 2000 movie reviews divided by 1000 negative and 1000 positive reviews. Second, we preprocessed the data using the Natural Language Toolkit (NLTK) to turn our dataset into clean tokens. Finally, we implemented the RNN with LSTM architecture and the BERT model to see which model could do better and generate a better result. As a result, we observed that the BERT model performed a lot better than the RNN model in terms of accuracy, however, it is a lot more time consuming. Furthermore, we believe if we could work on bigger data sets, such as all IMDB websites, more accurate comparisons would be possible.

# Installation and build guide
Preprocessing is done with NLTK, so make sure that the NLTK package is installed on your computer before you begin. Also for testing the program make sure to use the relative path instead of absolute path. For the RNN program, you need to Install TensorFlow and Keras library first and then run the program. Also, for the BERT model, make sure to upload the dataset to your Google Drive and use the relative path for addressing instead of an absolute path.

# References
[1] Google developers, https://developers.google.com/machine-learning/crash-course/embeddings/videolecture. <br />
[2] Wikipedia: the free encyclopedia, dimension reduction. https://machinelearningmastery.com/datapreparation. <br />
[3] Wikipedia: the free encyclopedia, embedding. https://machinelearningmastery.com/data-preparation. <br />
[4] Wikipedia: the free encyclopedia, word2vec. https://machinelearningmastery.com/data-preparation. <br />
[5] Jason Brownlee. Data prepration for machine learning book. https://machinelearningmastery.com/datapreparation. <br />
[6] Bo Pang, Lillian Lee, and Shivakumar Vaithyanathan. Thumbs up? sentiment classification using machine <br />
learning techniques. page 79–86, 2002. <br />
[7] Shiliang Sun, Chen Luo, and Junyu Chen. A review of natural language processing techniques for opinion <br />
mining systems. Information Fusion, 36:10–25, 2017. <br />
