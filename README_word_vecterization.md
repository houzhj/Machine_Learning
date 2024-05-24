# Word Vecterization
### When any input feature comes from a finite (or a countably infinite) set, it is a discrete type. In NLP, we regard words, characters and letters as discrete symbols. Representing discrete types (e.g., words) as dense vectors is at the core of deep learning’s successes in NLP. When the discrete types are words, the dense vector representation is called a <font color='red'>word embedding</font>.


# 1. Collapsed One-Hot
## [Code Link](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/class_Vectorizer_collapsed_one_hots.ipynb)

**Description**: 

Assume that the number of different tokens in the vocabulary is N (the len(vocab)) and the token indices range from 0 to N − 1. A unique index is assigned to each unique category (word or phrase), creating a binary vector of length equal to N. Often used for simple text classification tasks where the order of words and context are not important. 

Often used for simple text classification tasks where the order of words and context are not important.

Limitations of this method: (1) sparseness, n_unique_tokens in a text sample << n_unique_tokens in a vocabulary; (2) discarding the order of the characters' appearance; (3) the one-hot word vectors cannot accurately express the similarity between different words, such as the cosine similarity that is commonly use. Since the cosine similarity between the one-hot vectors of any two different words is 0, it is difficult to use the one-hot vector to accurately represent the similarity between multiple different words. Denote $x$ and $y$ to be the vector for two different words, their cosine similarities are the cosines of the angles between them
$$\frac{x^Ty}{|x||y|}\in (-1,1)$$

**Example:**

Vocabulary: {0: 'apple', 1: 'banana', 2: 'orange'}

Sentence: "apple banana banana orange apple"

Collapsed one-hot: [1,1,0], shape = [1, The size of the vocabulary]
- 0s for all words that are not in the sentence
- 1s for all words that are in the sentence


# 2. One-hot Matrix
## [Code Link](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/class_Vectorizer_matrix_of_one_hots.ipynb)

**Description**: 

A 2D matrix where each row corresponds to a word in the document, and each row is a one-hot encoded vector representing that word. Used in more complex NLP tasks where word order and context are important, such as sequence models and neural networks.

**Example:**

Vocabulary: {0: 'apple', 1: 'banana', 2: 'orange'}

Sentence: "apple banana banana orange apple"

Collapsed one-hot: [[1,0,0,0,1,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0]], shape = [the size of the vocabulary, max_length = 10]
- Given a specific row (correspoingind a word in a sentence), 0s for all except the position of that that in the vocabulary. 
- The purpose of specifying a max_length is to ensure that all sentences have the same length when converted into vectors. This allows sentences of different lengths to have consistent dimensions during processing, facilitating subsequent model handling. Padding is a common technique: for sentences shorter than the max length, special padding symbols (usually 0 or a specified fill value) are used to extend them to the max length.

# 3. Bag of Words
## [Code Link](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Amazon_Reviews/amazon_linear_classifiers.ipynb)

**Description**: 

By converting text into a set of unique words and counting their occurrences, it creates a word frequency vector. BoW is often used as a baseline method for text representation before moving on to more complex methods like TF-IDF, word embeddings (Word2Vec, GloVe), or contextual embeddings (BERT, GPT).

**Example:**

Vocabulary: {0: 'apple', 1: 'banana', 2: 'orange'}

Sentence: "apple banana banana orange apple"

Bag of words: [2,2,1], shape = [1, the size of the vocabulary]
- In the vector, the i-th number represents the number of times the corresponding word in the vocabulary appears in the sentence.

# 4. Token IDs
## [Code Link](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/AGNews/class_Vectorizer.ipynb)

**Description**: 

Each unique token in the final vocabulary is assigned a unique numerical index or ID. Token IDs are a straightforward numerical representation of tokens. It is, in fact, a basic form of vectorization. They do not capture any deeper relationships or patterns between the tokens.

Often used in CNN models, sequence models (RNN, LSTM), transformer models, and etc.

**Example:**

Vocabulary: {0: 'apple', 1: 'banana', 2: 'orange'}

Sentence: "apple banana banana orange apple"

Token IDs: [0,1,1,2,0], shape = [1, the size of the input text]
- In the vector, the i-th number represents the token ID for the i-th word.
- If a max_length is used (i.e., max_length = 10), special padding symbols (usually 0 or a specified fill value) are used to extend them to the max length so that all sentences have the same length when converted into vectors. In this case the word vector is [0,1,1,2,0,0,0,0,0,0].



# 5. TF-IDF
## [Code Link](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/tfidf.ipynb)

**Description**: 

TFIDF is a measure of importance of a word to a document in a collection or corpus, adjusted for the fact that some words appear more frequently in general.
- Term Frequency (TF)
- Inverse Document Frequency (IDF)

**Example:**

Corpus (include four sentences): 
['this is the first document',
 'this document is the second document',
 'and this is the third one',
 'is this the first document',]

Vocabulary = ['and','document','first','is','one','second','the','third','this']

For word "document":
- In sentence 1: TF = 1, IDF = 1.223, TFIDF=1.223
- In sentence 2: TF = 2, IDF = 1.223, TFIDF=2.446
- In sentence 3: TF = 0, IDF = 1.223, TFIDF=0.000
- In sentence 4: TF = 1, IDF = 1.223, TFIDF=1.223

TFIDF matrix: below, shape = [the size of the corpus, the size of the vocabulary]
<img width="971" alt="image" src="https://github.com/houzhj/Machine_Learning/assets/33500622/83d5794a-a8a6-47c7-ac58-b2eee11fcaa3">


# 6. Word2vec

## [Code Link (Training word embedding with a CBOW task)](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Frankenstein/MAIN_frankenstein_Embedding.ipynb)
## [Code Link (Fine-tuning pretrained word embedding)](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/AGNews/MAIN_AGnews_CNN_embedding.ipynb)

**Description**: 

Word2vec represents each word with a fixed-length vector and uses these vectors to better indicate the similarity and analogy rela- tionships between different words. The word embedding methods train with just words (i.e., unlabeled data), but in a supervised fashion. This is possible by constructing auxiliary supervised tasks in which the data is implicitly labeled, with the intuition that a representation that is optimized to solve the auxiliary task will capture many statistical and linguistic properties of the text corpus in order to be generally useful.

The choice of the auxiliary task depends on the intuition of the algorithm designer and the computational expense. Examples include Continuous Bag­of­ Words (CBOW), Skipgrams, and so on. 
- The skip-gram model assumes that a word can be used to generate the words that surround it in a text sequence. 
- The continuous bag of words (CBOW) model assumes that the central target word is generated based on the context words before and after it in the text sequence.

In the initialization the embedding matrix of an Embedding layer, there are two options: 1) Load pre-trained embedding (GloVe) from disk and use it and fine-tune it for the task at hand; 2) Random initial weights. 

**Example:**

[GloVe](https://nlp.stanford.edu/projects/glove/)

Developed by Stanford, is an algorithm used for generating word embeddings, based on the Global Vectors model. GloVe (Global Vectors for Word Representation) is a word embedding model that represents words as high-dimensional vectors and encodes the semantic information of words into vector space.

glove.6B.zip is a pre-trained version of the Stanford GloVe model. It is a word embedding model that has been trained on large-scale corpora and can be directly used for natural language processing (NLP) tasks such as word similarity calculation, text classification, sentiment analysis, etc. Pre-trained GloVe models typically include vector representations of millions of words, with each word corresponding to a high-dimensional vector. It has 6B tokens and a vocabulary of 400K words.



