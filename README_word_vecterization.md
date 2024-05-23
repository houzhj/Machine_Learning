# Word Vecterization
### When any input feature comes from a finite (or a countably infinite) set, it is a discrete type. In NLP, we regard words, characters and letters as discrete symbols. Representing discrete types (e.g., words) as dense vectors is at the core of deep learning’s successes in NLP. When the discrete types are words, the dense vector representation is called a <font color='red'>word embedding</font>.


# 1. Collapsed One-Hot
## [Code Link](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/class_Vectorizer_collapsed_one_hots.ipynb)

**Description**: 

A unique index is assigned to each unique category (word or phrase), creating a binary vector of length equal to the total number of categories. Often used for simple text classification tasks where the order of words and context are not important. Often used for simple text classification tasks where the order of words and context are not important.

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

Bag of words: [2,2,1], shape = [1, The size of the vocabulary]
- In the vector, the i-th number represents the number of times the corresponding word in the vocabulary appears in the sentence.
- 











