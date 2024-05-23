# Word Vecterization
### When any input feature comes from a finite (or a countably infinite) set, it is a discrete type. In NLP, we regard words, characters and letters as discrete symbols. Representing discrete types (e.g., words) as dense vectors is at the core of deep learning’s successes in NLP. When the discrete types are words, the dense vector representation is called a <font color='red'>word embedding</font>.


# 1. Collapsed One-Hot
**Description**: 

A unique index is assigned to each unique category (word or phrase), creating a binary vector of length equal to the total number of categories. Often used for simple text classification tasks where the order of words and context are not important. Often used for simple text classification tasks where the order of words and context are not important.

**Example:**

Vocabulary: {0: 'apple', 1: 'banana', 2: 'orange'}

Sentence: "apple banana banana orange apple"

Collapsed one-hot: [1,1,0]

