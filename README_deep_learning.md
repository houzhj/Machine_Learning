# Sentiment Analysis using Amazon reviews
- **Introduction**: The goal is to design a classifier for sentiment analysis.
- **Data**: Amazon reviews
  
| **Text**                | **Sentiment**           |
|-----------------------------|--------------------------|
| The chips are okay Not near as flavorful as the regular blue chips. Nice size bag for a family. |  -1  |
| I really enjoyed this flavor, this has a very nice subtle coconut flavor that is not too sweet.  It's a hit in our household, I give them to my grand kids every time they come over and needless to say they keep coming back!       | 1 |

- **Word vectorization**: Bag of words (binary or count)
- **Notebook**: [**Build several linear classifiers based on three algorithms**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Amazon_Reviews/amazon_linear_classifiers.ipynb)
  - Perceptron, Average Perceptron, Pegasos.


# Visualization of training Single Layer Perceptron and Multilayer Perceptron using Toy Data
- **Introduction**: The goal is to design a binary classifier using a generated toy dataset (classifying two-dimensional points into one of two classes).
- **Data**: Generated toy datasets, two dimensions
- **Notebook**: [**Perceptron and Multiple Layer Perceptron Models using PyTorch**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Perceptron_ToyData/MAIN_perceptron_mlp.ipynb)
  - Model: Perceptron and Multilayer Perceptron
  - Visualization of the training - changes in the loss and the hyperplain (an [application](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Perceptron_ToyData/perceptron_visualization.ipynb) of Axes.contour)
<img width="1176" alt="image" src="https://github.com/houzhj/Machine_Learning/assets/33500622/cf8c9a6a-a2bb-41c0-980a-f5727108767a">



# Sentiment Analysis using Yelp Reviews
- **Introduction**: The goal is to to classify whether restaurant reviews on Yelp are positive or negative using a perceptron.

- **Data**: The Yelp dataset includes 56,000 reviews. This is a sample of the dataset created by Zhang, Zhao, and Lecun (2015).
  
| **Text**                | **Sentiment**           |
|-----------------------------|--------------------------|
| Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC -- good doctor, terrible staff.  It seems that his staff simply never answers the phone.  It usually takes 2 hours of repeated calling to get an answer.  Who has time for that or wants to deal with it?  I have run into this problem with many other doctors and I just don't get it.  You have office workers, you have patients with medical needs, why isn't anyone answering the phone?  It's incomprehensible and not work the aggravation.  It's with regret that I feel that I have to give Dr. Goldberg 2 stars. | 1  |
| Been going to Dr. Goldberg for over 10 years. I think I was one of his 1st patients when he started at MHMG. He's been great over the years and is really all about the big picture. It is because of him, not my now former gyn Dr. Markoff, that I found out I have fibroids. He explores all options with you and is very patient and understanding. He doesn't judge and asks all the right questions. Very thorough and wants to be kept in the loop on every aspect of your medical health and your life.       | 2 |
- **Word vectorization**: Collapsed one-hot
- **Notebook**: [**Perceptron classifier for sentimental analysis using Yelp reviews data**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Yelp_Reviews/MAIN_yelp_perceptron.ipynb)



s

s

s


s

s

ss







# Perceptron-based Classifier for Sentiment Analysis (Amazon reviews)
- **Introduction**:
  - The training/testing datasets contains reviews written by Amazon customers for various food products. The reviews have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively.
  - The goal is to design a classifier for sentiment analysis.
- **Notebook**: [**Build several linear classifiers based on three algorithms**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Amazon_Reviews/amazon_linear_classifiers.ipynb)
  - Perceptron, Average Perceptron, Pegasos.

# Single Layer Perceptron and Multilayer Perceptron for a Binary Classification (Toy Data)
- **Introduction**:
  - The goal is to design a binary classifier using a generated toy dataset (classifying two-dimensional points into one of two classes).
  - Using a preceptron-based algorithm to discriminate the points of one class from the other.
  - Compared the performance of perceptrons with different numbers of hidden layers (other hyperparameters, such as learning rate, size of hidden layers are specified, rather than tuned), in scenarios involving linearly separable and not linearly separable toy data.
- **Notebook**: [**Perceptron and Multiple Layer Perceptron Models using PyTorch**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Perceptron_ToyData/MAIN_perceptron_mlp.ipynb)
  - Model: Perceptron and Multilayer Perceptron
  - Visualization of the training - changes in the loss and the hyperplain (an [application](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Perceptron_ToyData/perceptron_visualization.ipynb) of Axes.contour)

# Perceptron Classifier for Sentiment Analysis (Yelp Reviews)
- **Introduction**:
  - The goal is to to classify whether restaurant reviews on Yelp are positive or negative using a perceptron.
  - The Yelp dataset includes 56,000 reviews. This is a sample of the dataset created by Zhang, Zhao, and Lecun (2015).
- **Notebook**: [**Perceptron classifier for sentimental analysis using Yelp reviews data**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Yelp_Reviews/MAIN_yelp_perceptron.ipynb)
  - Model: Perceptron
  - NLP data treatment: one-hot encoding with an "unknown" token and restrictions to infrequent tokens. 

# Surname Classification
- **Introduction**:
  - The goal is to classify surnames to their country of origin.
  - The surnames dataset includes 10,980 surnames from 18 different nationalities collected from different name sources on the internet. The top four classes account for 70% of the data: 27% are English, 22% are Russian, 15% are Arabic and 7% are Japanese. 
- **Notebook**: [**Surname Classification using MLP**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/MAIN_surname_MLP.ipynb)
  - Model: MLP (Multilayer Perceptron)
  - NLP data treatment: One-hot encoding with an "unknown" token.
  - A weight is assigned to each surname class that is inversely proportional to its frequency.
- **Notebook**: [**Surname Classification using CNN**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/MAIN_surname_CNN.ipynb)
  - Model: CNN (Convolutional neural network)
  - NLP data treatment: Matrix of one-hots with an "unknown" token.
  - A weight is assigned to each surname class that is inversely proportional to its frequency.
- **Notebook**: [**Surname Classification using RNN**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/MAIN_surname_RNN.ipynb)
  - Model: Elman RNN (Recurrent neural network)
  - NLP data treatment: Vecterization based on token-IDs.
  - A weight is assigned to each surname class that is inversely proportional to its frequency.
  
# Learning Embeddings with Continuous Bag of Words (CBOW) using the novel Frankenstein
- **Introduction**:
  - The goal is to construct a classification task for the purpose of learning CBOW embeddings. The CBOW model is a multiclass classification task like a fill­in­the­blank task (there is a sentence with a missing word, and the model’s job is to figure out what that word should be).
  - The raw Frankenstein text dataset includes 3,427 sentences. The data treatment steps enumerate the dataset as a sequence of windows by iterating over the list of tokens in each sentence and group them into windows of a specified window size. With window size = 3, the modeling data include 90,700 windows (rows).
- **Notebook**: [**Learning Embeddings with CBOW using Frankenstein**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Frankenstein/MAIN_frankenstein_Embedding.ipynb)
  - Model: CBOW with embedding. The model has three essential steps:
    - Indices representing the words of the context are used with an nn.Embedding(.) layer to create vectors for each word in the context.
    - Combine the vectors in some way such that it captures the overall context. In the example below, we sum over the vectors. However, other options include taking the max, the average, or even using a Multilayer Perceptron on top.
    - The context vector is used with a nn.Linear(.) layer to compute a prediction vector. This prediction vector is a probability distribution over the entire vocabulary. The largest (most probable) value in the prediction vector indicates the likely prediction for the target word—the center word missing from the context.
  - NLP data treatment: learned-based word embedding

# Fine-tuning Pre-trained Embeddings (GloVe) using the AG News dataset 
- **Introduction**:
  - The goal is to construct a classification task of predicting the category given the headline..
  - The raw Frankenstein text dataset includes 3,427 sentences. The data treatment steps enumerate the dataset as a sequence of windows by iterating over the list of tokens in each sentence and group them into windows of a specified window size. With window size = 3, the modeling data include 90,700 windows (rows).
- **Notebook**: [**Fine-tuning Pre-trained Embeddings (GloVe) using the AG News dataset**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/AGNews/MAIN_AGnews_CNN_embedding.ipynb)
  - Model: CBOW with embedding. The model has three essential steps:
    - Indices representing the words of the context are used with an nn.Embedding(.) layer to create vectors for each word in the context.
    - Combine the vectors in some way such that it captures the overall context. In the example below, we sum over the vectors. However, other options include taking the max, the average, or even using a Multilayer Perceptron on top.
    - The context vector is used with a nn.Linear(.) layer to compute a prediction vector. This prediction vector is a probability distribution over the entire vocabulary. The largest (most probable) value in the prediction vector indicates the likely prediction for the target word—the center word missing from the context.
  - NLP data treatment: learned-based word embedding


# Modeling Components
- Algorithms: [[**Perceptron, MLP**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Perceptron_ToyData/perceptron_classifiers.ipynb)], [[**CNN**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/convolutional_layer.ipynb)]
- Visualization of the learning process: [[**Perceptron**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Perceptron_ToyData/perceptron_visualization.ipynb)]
- Word embedding:
  - [**Introduction**](https://github.com/houzhj/Machine_Learning/blob/main/README_word_vecterization.md)
  - Count-­based embedding methods: [[**Collapsed One-hot**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/class_Vectorizer_collapsed_one_hots.ipynb)], [[**One-hot Matrix**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/class_Vectorizer_matrix_of_one_hots.ipynb)], [[**Bag of Words**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Amazon_Reviews/amazon_linear_classifiers.ipynb)], [[**TFIDF**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/tfidf.ipynb)], [[**Token IDs**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/AGNews/class_Vectorizer.ipynb)]
  - Learning-­based embedding methods:
    - Training embedding: [[**Trained embedding layer using CBOW**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Frankenstein/Embedding_layer.ipynb)]
    - Pretrained GloVe embedding: [[**Application in an analogy task**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/AGNews/pretrained_embeddings_GloVe.ipynb)], [[**Fine-tuning GloVe**]()]
