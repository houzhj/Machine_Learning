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


# Surname Classification
- **Introduction**: The goal is to classify surnames to their country of origin.
- **Data**: The surnames dataset includes 10,980 surnames from 18 different nationalities collected from different name sources on the internet. The top four classes account for 70% of the data: 27% are English, 22% are Russian, 15% are Arabic and 7% are Japanese.

  | **nationality**                | **nationality_index**           | **surname**     |
  |-----------------------------|--------------------------|--------------|
  | Arabic | 15 | Totah|
  | English | 12 | Foxall|
  | German | 9 | Neuman|
  | Japanese | 7 | Yamanaka|

- **Notebook**: [**Surname Classification using MLP**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/MAIN_surname_MLP.ipynb)
  - **Word vectorization**: Collapsed one-hot
- **Notebook**: [**Surname Classification using CNN**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/MAIN_surname_CNN.ipynb)
  - **Word vectorization**: Matrix of one-hots
- **Notebook**: [**Surname Classification using Elman RNN**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/MAIN_surname_Elman_RNN.ipynb)
  - **Word vectorization**: Token-IDs.

# Learning Embeddings with Continuous Bag of Words (CBOW) using the novel Frankenstein
- **Introduction**: The goal is to construct a classification task for the purpose of learning CBOW embeddings. The CBOW model is a multiclass classification task like a fill­-in­-the-­blank task (there is a sentence with a missing word, and the model’s job is to figure out what that word should be).
- **Data**: The raw Frankenstein text dataset includes 3,427 sentences. The data treatment steps enumerate the dataset as a sequence of windows by iterating over the list of tokens in each sentence and group them into windows of a specified window size. With window size = 3, the modeling data include 90,700 windows (rows).

  Raw text:

  "
  ......
  *You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have regarded with such evil forebodings.  I arrived here yesterday, and my first task is to assure my dear sister of my welfare and increasing confidence in the success of my undertaking......*
  "
  | **Context**                 | **Target**         |
  |-----------------------------|--------------------|
  | you will rejoice hear that no         | to       |
  | will rejoice to that no disaster      | that     |
  | rejoice to hear no disaster has       | hear     |
  | to hear that disaster has accompanied | no       |
  | hear that no has accompanied the      | disaster |

- **Word vectorization**: Token-IDs and trained embedding based on CBOW
- **Notebook**: [**Learning Embeddings with CBOW using Frankenstein**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Frankenstein/MAIN_frankenstein_Embedding.ipynb)


# News Classification Using Pre-trained Embeddings (Fine-Tuned GloVe)
- **Introduction**: The goal is to construct a classification task of predicting the category given the headline..
- **Data**: 
  | **category**                 | **Title**         |
  |-----------------------------|--------------------|
  | Boeing Expects Air Force Contract     | Business |
  | Mars Rovers Reports Published         | Sci/Tech |
  | Jackson has a tough match             | Sports   |
  | Why AIDS keeps spreading in Africa    | World    |

- **Notebook**: [**Fine-tuning Pre-trained Embeddings (GloVe) using the AG News dataset**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/AGNews/MAIN_AGnews_CNN_embedding.ipynb)

# Modeling Components
- Algorithms: [[**Perceptron, MLP**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Perceptron_ToyData/perceptron_classifiers.ipynb)], [[**CNN**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/convolutional_layer.ipynb)], [[**Elman RNN**]](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/MAIN_surname_RNN.ipynb)
- Visualization of the learning process: [[**Perceptron**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Perceptron_ToyData/perceptron_visualization.ipynb)]
- Word embedding:
  - [**Introduction**](https://github.com/houzhj/Machine_Learning/blob/main/README_word_vecterization.md)
  - Count-­based embedding methods: [[**Collapsed One-hot**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/class_Vectorizer_collapsed_one_hots.ipynb)], [[**One-hot Matrix**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Surname_Nationality/class_Vectorizer_matrix_of_one_hots.ipynb)], [[**Bag of Words**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Amazon_Reviews/amazon_linear_classifiers.ipynb)], [[**TFIDF**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/tfidf.ipynb)], [[**Token IDs**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/AGNews/class_Vectorizer.ipynb)]
  - Learning-­based embedding methods:
    - Training embedding: [[**Trained embedding layer using CBOW**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Frankenstein/Embedding_layer.ipynb)]
    - Pretrained GloVe embedding: [[**Application in an analogy task**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/AGNews/pretrained_embeddings_GloVe.ipynb)], [[**Fine-tuning GloVe**]()]
