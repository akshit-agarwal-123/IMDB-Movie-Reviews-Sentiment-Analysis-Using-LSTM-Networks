# IMDB Movie Reviews Sentiment Analysis using LSTM (Long-Short Term Memory) Networks
This project performs sentiment analysis by classification an IMDB movie review into positive (1) and negative (0)

### Description of the Dataset
The core dataset contains 50,000 reviews split evenly into 25,000 training and 25,000 testing sets. 
The overall distribution of labels is balanced. You can learn more about this dataset:   
[http://ai.stanford.edu/~amaas/data/sentiment/]

### Features
- **Review**: IMDB Moview Review Text.
- **Sentiment**: 1 (positive), 0 (negative)

### Dependencies
* [python](https://www.python.org/) - Programming Language
* [tensorflow](https://www.tensorflow.org/) - TensorFlow is an open-source machine learning library for research and production
* [keras](https://keras.io/) - Keras is a high-level neural networks API
* [sklearn](http://scikit-learn.org/stable/documentation.html) - Scikit-learn is a free software machine learning library for the Python 
* [numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing
* [pandas](https://pandas.pydata.org/) - Pandas is a software library used for data manipulation and analysis
* [pprint](https://python.readthedocs.io/en/stable/library/pprint.html#module-pprint) - The pprint module provides a capability to “pretty-print” arbitrary Python data structures in a form which can be used as input to the interpreter.


### Steps of Appylying the Machine Algorithm
**Step 1**. Get data using `get_reviews_data` <br/>
 - loops through multiple text files, imports the reviews into a list, cleans, and stores it into pandas dataframe

**Step 2**. Prepares the data for machine learning using `preproces_text_data` <br/>
 - tokenizes the Textual data
 - creates a padded sequence of numbers representing the textual review data
 - returns a numpy array

**Step 3**. Create a Baseline LSTM (Long Short Term Memory) Neural Network using `create_baseline_model` method <br/>
 - be careful when designing the model. Here input_dim of the emmbedding layer should be greater than the size of input vocabulary.
    The size of the vocabulary can be calculated by `print(tokenizer.word_index)`
 - The output layer will be `sigmoid` layer consisting of 1 neuron (for binary classification) 
 - This will output fuzzy values ranging between 0 and 1 depicting the class probabilities.
 - `compile` the model using loss function, optimizer, and metric: Here, we have chosen `binary_crossentropy` as the loss function since we are performing classification task. Optimiser is `adam` and metric is `accuracy`
 
**Step 4**. `split_and_fit_dataset` <br/>
 - `train_test_split` splits the data in the ratio of test-size to the total dataset, i.e. test_size = 0.2 implies train_size = 0.8 
 - `fit` the training data `(x_train, y_train)` in the model using `model.fit` method 
 
**Step 5** `evaluate` the model performance on test set using `model.evaluate` method.
 - print the `test_acc` and `test_loss`. This provides the model performance on test_data
 
**Step 5**. Iteratively tune the model hyperparameters <br/>
 - Tune the hyperparameters of the network such as learning rate, activation functions, optimizer, depth of the network (number of hidden layers), width of layers (no. of neurons in each layer), DropOut layers, batch size, epochs, L1 or Ll2 regularization. This is basically the part where the magic happens.

### Hyperparameters
Activation function: `tanh`  
Optimizer: `adam`   
learning rate,`α`: `0.01` (default)  
dropout: `0.2`, `0.9`  
train-test: `0.8/0.2`  
batch_size: `32`  
epochs: `50`  
loss function: `binary_crossentropy`

### Results
**Accuracy**: 55.15%  

### Author
[Akshit Agarwal](https://www.linkedin.com/in/akshit-agarwal93/)

### Citations
 - Maas, A., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies (pp. 142-150).
 - http://ai.stanford.edu/~amaas/data/sentiment/
 - https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
