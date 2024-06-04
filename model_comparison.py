import warnings
warnings.filterwarnings("ignore")
import ftfy
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import pandas as pd
import re
from math import exp
from numpy import sign
from collections import Counter
from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer

from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

# Reproducibility
np.random.seed(1234)

DEPRES_NROWS = 4000  # number of rows to read from DEPRESSIVE_TWEETS_CSV
RANDOM_NROWS = 4000 # number of rows to read from RANDOM_TWEETS_CSV
MAX_SEQUENCE_LENGTH = 140 # Max tweet size
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
TRAIN_SPLIT = 0.6
TEST_SPLIT = 0.2
LEARNING_RATE = 0.1
EPOCHS= 10

df = 'depressed.csv'
RANDOM_TWEETS_CSV = 'random.csv'
depressive_tweets_df = pd.read_csv(df, sep = '|', header = None, usecols = range(0,9), nrows = DEPRES_NROWS)
random_tweets_df = pd.read_csv(RANDOM_TWEETS_CSV, encoding = "ISO-8859-1", usecols = range(0,4), nrows = RANDOM_NROWS)
#Embedding_file is the file which taken from this link https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
print (depressive_tweets_df.head())



word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Expand Contraction
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            # remove hashtag, @mention, emoji and image URLs
            tweet = ' '.join(
                re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())

            # fix weirdly encoded texts
            tweet = ftfy.fix_text(tweet)

            # expand contraction
            tweet = expandContractions(tweet)

            # remove punctuation
            tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

            # stop words
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(tweet)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            tweet = ' '.join(filtered_sentence)

            # stemming words
            tweet = PorterStemmer().stem(tweet)

            cleaned_tweets.append(tweet)

    return cleaned_tweets
  
  
depressive_tweets_arr = [x for x in depressive_tweets_df[5]]
random_tweets_arr = [x for x in random_tweets_df['SentimentText']]
X_d = clean_tweets(depressive_tweets_arr)
X_r = clean_tweets(random_tweets_arr)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_d + X_r)

sequences_d = tokenizer.texts_to_sequences(X_d)
sequences_r = tokenizer.texts_to_sequences(X_r)


word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))


data_d = pad_sequences(sequences_d, maxlen=MAX_SEQUENCE_LENGTH)
data_r = pad_sequences(sequences_r, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data_d tensor:', data_d.shape)
print('Shape of data_r tensor:', data_r.shape)

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

for (word, idx) in word_index.items():
    if word in word2vec.key_to_index and idx < MAX_NB_WORDS:
        embedding_matrix[idx] = word2vec.word_vec(word)

# Assigning labels to the depressive tweets and random tweets data
labels_d = np.array([1] * DEPRES_NROWS)
labels_r = np.array([0] * RANDOM_NROWS)

# Splitting the arrays into test (60%), validation (20%), and train data (20%)
perm_d = np.random.permutation(len(data_d))
idx_train_d = perm_d[:int(len(data_d)*(TRAIN_SPLIT))]
idx_test_d = perm_d[int(len(data_d)*(TRAIN_SPLIT)):int(len(data_d)*(TRAIN_SPLIT+TEST_SPLIT))]
idx_val_d = perm_d[int(len(data_d)*(TRAIN_SPLIT+TEST_SPLIT)):]

perm_r = np.random.permutation(len(data_r))
idx_train_r = perm_r[:int(len(data_r)*(TRAIN_SPLIT))]
idx_test_r = perm_r[int(len(data_r)*(TRAIN_SPLIT)):int(len(data_r)*(TRAIN_SPLIT+TEST_SPLIT))]
idx_val_r = perm_r[int(len(data_r)*(TRAIN_SPLIT+TEST_SPLIT)):]

# Combine depressive tweets and random tweets arrays
data_train = np.concatenate((data_d[idx_train_d], data_r[idx_train_r]))
labels_train = np.concatenate((labels_d[idx_train_d], labels_r[idx_train_r]))
data_test = np.concatenate((data_d[idx_test_d], data_r[idx_test_r]))
labels_test = np.concatenate((labels_d[idx_test_d], labels_r[idx_test_r]))
data_val = np.concatenate((data_d[idx_val_d], data_r[idx_val_r]))
labels_val = np.concatenate((labels_d[idx_val_d], labels_r[idx_val_r]))

# Shuffling
perm_train = np.random.permutation(len(data_train))
data_train = data_train[perm_train]
labels_train = labels_train[perm_train]
perm_test = np.random.permutation(len(data_test))
data_test = data_test[perm_test]
labels_test = labels_test[perm_test]
perm_val = np.random.permutation(len(data_val))
data_val = data_val[perm_val]
labels_val = labels_val[perm_val]

model = Sequential()
# Embedded layer
model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False))
# Convolutional Layer
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
# LSTM Layer
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
print(model.summary())


early_stop = EarlyStopping(monitor='val_loss', patience=3)

hist = model.fit(data_train, labels_train, validation_data=(data_val, labels_val),epochs=EPOCHS, batch_size=40, shuffle=True, callbacks=[early_stop])
#plot_model(model, to_file='model.png')
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

labels_pred = model.predict(data_test)
labels_pred = np.round(labels_pred.flatten())
accuracy = accuracy_score(labels_test, labels_pred)
print("Accuracy: %.2f%%" % (accuracy*100))

print(classification_report(labels_test, labels_pred))


class LogReg:
  """
  Class to represent a logistic regression model.
  """

  def __init__(self, l_rate, epochs, n_features):
    """
    Create a new model with certain parameters.

    :param l_rate: Initial learning rate for model.
    :param epoch: Number of epochs to train for.
    :param n_features: Number of features.
    """
    self.l_rate = l_rate
    self.epochs = epochs
    self.coef = [0.0] * n_features
    self.bias = 0.0

  def sigmoid(self, score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """
    if abs(score) > threshold:
      score = threshold * sign(score)
    activation = exp(score)
    return activation / (1.0 + activation)

  def predict(self, features):
    """
    Given an example's features and the coefficients, predicts the class.

    :param features: List of real valued features for a single training example.

    :return: Returns the predicted class (either 0 or 1).
    """
    value = sum([features[i] * self.coef[i] for i in range(len(features))]) + self.bias
    return self.sigmoid(value)

  def sg_update(self, features, label):
    """
    Computes the update to the weights based on a predicted example.

    :param features: Features to train on.
    :param label: Corresponding label for features.
    """
    yhat = self.predict(features)
    e = label - yhat
    self.bias = self.bias + self.l_rate * e * yhat * (1 - yhat)
    for i in range(len(features)):
      self.coef[i] = self.coef[i] + self.l_rate * e * yhat * (1 - yhat) * features[i]
    return

  def train(self, X, y):
    """
    Computes logistic regression coefficients using stochastic gradient descent.

    :param X: Features to train on.
    :param y: Corresponding label for each set of features.

    :return: Returns a list of model weight coefficients where coef[0] is the bias.
    """
    for epoch in range(self.epochs):
      for features, label in zip(X, y):
        self.sg_update(features, label)
    return self.bias, self.coef

def get_accuracy(y_bar, y_pred):
  """
  Computes what percent of the total testing data the model classified correctly.

  :param y_bar: List of ground truth classes for each example.
  :param y_pred: List of model predicted class for each example.

  :return: Returns a real number between 0 and 1 for the model accuracy.
    """
  correct = 0
  for i in range(len(y_bar)):
    if y_bar[i] == y_pred[i]:

      correct += 1
    accuracy = (correct / len(y_bar)) * 100.0
    return accuracy


# Logistic Model
logreg = LogReg(LEARNING_RATE, EPOCHS, len(data_train[0]))
bias_logreg, weights_logreg = logreg.train(data_train, labels_train)
y_logistic = [round(logreg.predict(example)) for example in data_test]

# Compare accuracies
accuracy_logistic = get_accuracy(y_logistic, labels_test)
#print('Logistic Regression Accuracy: {:0.3f}'.format(accuracy_logistic))
print("Logistic Regression Accuracy: %.2f%%" % (accuracy_logistic*100))
print(classification_report(labels_test, y_logistic))


#Decision Tree
class DecisionTree:
    """
    Class to represent a decision tree classifier.
    """

    def __init__(self, max_depth=None):
        """
        Create a new decision tree classifier.

        :param max_depth: Maximum depth of the decision tree. If None, the tree will expand until all leaves are pure.
        """
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, y):
        """
        Calculate the entropy of a target variable.

        :param y: List or numpy array of target values.

        :return: Entropy value.
        """
        counter = Counter(y)
        total_samples = len(y)
        entropy = 0
        for label in counter:
            probability = counter[label] / total_samples
            entropy -= probability * np.log2(probability)
        return entropy

    def _information_gain(self, X, y, feature_idx, threshold):
        """
        Calculate the information gain for a specific feature and threshold.

        :param X: Feature matrix.
        :param y: Target vector.
        :param feature_idx: Index of the feature for which information gain is calculated.
        :param threshold: Threshold value for splitting the feature.

        :return: Information gain value.
        """
        # Split data based on the threshold
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        left_y, right_y = y[left_mask], y[right_mask]

        # Calculate parent entropy
        parent_entropy = self._entropy(y)

        # Calculate weighted average of child entropies
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)
        child_entropy = (len(left_y) / len(y)) * left_entropy + (len(right_y) / len(y)) * right_entropy

        # Calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _find_best_split(self, X, y):
        """
        Find the best split for the data.

        :param X: Feature matrix.
        :param y: Target vector.

        :return: Tuple (best_feature_idx, best_threshold) representing the best split.
        """
        best_information_gain = -float('inf')
        best_feature_idx, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_idx])
            for threshold in unique_values:
                information_gain = self._information_gain(X, y, feature_idx, threshold)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        :param X: Feature matrix.
        :param y: Target vector.
        :param depth: Current depth of the tree.

        :return: Dictionary representing the decision tree.
        """
        # Check termination conditions
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        # Find best split
        best_feature_idx, best_threshold = self._find_best_split(X, y)

        if best_feature_idx is None:
            return Counter(y).most_common(1)[0][0]

        # Split data based on the best split
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        # Recursively build subtrees
        subtree = {}
        subtree["feature_idx"] = best_feature_idx
        subtree["threshold"] = best_threshold
        subtree["left"] = self._build_tree(left_X, left_y, depth + 1)
        subtree["right"] = self._build_tree(right_X, right_y, depth + 1)

        return subtree

    def fit(self, X, y):
        """
        Fit the decision tree classifier to the data.

        :param X: Feature matrix.
        :param y: Target vector.
        """
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_instance(self, instance, tree):
        """
        Recursively predict the class for a single instance.

        :param instance: Feature vector of the instance.
        :param tree: Dictionary representing the decision tree.

        :return: Predicted class.
        """
        if isinstance(tree, dict):
            feature_idx, threshold = tree["feature_idx"], tree["threshold"]
            if instance[feature_idx] <= threshold:
                return self._predict_instance(instance, tree["left"])
            else:
                return self._predict_instance(instance, tree["right"])
        else:
            return tree

    def predict(self, X):
        """
        Predict the classes for multiple instances.

        :param X: Feature matrix.

        :return: Predicted classes.
        """
        if self.tree is None:
            raise ValueError("The model has not been trained yet.")

        predictions = []
        for instance in X:
            predictions.append(self._predict_instance(instance, self.tree))
        return predictions

# Instantiate Decision Tree Model
tree_model = DecisionTree(max_depth=3)  # You can specify the maximum depth of the tree
# Train Decision Tree Model
tree_model.fit(data_train, labels_train)
# Make Predictions
y_tree = tree_model.predict(data_test)

# Compute Accuracy for Decision Tree Model
accuracy_tree = get_accuracy(y_tree, labels_test)
# Print Accuracy for Decision Tree Model
#print('Decision Tree Accuracy: {:0.3f}'.format(accuracy_tree))
print("Decision Tree Accuracy: %.2f%%" % (accuracy_tree*100))
print(classification_report(labels_test, y_tree))

