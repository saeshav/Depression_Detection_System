import warnings
warnings.filterwarnings("ignore")
import ftfy
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import re
from math import exp
from numpy import sign
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn import metrics

from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from gensim.models import KeyedVectors
from flask import Flask, request, render_template
from flask import Flask, request, render_template, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors



app = Flask(__name__)
app.secret_key = 'abcd'
app.debug = True
q = ""

MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_DB = 'users'

# Connect to MySQL
conn = MySQLdb.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DB
)
cursor = conn.cursor()

@app.route('/')
@app.route('/login',methods=['GET','POST'])
def login():
    message = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        
        #cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s AND password = %s', (email, password,))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user[3]  # userid is the fourth element in the tuple
            session['username'] = user[0]  # username is the first element in the tuple
            session['email'] = user[1]  # email is the second element in the tuple
            message = 'Logged in successfully !'
            return render_template('website.html', message=message)
        else:
            message = 'Please enter correct email / password !'
    return render_template('login.html', message=message)
     

@app.route('/signup', methods=['GET', 'POST'])
def register():
    message = ''  # Changed variable name to message for clarity
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        userName = request.form['username']
        password = request.form['password']
        email = request.form['email']

        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account = cursor.fetchone()
        if account:
            message = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            message = 'Invalid email address!'
        elif not userName or not password or not email:
            message = 'Please fill out the form!'
        else:
            insert_query = "INSERT INTO users (email, password, username) VALUES (%s, %s, %s)"
            user_data = (email, password, userName)
            cursor.execute(insert_query, user_data)
            conn.commit()
            message = 'You have successfully registered!'

    return render_template('signup.html', message=message)
    


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return redirect(url_for('login'))
    
@app.route("/detect", methods=['POST'])
def depressionDetection():
    
    inputQuery1 = request.form['tweet']
    print("User input:", inputQuery1)
    
    np.random.seed(1234)

    DEPRES_NROWS = 3200  # number of rows to read from DEPRESSIVE_TWEETS_CSV
    RANDOM_NROWS = 12000 # number of rows to read from RANDOM_TWEETS_CSV
    MAX_SEQUENCE_LENGTH = 280 # Max tweet size
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300
    TRAIN_SPLIT = 0.6
    TEST_SPLIT = 0.2
    #LEARNING_RATE = 0.1
    #EPOCHS = 1

    df = 'depressed.csv'
    RANDOM_TWEETS_CSV = 'random.csv'
    depressive_tweets_df = pd.read_csv(df, sep='|', header=None, usecols=range(0,9), nrows=DEPRES_NROWS)
    random_tweets_df = pd.read_csv(RANDOM_TWEETS_CSV, encoding="ISO-8859-1", usecols=range(0,4), nrows=RANDOM_NROWS)
    EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
    print(depressive_tweets_df.head())

    inputQuery1 = request.form['tweet']

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

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

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
    # Add a condition to check input size before applying max pooling
    if MAX_SEQUENCE_LENGTH > 2:
        model.add(MaxPooling1D(pool_size=2, padding='valid'))  # Adjust padding to 'valid'
    model.add(Dropout(0.2))
    # LSTM Layer
    model.add(LSTM(300))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
    print(model.summary())

    # Add print statements to check the shapes of data_test before and after prediction
    print('Shape of data_test before prediction:', data_test.shape)
    prediction = model.predict(data_test)
    binary_prediction = (prediction > 0.5).astype(int)
    accuracy = metrics.accuracy_score(binary_prediction, labels_test)
    # Add print statement to check the shape of data_test after prediction
    print('Shape of data_test after prediction:', data_test.shape)
    
    
    
    
    data = [[inputQuery1]]
   
    inputQuery1 = request.form['tweet']
    # Apply fix_text directly to the string
    tweet = ftfy.fix_text(inputQuery1)  
    # Expand contractions
    tweet = expandContractions(tweet)  
    # Clean the tweet: remove mentions, hashtags, emojis, URLs, and punctuation
    tweet = re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet)
    tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())  
    # Tokenize, remove stopwords, and stem the tweet
    tweet = nltk.word_tokenize(tweet)
    tweet = [word for word in tweet if word not in stopwords.words('english')]
    tweet = PorterStemmer().stem(' '.join(tweet))  
    sequence = tokenizer.texts_to_sequences([tweet])
    data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    
    single = model.predict(data)
    probability = (single == 0.5).astype(int)  # This assumes a binary classification

    if single > 0.5:
        
    
        o1 = "Depression is detected in the user's Tweet"
        o2 = "Confidence: {}".format(probability*100)
    else:
        o1 = "Depression is not detected in the user's Tweet"
        o2 = "Confidence: {}".format(probability*100) # Format to percentage, ensuring we access the correct element


    return render_template('website.html', output1=o1,output2=o2, tweet=inputQuery1)
  

if __name__ == "__main__":
    app.run()