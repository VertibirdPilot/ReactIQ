from flask import Flask, render_template, request
import pickle
import re
from tweepy import OAuthHandler, API, TweepyException
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os
from dotenv import load_dotenv  # Import for .env file

load_dotenv() #Load environment variables

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# Load the trained ensemble sentiment analysis model
sentiment_model = None # Initialize to None
try:
    with open('ensemble_sentiment_model.pkl', 'rb') as model_file:
        sentiment_model = pickle.load(model_file)
    print("Ensemble sentiment model loaded successfully.")
except Exception as e:
    print(f"Error loading ensemble sentiment model: {e}")
    # Don't exit here, handle the missing model gracefully in predict_sentiment

# Load the TF-IDF vectorizer
tfidf_vectorizer = None # Initialize to None
try:
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
    print("TF-IDF vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading TF-IDF vectorizer: {e}")
    # Don't exit here, handle the missing vectorizer gracefully in predict_sentiment


def preprocess_text(text):
    """
    Preprocesses the input text by removing URLs, mentions, hashtags,
    special characters, converting to lowercase, tokenizing, lemmatizing,
    and removing stop words.

    Args:
        text: The text string to preprocess.

    Returns:
        A string containing the preprocessed text.
    """
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def predict_sentiment(text):
    """
    Predicts the sentiment of the given text using the loaded sentiment analysis model.

    Args:
        text: The text string for which to predict the sentiment.

    Returns:
        A string representing the predicted sentiment ('Negative', 'Neutral', 'Positive'),
        or an error message if the model or vectorizer failed to load.
    """
    if sentiment_model is None or tfidf_vectorizer is None:
        return "Error: Sentiment model or vectorizer not loaded.  Please check the application logs."
    processed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([processed_text])
    prediction = sentiment_model.predict(text_tfidf)[0]
    if prediction == 0:
        return 'Negative'
    elif prediction == 1:
        return 'Neutral'
    else:
        return 'Positive'



class TwitterClient(object):
    """
    A class for interacting with the Twitter API to fetch tweets.
    """
    def __init__(self):
        """
        Initializes the TwitterClient with API credentials from environment variables.
        """
        consumer_key = os.environ.get('CONSUMER_KEY')
        consumer_secret = os.environ.get('CONSUMER_SECRET')
        access_token = os.environ.get('ACCESS_TOKEN')
        access_token_secret = os.environ.get('ACCESS_TOKEN_SECRET')

        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            raise ValueError("One or more Twitter API keys are not set in the environment.")

        try:
            auth = OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            self.api = API(auth, wait_on_rate_limit=True)

        except TweepyException as e:
            print(f"Error: Tweeter Authentication Failed - \n{str(e)}")
            self.api = None  # Set self.api to None in case of error

    def get_tweets(self, query, maxTweets=100):
        """
        Fetches tweets based on the given query.

        Args:
            query: The search query string.
            maxTweets: The maximum number of tweets to retrieve.  Defaults to 100.

        Returns:
            A pandas DataFrame containing the fetched tweets, or an empty DataFrame
            if there's an error or no tweets are found.
        """
        if self.api is None:
            print("Error: Twitter API not initialized. Cannot fetch tweets.")
            return pd.DataFrame()  # Return empty DataFrame on error

        tweets = []
        try:
            fetched_tweets = self.api.search_tweets(q=query, count=maxTweets, tweet_mode='extended', lang="en")
            for tweet in fetched_tweets:
                parsed_tweet = {}
                parsed_tweet['tweets'] = tweet.full_text
                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
            return pd.DataFrame(tweets)
        except Exception as e:
            print(f"Tweepy error : {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error



def con1(sentence):
    """
    Performs a simple emotion analysis on the input sentence based on a word list.
    This function reads an 'emotions.txt' file, which should contain words
    and their associated emotions (e.g., "happy:joy").  It then checks if any of
    those words are present in the input sentence and returns the corresponding emotions.

    Args:
        sentence: The input sentence to analyze.

    Returns:
        A list of emotions found in the sentence.  Returns an empty list if
        the 'emotions.txt' file is not found or if no matching words are found.
    """
    emotion_list = []
    sentence = sentence.split(' ')
    try:
        with open('emotions.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')
                if word in sentence:
                    emotion_list.append(emotion)
    except FileNotFoundError:
        print("Warning: emotions.txt not found. Emotion analysis will be skipped.")
        return []  # Return an empty list in case of error
    return emotion_list



@app.route('/')
def home():
    """
    Renders the home page (index.html).
    """
    return render_template('index.html')  # Corrected template name

@app.route('/hello')  # Add this route
def hello():
    return "Hello, Flask is working!"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request when the user submits a tweet/text.
    Fetches tweets from Twitter, predicts their sentiment, performs emotion analysis,
    and renders the results page.
    """
    if request.method == 'POST':
        comment = request.form['Tweet']  # Get the tweet from the form
        twitter_client = TwitterClient()
        tweets_df = twitter_client.get_tweets(comment, maxTweets=100)

        if tweets_df.empty:
            return render_template('result.html', outputs={}, NU=0, N=0, P=0, happy=0, sad=0,
                                   angry=0, loved=0, powerless=0, surprise=0, fearless=0,
                                   cheated=0, attracted=0, singledout=0, anxious=0,
                                   error_message="No tweets found for the given query, or error fetching tweets.")

        predicted_sentiments = []
        cleaned_tweets = []
        original_tweets = []

        for tweet_data in tweets_df['tweets']:
            original_tweets.append(tweet_data)
            cleaned_tweet = preprocess_text(tweet_data)
            cleaned_tweets.append(cleaned_tweet)
            predicted_sentiments.append(predict_sentiment(tweet_data))  # Use the predict_sentiment function

        output = dict(zip(original_tweets, predicted_sentiments))

        Neucount = predicted_sentiments.count('Neutral')
        Negcount = predicted_sentiments.count('Negative')
        Poscount = predicted_sentiments.count('Positive')

        all_cleaned_text = " ".join(cleaned_tweets)
        emo = con1(all_cleaned_text)
        h = emo.count(' happy')
        s = emo.count(' sad')
        a = emo.count(' angry')
        l = emo.count(' loved')
        pl = emo.count(' powerless')
        su = emo.count(' surprise')
        fl = emo.count(' fearless')
        c = emo.count(' cheated')
        at = emo.count(' attracted')
        so = emo.count(' singled out')
        ax = emo.count(' anxious')

        return render_template('result.html', outputs=output, NU=Neucount, N=Negcount, P=Poscount, happy=h, sad=s, angry=a, loved=l, powerless=pl, surprise=su, fearless=fl, cheated=c, attracted=at, singledout=so, anxious=ax)

    else:
        return render_template('index.html') #handle non-post requests


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
