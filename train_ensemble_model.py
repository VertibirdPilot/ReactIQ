Import pandas as pd
From sklearn.model_selection import train_test_split
From sklearn.feature_extraction.text import TfidfVectorizer
From sklearn.naive_bayes import MultinomialNB
From sklearn.linear_model import LogisticRegression
From sklearn.svm import LinearSVC
From sklearn.ensemble import VotingClassifier
From sklearn.metrics import accuracy_score, classification_report
Import pickle
Import nltk
Import re
From nltk.corpus import stopwords
From nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (run this once)
Try:
    Stopwords.words(‘english’)
Except LookupError:
    Nltk.download(‘stopwords’)
Try:
    WordNetLemmatizer().lemmatize(‘running’)
Except LookupError:
    Nltk.download(‘wordnet’)
Try:
    Nltk.data.find(‘omw-1.4’)
Except LookupError:
    Nltk.download(‘omw-1.4’)

Stop_words = set(stopwords.words(‘english’))
Lemmatizer = WordNetLemmatizer()

Def preprocess_text(text):
    “””
    Preprocesses the input text by removing URLs, mentions, hashtags,
    Special characters, converting to lowercase, tokenizing, lemmatizing,
    And removing stop words.

    Args:
        Text (str): The text to preprocess.

    Returns:
        Str: The preprocessed text.
    “””
    Text = re.sub(r’http\S+’, ‘’, text)
    Text = re.sub(r’@\w+’, ‘’, text)
    Text = re.sub(r’#\w+’, ‘’, text)
    Text = re.sub(r’[^a-zA-Z\s]’, ‘’, text)
    Text = text.lower()
    Tokens = text.split()
    Tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    Return “ “.join(tokens)

# Load your dataset
Try:
    Data = pd.read_csv(‘App.csv’, encoding=’latin1’)
    Data = data[[‘text’, ‘sentiment’]].dropna()  # Select only ‘text’ and ‘sentiment’ and drop NaNs
Except FileNotFoundError:
    Print(“Error: App.csv not found. Make sure it’s in the same directory.”)
    Exit()
Except KeyError as e:
    Print(f”Error: Required column not found in App.csv.  Check column names.  Error was: {e}”)
    Exit()
Except Exception as e:
    Print(f”An unexpected error occurred while loading App.csv: {e}”)
    Exit()

#  No need to map, sentiment is already 0,1,2.  Check for bad values.
Valid_sentiments = [0, 1, 2]
Invalid_sentiments = data[~data[‘sentiment’].isin(valid_sentiments)]
If not invalid_sentiments.empty:
    Print(“Warning:  Invalid sentiment values found in App.csv.  These rows will be dropped:”)
    Print(invalid_sentiments)
    Data = data[data[‘sentiment’].isin(valid_sentiments)]
    
If data.empty:
    Print(“Error: No valid data remaining after filtering.  Please check your App.csv file.”)
    Exit()
    

Data[‘processed_text’] = data[‘text’].apply(preprocess_text)

X = data[‘processed_text’]
Y = data[‘sentiment’]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction using TF-IDF
Tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize individual classifiers
Nb_clf = MultinomialNB()
Lr_clf = LogisticRegression(solver=’liblinear’, random_state=42)
Svm_clf = LinearSVC(random_state=42)

# Train individual classifiers
Try:
    Nb_clf.fit(X_train_tfidf, y_train)
    Lr_clf.fit(X_train_tfidf, y_train)
    Svm_clf.fit(X_train_tfidf, y_train)
Except Exception as e:
    Print(f”Error training the classifiers: {e}”)
    Exit()

# Create a VotingClassifier
Voting_clf = VotingClassifier(estimators=[(‘nb’, nb_clf), (‘lr’, lr_clf), (‘svm’, svm_clf)], voting=’hard’)
Try:
    Voting_clf.fit(X_train_tfidf, y_train)
Except Exception as e:
    Print(f”Error training the VotingClassifier: {e}”)
    Exit()

# Evaluate the ensemble model
Y_pred = voting_clf.predict(X_test_tfidf)
Print(“Ensemble Accuracy:”, accuracy_score(y_test, y_pred))
Print(“Ensemble Classification Report:\n”, classification_report(y_test, y_pred))

# Save the trained ensemble model and the vectorizer
Try:
    With open(‘ensemble_sentiment_model.pkl’, ‘wb’) as model_file:
        Pickle.dump(voting_clf, model_file)

    With open(‘tfidf_vectorizer.pkl’, ‘wb’) as vectorizer_file:
        Pickle.dump(tfidf_vectorizer, vectorizer_file)
Except Exception as e:
    Print(f”Error saving the model or vectorizer: {e}”)
    Exit()
    

Print(“Trained ensemble sentiment model and TF-IDF vectorizer saved as ensemble_sentiment_model.pkl and tfidf_vectorizer.pkl.”)


