import pandas as pd
import numpy as np

data = pd.read_csv('preprocessed_data.csv',nrows=10000)

print(data.shape)
print(data.head())
print(data.keys())

data[data['category'] == 'e']

news_data = data[['headline','authors','category']]
print(news_data.head())



news_data['category'].value_counts()

news_data.isnull().sum()

#Tokenization
import string
punct = string.punctuation

print(punct)

#Data Cleaning
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

nlp = spacy.load("en")
stopwords = list(STOP_WORDS)


def text_data_cleaning(sentence):
    doc = nlp(sentence)

    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens

text_data_cleaning("  tis is the best in the Himansuh")

#Classification
from sklearn.svm import LinearSVC
tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
classifier = LinearSVC()

X = news_data['headline']
y = news_data['category']

X_train,X_test,y_train,y_test = train_test_split(X,y) #,test_size=1,random_state=6)

print(X_train.shape, y_train.shape)

clf = Pipeline([('tfidf',tfidf),('clf',classifier)])

print(y_train.head())

clf.fit(X_train,y_train)

text = 'MS Dhoni at No. 3 would have broken most records’: Gautam Gambhir'

pred = clf.predict([text])
print("data data === "+pred)
if pred == 'b':
    print("Business News")
elif pred == 't':
    print("Science and Technology")
elif pred == 'e':
    print("Entertainment")
elif pred == 'm':
    print("Health")



accuracy_score(y_test,clf.predict(X_test))

print(classification_report(y_test,clf.predict(X_test)))

#Model Save
import joblib
joblib.dump(clf,'news_classifier.pkl')





