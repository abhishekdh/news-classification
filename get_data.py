# Use the loaded pickled model to make predictions
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.symbols import punct

#Tokenization
import string
punct = string.punctuation
print(punct)

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
        if token not in list(STOP_WORDS) and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens












text = '‘MS Dhoni at No. 3 would have broken most records’: Gautam Gambhir'

def predictdata(text):
    tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
    joblib_LR_model = joblib.load('news_classifier1.pkl')
    print(joblib_LR_model)
    pred = joblib_LR_model.predict([text])
    print(pred)
    return pred




