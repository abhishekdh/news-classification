# Import libraries
from flask import Flask, request, jsonify, render_template
#import get_data
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.symbols import punct
from sklearn.pipeline import Pipeline

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


def predictdata(text):
    tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
    joblib_LR_model = joblib.load('news_classifier.pkl')
    print(joblib_LR_model)
    pred = joblib_LR_model.predict([text])
    print(pred)
    return pred


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")
# Load the model
#model = pickle.load(open('news_classifier.pkl','rb'))
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    print(data)
    if data['headline'] != "":
     output = predictdata(data['headline'][0])
     print("output = " + output[0])
     return jsonify(output[0])
    else:
     output = 'Input cannot pe empty!!'
     return jsonify(output)



@app.route('/form',methods=['POST'])
def form_predict():
    # Get the data from the POST request.
    data = request.form
    print("data = ",data['headline'])
    if data['headline'] != "":
     output = predictdata(data['headline'][0])
     return jsonify(output[0])
    else:
     output = 'Input cannot pe empty!!'
     return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)








