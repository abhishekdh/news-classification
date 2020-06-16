import pickle
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

nlp = spacy.load("en")
stopwords = list(STOP_WORDS)
class Foo(object):
    def __init__(self, name):
        self.name = name


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

if __name__=='__main__':
    main()