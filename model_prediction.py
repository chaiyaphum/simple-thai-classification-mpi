# Run in CMD: python model_prediction.py

from flask import Flask
from flask import request

import json
import deepcut
import pickle
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

stopwords = open('data/stopwords-th_new.txt', 'r', encoding="utf8").read().split()
print('Stop word list: Done!')

loaded_tfidf_transformer = joblib.load('finalized_tfidftransformer.pkl')
print('tf-idf: Done!')

model_fname = 'model/_finalized_model_svc_c_3.0.sav'
loaded_model = joblib.load(open(model_fname, 'rb'))
print('Pre-trained model: Done!')

def tokenize_text_list(_text):
    _text_split_list = ''.join(_text).strip().split(',')
    
    markeredtext= []
    for word in _text_split_list:
        if word not in stopwords and word.strip():
            markeredtext.append(word)
    
    return ''.join(markeredtext)


app = Flask(__name__)

@app.route("/")
def default_page():
    return "It's work!"

@app.route("/api/prediction")
def get_prediction():
	text_input = request.args.get('text', default = '', type = str)

	_text_wordcut = ','.join(deepcut.tokenize(text_input))
	_text_wordcut = tokenize_text_list(_text_wordcut)
	X_val_tfidf = loaded_tfidf_transformer.transform([_text_wordcut])

	y_pred = loaded_model.predict(X_val_tfidf).tolist()
	print(y_pred)

	return json.dumps({'prediction': y_pred[0]})

if __name__ == '__main__':
	app.run(host='0.0.0.0')