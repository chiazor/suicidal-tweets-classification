from flask import Flask, render_template, url_for, request
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    vectorizer = open('vectorizer.pkl','rb')
    cv = joblib.load(vectorizer)
    _model = open('model3.pkl','rb')
    clf = joblib.load(_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('result.html', prediction=my_prediction)




if __name__ == '__main__':
    app.run(debug=True)