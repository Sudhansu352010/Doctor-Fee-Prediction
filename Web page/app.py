

from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('mlpro.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('indextest.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['speciality_of_doctor']
    degree_type = request.form['degree_type']
    year_of_experianc = request.form['year_of_experianc']
    city = request.form['city']
    location = request.form['location']
    dp_score = request.form['dp_score']
    npv = request.form['npv']
    arr = np.array([[speciality_of_doctor, degree_type, year_of_experianc, city,location,dp_score,npv]])
    pred = model.predict(arr)
    return render_template('indextest.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)