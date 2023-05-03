from flask import Flask, render_template, request
import pickle
import numpy as np
from dict import Speciality_dict, degree_dict, City_dict, location_dict

model = pickle.load(open('mlpro.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('indextest.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['speciality_of_doctor']
    degree_type = request.form['degree_type']
    year_of_experience = request.form['year_of_experience']
    city = request.form['city']
    location = request.form['location']
    dp_score = request.form['dp_score']
    npv = request.form['npv']

    # map user inputs to encoded values using dictionaries
    encoded_speciality = Speciality_dict.get(data1)
    encoded_degree = degree_dict.get(degree_type)
    encoded_city = City_dict.get(city)
    encoded_location = location_dict.get(location)

    # create input array for the model
    arr = np.array([[encoded_speciality, encoded_degree, year_of_experience, encoded_location, encoded_city, dp_score, npv]])
    arr = arr.astype(int)
    arr = arr.reshape(1, -1)

    print("Input: ", arr)

    # make prediction
    pred = model.predict(arr)
    pred_rounded = round(pred[0] / 50) * 50
    # return prediction result to the template
    return render_template('indextest.html', prediction_text='fees should be  ₹ {}'.format(pred_rounded))

if __name__ == "__main__":
    app.run()
