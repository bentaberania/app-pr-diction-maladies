import pickle 
from flask import Flask, redirect, render_template, request, session, url_for, flash
import MySQLdb.cursors
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import tensorflow as tf
#import imageio
from PIL import Image
from keras.models import load_model
import cv2



app = Flask(__name__)

from flask_mysqldb import MySQL

mysql = MySQL(app)
app.secret_key = 'xyzsdfg'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user-system'


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    message = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form:
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user1 WHERE email = %s', (email,))
        account = cursor.fetchone()
        if account:
            message = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            message = 'Invalid email address !'
        elif not userName or not password or not email:
            message = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user1 (name, email, password) VALUES (%s, %s, %s)', (userName, email, password))
            mysql.connection.commit()
            message = 'You have successfully registered !'
            return redirect(url_for('login'))
    elif request.method == 'POST':
        message = 'Please fill out the form !'
    return render_template('signup.html', message=message)


@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user1 WHERE email = %s AND password = %s', (email, password))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['email'] = user['email']
            message = 'Logged in successfully !'
            return redirect(url_for('home2'))  # Rediriger vers la page d'accueil après connexion
        else:
            message = 'Please enter correct email / password !'
    return render_template('login.html', message=message)



@app.route("/logout")
def logout():
    session.pop('email', None)
    session.pop('password', None)
    return redirect(url_for('index'))


# Charger le modèle depuis le fichier
with open('modele_regression_logistique.pkl', 'rb') as model_heart:
    model = pickle.load(model_heart)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home2')
def home2():
    if 'loggedin' in session:
        return render_template('home2.html')
    return redirect(url_for('login'))



@app.route('/aboutus')
def about():
    return render_template("aboutus.html")

@app.route('/help')
def help():
    return render_template("help.html")
#model cnn

       
class HeartDiseasePredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        input_data_reshaped = np.asarray(input_data).reshape(1, -1)
        probabilities = self.model.predict_proba(input_data_reshaped)
        predicted_class = np.argmax(probabilities)
        return probabilities[0][0], probabilities[0][1], predicted_class

predictor = HeartDiseasePredictor(model)

@app.route('/Heart_prediction', methods=['GET', 'POST'])
def Heart_prediction():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        age = float(request.form['Age'])
        sex = float(request.form['Sex'])
        cp = float(request.form['Chest Pain types'])
        trestbps = float(request.form['Resting Blood Pressure'])
        chol = float(request.form['Serum Cholestoral in mg/dl'])
        fbs = float(request.form['Fasting Blood Sugar > 120 mg/dl'])
        restecg = float(request.form['Resting Electrocardiographic results'])
        thalach = float(request.form['Maximum Heart Rate achieved'])
        exang = float(request.form['Exercise Induced Angina'])
        oldpeak = float(request.form['ST depression induced by exercise'])
        slope = float(request.form['Slope of the peak exercise ST segment'])
        ca = float(request.form['Major vessels colored by flourosopy'])
        thal = float(request.form['thal'])

        # Utiliser le modèle pour faire des prédictions
        prob_healthy, prob_disease, predicted_class = predictor.predict([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

        # Interpréter la prédiction
        if predicted_class == 1:
            result = "The Person has Heart Disease."
        else:
            result = "The Person does not have a Heart Disease."

        # Renvoyer les résultats à la page HTML appropriée
        return render_template('results.html', prob_healthy=prob_healthy, prob_disease=prob_disease, prediction=result)

    # Si la méthode est GET, affichez simplement la page d'accueil
    return render_template('Heart_prediction.html')

# Charger le modèle depuis le fichier
with open('modele_svc.pkl', 'rb') as model_diabetes:
    model = pickle.load(model_diabetes)

class DiabetesPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        input_data_reshaped = np.asarray(input_data).reshape(1, -1)
        probabilities = self.model.predict_proba(input_data_reshaped)
        predicted_class = np.argmax(probabilities)
        return probabilities[0][0], probabilities[0][1], predicted_class

predictor_diabetes = DiabetesPredictor(model)

@app.route('/Diabete_prediction', methods=['GET', 'POST'])
def Diabete_prediction():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        pregnancies = float(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        blood_pressure = float(request.form['BloodPressure'])
        skin_thickness = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])

        # Utiliser le modèle pour faire des prédictions
        prob_negative, prob_positive, predicted_class = predictor_diabetes.predict([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])

        # Interpréter la prédiction
        if predicted_class == 1:
            result = "The person is diabetic."
        else:
            result = "The person is not diabetic."

        # Renvoyer les résultats 
        return render_template('Diabete_results.html', prob_negative=prob_negative, prob_positive=prob_positive, predictions=result)

    # Si la méthode est GET, affichez simplement la page d'accueil
    return render_template('Diabete_prediction.html')

#cnn
#modelTumor

model = load_model('model.h5')  

labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Récupérer le fichier téléchargé
        file = request.files['image']
        
        # Charger l'image et la prétraiter
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150, 150))
        img_array = np.array(img)
        img_array = img_array.reshape(1, 150, 150, 3)
        
        # Faire la prédiction
        prediction = model.predict(img_array)
        predicted_class = labels[np.argmax(prediction)]

        return render_template('classification.html', prediction_result=predicted_class)
    
    # Si la méthode est GET, retournez simplement la page de prédiction
    return render_template('classification.html')

    
if __name__ == '__main__':
    app.run(debug=True)
