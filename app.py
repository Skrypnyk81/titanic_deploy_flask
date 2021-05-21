from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home_page():

    return render_template('home.html')


@app.route('/form', methods=['GET', 'POST'])
def form_case():
    pclass = request.form.get("pclass")
    sex = request.form.get("sex")
    age = request.form.get("age")
    sibsp = request.form.get("sibsp")
    parch = request.form.get("parch")
    fare = request.form.get("fare")
    embarked = request.form.get("embarked")
    data = [pclass, sex, age, sibsp, parch, fare, embarked]
    predict_df = pd.DataFrame(data).T

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    result = model.predict(predict_df)
    if result[0] == 0:
        return render_template('dead.html')
    else:
        return render_template('survival.html')
    return render_template('home.html')
