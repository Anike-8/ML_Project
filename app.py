from flask import Flask,render_template,request,redirect
import pickle
import numpy as np

model=pickle.load(open("model.pkl","rb"))

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("page.html")

@app.route("/predict",methods=["POST"])
def predict_placement():
    Pregnancies=float(request.form.get("Pregnancies"))
    Glucose=float(request.form.get("Glucose"))
    BloodPressure=float(request.form.get("BloodPressure"))
    SkinThickness=float(request.form.get("SkinThickness"))
    Insulin=float(request.form.get("Insulin"))
    BMI=float(request.form.get("BMI"))
    DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction"))
    Age=float(request.form.get("Age"))
    
    
    
    result=model.predict(np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age]]))
    
    if result[0]==1:
        return "<h1 style='color:green'>Diabetic</h1>"
    else:
        return "<h1 style='color:red'>Non Diabetic</h1>"
    
    
# @app.route("/predict",methods=["GET"])
# def predict_placement():
#     cgpa=float(request.args.get("cgpa"))
#     iq=float(request.args.get("iq"))
#     profile_score=float(request.args.get("profile_score"))
    
    
#     result=model.predict(np.array([[cgpa,iq,profile_score]]))
    
#     if result[0]==1:
#         return "<h1 style='color:green'>PLACED</h1>"
#     else:
#         return "<h1 style='color:red'>NOT PLACED</h1>"   

#sample.run(debug=True,port=5001)

app.run(debug=True)