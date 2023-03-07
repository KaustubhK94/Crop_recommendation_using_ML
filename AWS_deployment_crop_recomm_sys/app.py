from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open("crop_recommendation_classifier.pkl","rb"))
### IF you encounter with an error like jinja2.templatenotfound please specify template folder argument while creating flask instance
### and mention the path where template is stored
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict' , methods = ["POST"])
def crop_recommend():
    Nitrogen= float(request.form.get("Nitrogen"))
    Phosphorous = float(request.form.get("Phosphorous"))
    Potassium = float(request.form.get("Potassium"))
    Temperature =float(request.form.get("Temperature"))
    Humidity = float(request.form.get("Humidity"))
    pH = float(request.form.get("pH"))
    Rainfall = float(request.form.get("Rainfall"))

    result = model.predict(np.array([Nitrogen,Phosphorous,Potassium,Temperature,Humidity,pH,Rainfall]).reshape(1,7))
    return render_template("index.html", result=result)
    # return result
if __name__=="__main__":
    app.run(host='0.0.0.0', port=8080) ### use this while deploying your model on AWS
    # app.run(debug=True) ### uncomment this when you want to see how it works whne your hosting it on local server while commenting out the line 29 above..

