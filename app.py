from flask import Flask,request,url_for,render_template,redirect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/about",methods=['GET','POST'])
def about():
	return render_template('about.html')
@app.route("/index",methods=['GET','POST'])
def about1():
	return render_template('index.html')


@app.route("/contact",methods=['GET','POST'])
def contact():
	return render_template('contact.html')


@app.route("/breastcancer",methods=['GET','POST'])
def breastcancer():
	if request.method == 'POST':
		age = request.form['age']
		gender = request.form['gender']
		diagnosis = request.form['diagnosis']
		tumor = request.form['tumor']
		lymph = request.form['lymph']
		insito = request.form['insito']
		histologic = request.form['histologic']
		

		# Clean the data by convert from unicode to float 
		sample_data = [age,gender,diagnosis,tumor,lymph,insito,histologic]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)
		data = pd.read_csv('ubc_train_dataset.csv')
		X = data.drop('Stage',axis=1)
		Y=pd.DataFrame(data['Stage'])
		X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=5)
		knn = KNeighborsClassifier()
		knn.fit(X_train,Y_train)
		a=knn.score(X_test,Y_test)
		example = pd.DataFrame([age,gender,diagnosis,tumor,lymph,insito,histologic])
		example = example.transpose()
		pred=knn.predict(example)[0]
		if pred == 1:
			return redirect(url_for('output2'))
		
		else:
			return render_template('output21.html',b='U dont have Breast cancer',c=a)
	return render_template('breastcancer.html')


@app.route("/lungscancer",methods=['GET','POST'])
def lungscancer():
	if request.method == 'POST':
		age = request.form['age']
		gender = request.form['gender']
		smoking = request.form['SMOKING']
		anxiety= request.form['ANXIETY']
		coughing = request.form['COUGHING']
		alcoholconsuming = request.form['ALCOHOLCONSUMING']
		chestpain = request.form['CHESTPAIN']

		

		# Clean the data by convert from unicode to float 
		sample_data = [age,gender,smoking,anxiety,coughing,alcoholconsuming,chestpain]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)
		data = pd.read_csv('survey_lung_cancer.csv')
		X = data.drop('lung_cancer',axis=1)
		Y=pd.DataFrame(data['lung_cancer'])
		X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=5)
		knn = KNeighborsClassifier()
		knn.fit(X_train,Y_train)
		a=knn.score(X_test,Y_test)
		example = pd.DataFrame([age,gender,smoking,anxiety,coughing,alcoholconsuming,chestpain])
		example = example.transpose()
		pred=knn.predict(example)[0]
		if pred == 1:
			return redirect(url_for('output1'))
		
		else:
			return render_template('output11.html',b='U dont have lungs cancer',c=a)
	return render_template('lungscancer.html')


@app.route("/cervical",methods=['GET','POST'])
def cervical():
	if request.method == 'POST':
		age = request.form['age']
		sexual = request.form['sexual']
		firstsex = request.form['firstsex']
		preg = request.form['preg']
		smokes = request.form['smokes']
		
		
		

		# Clean the data by convert from unicode to float 
		sample_data = [age,sexual,firstsex,preg,smokes]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)
		data = pd.read_csv('cervical_cancer.csv')
		X = data.drop('Biopsy',axis=1)
		Y=pd.DataFrame(data['Biopsy'])
		X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=5)
		dtree = DecisionTreeClassifier()
		dtree.fit(X_train,Y_train)
		a=dtree.score(X_test,Y_test)
		example = pd.DataFrame([age,sexual,firstsex,preg,smokes])
		example = example.transpose()
		pred=dtree.predict(example)[0]
		if pred == 1:
			return redirect(url_for('output3'))
		else:
			return render_template('output31.html',b='U dont have cervical cancer',c=a)
		
	
	return render_template('cervical.html')
















@app.route("/output2",methods=['GET','POST'])
def output2():
	return render_template('output2.html')

@app.route("/output21",methods=['GET','POST'])
def output21():
	return render_template('output21.html')



@app.route("/output1",methods=['GET','POST'])
def output1():
	return render_template('output1.html')

@app.route("/output11",methods=['GET','POST'])
def output11():
	return render_template('output11.html')


@app.route("/output3",methods=['GET','POST'])
def output3():
	return render_template('output3.html')

@app.route("/output31",methods=['GET','POST'])
def output31():
	return render_template('output31.html')










if __name__ == "__main__":
    app.run()
