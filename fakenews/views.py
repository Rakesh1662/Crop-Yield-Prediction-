# Libraries
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pymysql


def reg(request):
	if request.method=='POST':
		if request.POST.get('username') and request.POST.get('password') and request.POST.get('email') and request.POST.get('phone') and request.POST.get('address'):
			db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'crop',charset='utf8')
			db_cursor = db_connection.cursor()
			my_table_query="create table if not exists user(username varchar(50),password varchar(20),email varchar(30))"
			db_cursor.execute(my_table_query)
			student_sql_query = "INSERT INTO user(username,password,email,phone,address) VALUES('"+request.POST.get('username')+"','"+request.POST.get('password')+"','"+request.POST.get('email')+"','"+request.POST.get('phone')+"','"+request.POST.get('address')+"')"
			db_cursor.execute(student_sql_query)
			db_connection.commit()	  
			print('helllllllllllllllllllllllllllllllllllllllllllllllllloo')
			return render(request,'loginpage.html') 
		return render(request,'loginpage.html') 
def loginuser(request):
	if request.method=='POST':
		print('hiiiiiiiiiiiiiiiiii')
		username = request.POST.get('username', False)
		password = request.POST.get('password', False)		
		print('hiiiiiiiiiiiiiiiiii1111111111111')
		con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'crop',charset='utf8')
		utype = 'none'
		with con:
			print('hiiiiiiiiiiiiiiiiii33333333333333333333')
			cur = con.cursor()
			cur.execute("select * FROM user")
			rows = cur.fetchall()
			for row in rows:
				if row[0] == request.POST.get('username') and row[1] == request.POST.get('password'):
					utype = 'success'
					#status_data = row[5] 
					break
		if utype == 'success':
	 		print('hiiiiiiiiiiiiiiiiii11111111111122222222222222222222')
	 		return render(request, 'index.html')
		if utype == 'none':
	 		return render(request, 'loginpage.html')		
	return render(request,'index.html')
################ Home #################
def home(request):
	return render(request,'cropyield.html')

######## SVM ######
def nvb(request):
	data = pd.read_csv('C:/Users/Badri/OneDrive/Desktop/crop yield and crop prediction/crop.csv')
	from sklearn import preprocessing		
	labelencoder_X = preprocessing.LabelEncoder()
	X = data.iloc[:, 1:8].values
	y = data.iloc[:, 9].values	   
	X.shape
	y.shape

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

	A_test=[[1997,598400,24.243,42.3484,84,217000,1]]
	#testing  
	from sklearn import linear_model
	reg = linear_model.LinearRegression()
	reg.fit(X_train,y_train)
	pred = reg.predict(X_test)
	pred1 = reg.predict(A_test)
	print(pred1)
	score = reg.score(X_train,y_train)
	print("R-squared:", score)
	d = {'a': score}
	#print(reg.score(X_test,y_test))
	#acclogistic=reg.score(X_test,y_test)
	return render(request,'NaiveBayes.html',d)
def rf(request):
	data = pd.read_csv('C:/Users/Badri/OneDrive/Desktop/crop yield and crop prediction/crop.csv')
	X = data.iloc[:, 1:8].values
	y = data.iloc[:, 9].values	   
	X.shape
	y.shape

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

	A_test=[[1997,598400,24.243,42.3484,84,217000,1]]
	#testing  
	from sklearn.datasets import make_regression
	from sklearn.ensemble import RandomForestClassifier
	regr = RandomForestClassifier()
	regr.fit(X_train,y_train)
	pred = regr.predict(X_test)
	pred1 = regr.predict(A_test)
	print(pred1)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)
	d = {'a': score}
	#print(reg.score(X_test,y_test))
	#acclogistic=reg.score(X_test,y_test)
	return render(request,'NaiveBayes.html',d)
	
def svr(request):
	data = pd.read_csv('C:/Users/Badri/OneDrive/Desktop/crop yield and crop prediction/crop.csv')
	X = data.iloc[:, 1:8].values
	y = data.iloc[:, 9].values	   


	#testing  
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

	A_test=[[1997,598400,24.243,42.3484,84,217000,1]]
	
	from sklearn.svm import SVR
	regressor = SVR(kernel = 'rbf')
	
	regressor.fit(X_train,y_train)
	pred = regressor.predict(X_test)
	pred1 = regressor.predict(A_test)
	print(pred1)
	score = regressor.score(X_train,y_train)
	print("R-squared:", score)
	'''score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)'''
	d = {'a': score}
	#print(reg.score(X_test,y_test))
	#acclogistic=reg.score(X_test,y_test)
	return render(request,'NaiveBayes.html',d)
	
def pac(request):	
	return render(request,'NaiveBayes.html')
def svm(request):	
	return render(request,'NaiveBayes.html')
					
def accuracy(request):
	return render(request,'index.html')
  
def test(request):
	if request.method=='POST':
		headline1= request.POST.get('headline1')
		headline2= request.POST.get('headline2')
		headline3= request.POST.get('headline3')
		headline4= request.POST.get('headline4')
		headline5= request.POST.get('headline5')
		headline6= request.POST.get('headline6')
		
		from sklearn import preprocessing		
		labelencoder_X = preprocessing.LabelEncoder()
		headline6 = labelencoder_X.fit_transform([[headline6]])
		
		headline7= request.POST.get('headline7')
			
		print(headline1)
			
		headline1= int(headline1)
		headline2 = int(headline2)
		headline3 = float(headline3)
		headline4 = float(headline4)
		headline5 = int(headline5)
		headline6 = int(headline6)
		headline7 = int(headline7)
			
		data = pd.read_csv('C:/Users/Badri/OneDrive/Desktop/crop yield and crop prediction/crop.csv')

		X = data.iloc[:, 1:8].values
		y = data.iloc[:, 9].values	   
		X.shape
		y.shape

		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)
		A_test=[[headline1,headline2,headline3,headline4,headline5,headline6,headline7]]
		#testing  
		from sklearn.datasets import make_regression
		from sklearn.ensemble import RandomForestClassifier
		reg = RandomForestClassifier()
		reg.fit(X_train,y_train)
		pred = reg.predict(X_test)
		pred1 = reg.predict(A_test)
		print(pred1)
		print('------------------------------------------------')
		print(pred)
						
		fakefalse=''
		if pred1==0:
			fakefalse='less crop yield'
		else:
			fakefalse='high crop yield'
				
		score = metrics.accuracy_score(y_test, pred)
		print("accuracy:   %0.3f" % score)
		d = {'a':pred1,'crop':request.POST.get('headline6')}		   
		print('hellllllllllllllllllllllllllllllllo')
		return render(request,'NaiveBayes.html',d);

def fertilizerform(request):
	return render(request,'fertilizerform.html')
def fertilizerRf(request):
	if request.method=='POST':
		headline1= request.POST.get('headline1')
		headline2= request.POST.get('headline2')
		headline3= request.POST.get('headline3')
		headline4= request.POST.get('headline4')
		headline5= request.POST.get('headline5')
		headline6= request.POST.get('headline6')
		headline7= request.POST.get('headline7')
		
		from sklearn import preprocessing		
		labelencoder_X = preprocessing.LabelEncoder()
			
		print(headline1)
			
			
		headline1= float(headline1)
		headline2 = float(headline2)
		headline3 = float(headline3)
		headline4 = float(headline4)
		headline5 = float(headline5)
		headline6 = float(headline6)
		headline7 = float(headline7)
			
		data = pd.read_csv('C:/Users/Badri/OneDrive/Desktop/crop yield and crop prediction/cpdata.csv')

		X = data.iloc[:, 0:7].values
		y = data.iloc[:, 7].values	   
		X.shape
		y.shape

		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)
		A_test=[[headline1,headline2,headline3,headline4,headline5,headline6,headline7]]
		#testing  
		from sklearn.datasets import make_regression
		from sklearn.ensemble import RandomForestClassifier
		reg = RandomForestClassifier()
		reg.fit(X_train,y_train)
		pred = reg.predict(X_test)
		pred1 = reg.predict(A_test)
		print(pred1)
		print('------------------------------------------------')
		print(pred)
						
		
		
				
		score = metrics.accuracy_score(y_test, pred)
		print("accuracy:   %0.3f" % score)
		#d = {'a':pred1,'score':score}
		d = {'a':pred1,'crop':request.POST.get('headline6')}		   
		print('hellllllllllllllllllllllllllllllllo')
	#return render(request,'fres.html',d)
	return render(request,'indexfile1.html')
def simple_upload(request):
	return render(request,'indexfile1.html')
def simple(request):
	return render(request,'indexfile1.html')	
def fileshow(request):		
	return render(request,'indexfile1.html')
def fileshow1(request):
	return render(request,'indexfile2.html')
def loginpage(request):
	return render(request,'loginpage.html')
def register(request):
	return render(request,'register.html')

def input(request):
	return render(request,'input.html')	 









#return render(request,'indexfile1.html')	
# def fileshow(request):		
# 	return render(request,'indexfile1.html')
# def fileshow1(request):
# 	return render(request,'indexfile2.html')
# def loginpage(request):
# 	return render(request,'loginpage.html')
# def register(request):
# 	return render(request,'register.html')

# def input(request):
# 	return render(request,'input.html')
