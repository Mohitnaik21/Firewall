#to run this us the command 'set FLASK_APP=hello.py' then 'flask run'
from flask import Flask,render_template,request
import  subprocess,os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pickle
df=pd.read_csv('data.csv')
name = df['Queries'].values.tolist()
y = df['Label'].values.tolist()
print(len(y))
x_train, x_test, y_train, y_test = train_test_split(name, y, test_size=0.2, random_state=1000)
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
pkl_filename = "log_res.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)
app=Flask(__name__)
df=pd.read_csv('Output.csv') #change file name
name=df['name'].values.tolist()
label=df['Label'].values.tolist()

@app.route('/')

def Compiler():
        check=""
        return render_template('Index.html')

@app.route('/submit',methods=['GET','POST'])

def submit():
        if request.method=='POST':
                code=request.form['code']
                result=classifier_output(code)
                output=""
                if(result==0):
                        output='Malicious Query'
                elif(result==1):
                        output='Safe Query'
                
        return render_template('Index.html',code=code,output=output)

def classifier_output(code):
        test=[]
        test.append(code)
        Test=vectorizer.transform(test)
        R=model.predict(Test)
        name.append(code)
        label.append(R[0])
        print("len of name is",len(name),"len of label",len(label))
        df1=pd.DataFrame(name, columns =['name'])
        df1['Label']=label
        df1.to_csv("output.csv",index=False,header=True)
        a=R[0]
        return a
        

if __name__=='__main__':
        app.run(debug=True)
        
