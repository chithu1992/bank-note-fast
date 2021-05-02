


import os
import pandas as pd
import numpy as np


#os.chdir(r"D:\DS Study Material\T3\DE MD")
df = pd.read_csv("BankNote_Authentication.csv")
df.head()
df.isna().sum()
df.describe()
x = df.iloc[:,:-1]
y = df.iloc[:,-1:]
x.head()
y.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 1)


x_train.shape
x_test.shape
# using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
score
import pickle

pickle_out = open("classifier.pkl", "wb") # create a binary file, open it and then only we can save the model

# wb -> opening file in writing mode
pickle.dump(classifier, pickle_out) # saving of the trained model into pkl file


# close the file
pickle_out.close()
classifier.predict([[2,3,4,1]])



#######################################main.py#############################################33



import uvicorn #taking ASGI request
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd



#create app object
app = FastAPI()
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message':"hello world"}

#giving names
@app.get("/{name}")
def get_name(name:str):
    return {'welcome to this':f'{name}'}

@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    print(data)
    print("Hello")
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    print("Hello")
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction = 'Fake note'
    else:
        prediction = 'its a bank note'
    return{
        'prediction':prediction
    }


if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)







