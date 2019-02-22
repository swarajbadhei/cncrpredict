#from google.colab import files
import pandas as pd
#file=files.upload()
X_train=pd.read_csv("xtrain.csv",header=None)
Y_train=pd.read_csv("ytrain.csv",header=None)
X_test=pd.read_csv("xtest.csv",header=None)
Y_test=pd.read_csv("ytest.csv",header=None)
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(units=16,activation='relu',input_dim=30))
classifier.add(Dense(units=8,activation='relu'))
classifier.add(Dense(units=6,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy')
classifier.fit(X_train,Y_train,batch_size=1,epochs=20);
Y_pred=classifier.predict(X_test)
Y_pred=[1 if y>=0.5 else 0 for y in Y_pred]
total=0
correct=0
wrong=0
for i in Y_pred:
  total+=1
  if(Y_test.at[i,0]==Y_pred[i]):
    correct+=1
  else:
    wrong+=1
print("total tried=",total)
print("total correctly predicted=",correct)
print("total wrongly classified=",wrong)
