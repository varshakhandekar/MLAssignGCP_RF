"""In this assignment students will build the random forest model after
normalizing the variable to house pricing from boston data set."""
import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,r2_score,mean_squared_error
import pickle
df=load_boston()
df.feature_names
df.target
data=pd.DataFrame(df.data,columns=df.feature_names)
data.columns
data['House_Price']=df['target']
fig=plt.figure(figsize=(20,30))
cor_matrix=data.corr().round(2)
sns.heatmap(data=cor_matrix,annot=True)
X=data[['RM','LSTAT','PTRATIO','INDUS']]
y=data['House_Price']
X.head()
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.30,random_state=345)
model=RandomForestRegressor()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
r2score=r2_score(ytest,ypred)
print('R2 Score: ',r2score)
mse=mean_squared_error(ytest,ypred)
print('Mean Squared Error:',mse)
from sklearn import model_selection
import math
import numpy as  np
kfold=model_selection.KFold(n_splits=10,random_state=42,shuffle=True)
cv = model_selection.cross_val_score(model, xtrain, ytrain, cv=kfold) #scoring=scoring)
print('Mean score',np.mean(cv))
sqrt_cv = [math.sqrt(abs(i)) for i in cv]
print("{} ({})".format( np.mean(sqrt_cv), np.std(sqrt_cv)))
print('Result from each iteration of cross validation:', cv, '\n')
filename="MLAssign_RandomForest.pickle"
model=pickle.dump(model,open(filename,"wb"))
import numpy as np
narr=np.array([6.57,4.98,15.3,2.31])
narr=narr.reshape(1,-1)
loadedmodel=pickle.load(open(filename,"rb"))
predprice=loadedmodel.predict(narr)
print(f'Predicted House Price = ', predprice[0])


