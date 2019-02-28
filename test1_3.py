import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('/home/hari/Downloads/Documents/iris.csv')
df=df.sample(frac=1).reset_index(drop=True)

le=LabelEncoder()
df.Species=le.fit_transform(df.Species)

df.replace(0,np.NaN)
df.fillna(df.mean(),inplace=True)

def z_score(df):
    df.columns = [x + "_zscore" for x in df.columns]
    return ((df - df.mean())/df.std())
z_score(df)

array=df.values
x_cols= [0,1,2,3]
X=array[:,x_cols]
Y = array[:,4]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import  classification_report


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,Y_train)

Y_pred=gnb.predict(X_test)

from sklearn import metrics

print("GNB accuracy: ",metrics.accuracy_score(Y_test,Y_pred))


print(confusion_matrix(Y_test,Y_pred))

from sklearn.svm import SVC
clf=SVC(kernel='linear')

clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)


print("SVM accuracy:", metrics.accuracy_score(Y_test, Y_pred))

print(confusion_matrix(Y_test,Y_pred))


from sklearn import linear_model
logreg=linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train,Y_train)
Y_pred=logreg.predict(X_test)

print("Logistic regressio on SVM accuracy: ",metrics.accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
