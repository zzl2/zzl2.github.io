---
title: sklearn
date: 2017-12-05 10:39:39
tags:  
---

{%asset_img ml_map.png %}

话不多说先上一张大图

小插曲：我把文件名弄成了sklearn ，然后在调用sklearn的时候估计系统懵逼了不知道我要用谁！！导致我吧数据集又重新安装了在卸载！

OK ！ 开心的敲代码！

sklearn的基本操作

```python
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target
# print(iris_X[:2,:])
# print(iris_y)
X_train,X_test,y_train,y_test=train_test_split(
    iris_X,iris_y,test_size=0.3
)
# print(y_train)
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
print(knn.predict(X_test))
print(y_test)
```

{%asset_img   2017-12-05_143558.png %}

```python
from sklearn import datasets
from sklearn.linear_model import LinearRegression
loaded_data=datasets.load_boston()
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)
print(model.predict(data_X[:4,:]))
print(data_y[:4])
```

{%asset_img 2017-12-05_144429.png%}

生成数据并图像化

```python
import matplotlib.pyplot as plt
X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=1)
plt.scatter(X,y)
plt.show()
```

{%asset_img 2017-12-05_145254.png%}

看生成的模型的参数

```python
print(model.coef_)#y=0.1x0.3   -> 0.1
print(model.intercept_)  #-->0.3
```

```python
print(model.get_params()) #定义的模型的参数{'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False}
print(model.score(data_X,data_y))# R^2 cofficient of determination 0.740607742865
```

normalization 标准化

```python
from sklearn import preprocessing
import numpy as np
a=np.array([[10,2.7,3.6],
            [-100,5,-2],
            [120,20,40]],dtype=np.float64)
print(a)
print(preprocessing.scale(a))
```

{%asset_img 2017-12-05_151418.png%}

采用了normalization 后得分是0.96666666，不采用只有0.5

```python
from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt
X,y=make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,random_state=22,
                        n_clusters_per_class=1,scale=100)
X=preprocessing.scale(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
clf=SVC()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
```

交叉验证

```python
iris=datasets.load_iris()
X=iris.data
y=iris.target
from sklearn.cross_validation import cross_val_score
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4)
knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,X,y,cv=5,scoring='accuracy')#分成5组
print(scores.mean())
```

利用交叉验证和可视化工具来 选择KNN的K

```python
k_range=range(1,31)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')#for classification
    loss=-cross_val_score(knn,X,y,cv=10,scoring='mean_squared_error')#for regression
    k_scores.append(scores.mean())
plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
```

{%asset_img 2017-12-05_154705.png%}

损失函数的时候是这样，其他的model  直接换掉KNN

{% asset_img 2017-12-05_155106.png %}

过拟合问题

```python
from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_digits

digits=load_digits()
X=digits.data
y=digits.target
train_size,train_loss,test_loss=learning_curve(SVC(gamma=0.001),
                                               X,y,cv=10,scoring='mean_squared_error',
               train_sizes=[0.1,0.25,0.5,0.75,1])
train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)
plt.plot(train_size,train_loss_mean,'o-',color='r',label='Traing'
         )
plt.plot(train_size,test_loss_mean,'o-',color='g',label='Cross-validation'
         )
plt.xlabel('Training examples')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()
```

{%asset_img  2017-12-05_161902.png%}

如何选择模型的参数 gamma

```python
from sklearn.learning_curve import validation_curve
from sklearn.datasets import load_digits

digits=load_digits()
X=digits.data
y=digits.target
param_range=np.logspace(-6,-2.3,5) #5代表5个点
train_loss,test_loss=validation_curve(SVC(), X,y,param_name='gamma',
                                      param_range=param_range,cv=10,scoring='mean_squared_error',
                            )
train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)
plt.plot(param_range,train_loss_mean,'o-',color='r',label='Traing'
         )
plt.plot(param_range,test_loss_mean,'o-',color='g',label='Cross-validation'
         )
plt.xlabel('gamma')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()
```

{%asset_img 2017-12-05_163810.png%}

模型训练完毕以后的保存

```python
from sklearn import svm
from sklearn import datasets
clf=svm.SVC()
iris=datasets.load_iris()
X=iris.data
y=iris.target
clf.fit(X,y)
#method 1:pickle
import pickle
with open('save/clf.pickle','wb') as f:
    pickle.dump(clf,f)
```

{%asset_img   2017-12-05_165208.png%}

再次用的时候

```python
with open('save/clf.pickle','rb') as f:
    clf2=pickle.load(f)
```

```python
#method2 :joblib
from sklearn.externals import joblib
#Save
joblib.dump(clf,'save/clf.pkl')
#restore
clf3=joblib.load('save/clf.pkl')
print(clf3.predict(X[0:1]))
```

{%asset_img 2017-12-05_165858.png%}