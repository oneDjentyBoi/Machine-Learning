import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

flower=load_iris()
x_train, x_test, y_train, y_test = train_test_split(flower.data,flower.target,test_size=0.2)
model=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=1000)
model.fit(x_train,y_train)
model.score(x_test,y_test)
b=list(model.predict([flower.data[98]]))
print(flower.target_names[b[0]])