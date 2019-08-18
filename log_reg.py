import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
digits=load_digits()
x_train, x_test, y_train, y_test = train_test_split ( digits.data, digits.target, test_size=0.2)
model=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=10000)

model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.predict([digits.data[67]]))

y_pred=model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
