'''
Logistic regression implementation from scratch using sigmoid function
'''
import numpy as np
import matplotlib.pyplot as plt
import math
def sigmoid(t):
	return (1/(1+math.exp(-t)))
def sigmoid_util(z):
	size=len(z)
	i=0
	while(i<size):#computes sigmoid value element wise. z[i]=g(th(0)*x(0) + th(1)*x(1)) 
		z[i]=sigmoid(z[i])
		i+=1
	return z
def GradDec(x,y,itr,alpha,theta): # normal gradient descent. Here just the def of hypothesis changes from lin reg.
	m=len(y)
	for _ in range(0,itr):
		h=sigmoid_util(x.dot(theta))
		theta=theta-(alpha/m)*(x.T.dot(h-y))
	return theta
def main():
	data=np.genfromtxt('breast_cancer.txt', delimiter=',')
	x=np.array([data[:,0]]).transpose()
	y=np.array([data[:,1]]).transpose()
	i=0
	alpha=0.001
	itr=50000
	s=len(x)
	on=np.ones([s,1],dtype=int)
	theta=np.array([[0],[0]])
	x=np.concatenate((on,x),axis=1)
	theta=GradDec(x,y,itr,alpha,theta)
	prob=sigmoid_util(x.dot(theta))# probablilty estimation done on the same dataset. You can import your own dataset.
	while(i<len(prob)):
		if(prob[i]<0.5):
			print('probablilty={} so begnin'.format(prob[i]))
		else:
			print('probablilty={} so malignant'.format(prob[i]))
		i+=1
if __name__=='__main__':
	main()

		