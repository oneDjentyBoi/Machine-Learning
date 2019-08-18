import numpy as np 
import matplotlib.pyplot as plt 
def GradDec(x,y,itr,alpha,theta):
	m=len(y)
	for _ in range(1,itr):
		h=np.dot(x,theta)
		theta=theta-(alpha/m)*(x.T.dot(h-y))
	return theta
def analytical(x,y):
	theta=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
	return theta

def main():
	data=np.genfromtxt("document.txt",delimiter=",")
	x=np.array([data[:,0]]).transpose()
	y=np.array([data[:,1]]).transpose()
	theta=np.array([[0],[0]])
	itr=1500
	alpha=0.01
	plt.scatter(x,y)
	s=len(x)
	on=np.ones([s,1],dtype=int)
	x=np.concatenate((on,x),axis=1)
	theta=GradDec(x,y,itr,alpha,theta)
	#theta=analytical(x,y)
	y=x.dot(theta)
	plt.plot(x[:,0],y,color='#FF0000')
	plt.show()
	
	print(theta)

if __name__=="__main__":
	main()