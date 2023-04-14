import numpy as np
import matplotlib.pyplot as plt
from numpy import random


#Defining the dataset
X=np.array([[0,0,0,-1],
   [0,0,1,1],
   [1,0,0,1],
   [0,1,1,1],
   [1,1,0,1],
   [1,0,1,1],
   [1,1,1,1],
   [0,1,0,1]])



#Training the OR function using perceptron algortith 
def train():
	w1,w2,w3,b=(0,0,0,0)#assign 0 to the weights
	#w1= np.random.uniform(-50,50)
	#w2= np.random.uniform(-50,50)
	#w3= np.random.uniform(-50,50)
	#b= np.random.uniform(-50,50)

	while(True):# perceptron algorithm
		print('weights are:', w1,w2,w3,b)#print the weights
		iteration=0
		for x1,x2,x3,y in X: #training using the entire truth table
			inp = x1*w1+x2*w2+x3*w3 +b
			if((y*inp)<=0):#if the point is missclassified update the weight
				w1+=y*x1
				w2+=y*x2
				w3+=y*x3
				b+=y
				iteration+=1
			
		if(iteration==0):
			break
	return w1,w2,w3, b#return the updated weights



#assigning the trained values
w1,w2,w3,b = train()

#Defining the OR function as a perceptron
def func_OR(x1,x2,x3):
	inp= x1*w1 + x2*w2 + x3*w3 +b
	output = 1 if inp>=0 else -1 #output with 1,-1 activation
	return output

#testing the OR function
print (X)
for x1,x2,x3,y in X:
	print('OR gate truth table output as per perceptron:',func_OR(x1,x2,x3))
 
#Plotting the 3d space, selecting 5 points using linspace command for the 2 indpendent variables and working out the value of the 3rd dependent variable. 
fig = plt.figure()
ax = plt.axes(projection='3d')#axes explicitly accessed for 3d graph
x1 = np.linspace(-0.1,1.1,5)#randomly assigning 5 inputs to x1 from -0.1 to 1, the points are lying close to dataset  
x2 = np.linspace(-0.1,1.1,5)#randomly  assigning 5 inputs to x2 from -0.1 to 1
x1, x2 = np.meshgrid(x1,x2)#create a 2-d array each from 1-d array of x1 and x2
x3 = -(w1*x1+w2*x2+b)/w3# x3 also becomes 2-d, perceptron trained weights used
xuser = -(3*x1+3*x2-1)/3#user defined weights w1,w2,w3,b = 3,3,3,-1 applied to inputs, 2-d array
ax.plot_wireframe(x1,x2,x3)#draw decision plane using perceptron algorithm
ax.plot_wireframe(x1,x2,xuser)#draw userdefined decision plane
ax.scatter(X[1:,0], X[1:,1], X[1:,2], c ='r', s = 80)#plot the 9 TRUE values of the truth table on the 3-d projection, slicing the dataset from 0th row)
ax.scatter(X[0,0], X[0,1], X[0,2], c = 'y', s = 80, marker ='^')#plot the FALSE value of truth table, corrsponding to 0th row of the dataset
plt.show()