# -*- coding: utf-8 -*-
"""
Created on Mon May 14 17:27:00 2018
@author: Bhavani
"""

'''
IMPORTANT :

1.	
The example generates some artificial/toy data, fits a logistic regression model
and computes the accuracy before and after training. This is a toy dataset for the
purposes of illustration; the model is not generalizing/learning, as the data is
generated randomly.
	2.	
We define a function to compute L2 regularization as covered in listing 4-8
earlier.
	3.	
We generate input data, which consists of 1000 vectors of dimensionality 100.
Basically, 1000 examples with 100 features.
	4.	
We generate random target/output labels as zeros and ones.
	5.	
We define the expressions for logistic regression involving the data (denoted
by x), the outputs (denoted by y), the bias term (denoted by b), and the weight
vector (denoted by w). The weight vector and the bias term are shared variables.
	6.	
We compute the prediction, the error, and the loss using binary cross entropy as
introduced in listing 4-7 earlier.
	7.	
Having defined these expressions, we can now use the grad construct in Theano
(introduced in listing (4-6)) to compute the gradient.
	8.	
We define a train function based on the gradient function. The train function
defines the inputs, outputs, and how the internal state (shared variables) are to
be updated.
	9.	
The train function is invoked for 1000 steps; in each step the gradient is
computed internally and the shared variables are updated.
	10.	
Accuracy is computed before and after the training steps using sklearn.metrics

'''

import numpy

import theano
import theano.tensor as T
import sklearn.metrics

def l2(x):
    return T.sum(x ** 2)

examples = 1000
features = 100


D = (numpy.random.randn(examples,features),numpy.random.randint(size=examples,low=0,high=2))
training_steps = 1000
x = T.dmatrix('x')
y = T.dvector('y')
w = theano.shared(numpy.random.randn(features),name='w')
b = theano.shared(0.,name='b')

p = 1/(1+ T.exp(-T.dot(x,w)-b))
error =  T.nnet.binary_crossentropy(p,y)

loss = error.mean() +  0.01 * l2(w)
prediction = p >0.5
gw,gb = T.grad(loss,[w,b])

train = theano.function(inputs=[x,y],outputs= [p,error],updates=((w,w-0.1 * gw),(b,b-0.1 * gb)))
predict = theano.function(inputs=[x],outputs=prediction)

print('Accuracy before training :',sklearn.metrics.accuracy_score(D[1],predict(D[0])))

for i in range(training_steps):
    prediction,error = train(D[0],D[1])

print('Accuracy After training :',sklearn.metrics.accuracy_score(D[1],predict(D[0])))






