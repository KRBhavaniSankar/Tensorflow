# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:24:00 2018
@author: Bhavani
"""

"""
1.	
The example generates some artificial/toy data, fits a linear regression model
and computes the accuracy before and after training. This is a toy dataset for
the purposes of illustration, the model is not generalizing/learning as the data is
generated randomly.
	2.	
We define a function to compute L2 regularization as covered in listing 4-8
earlier.
	3.	
We define a function for squared error as covered in listing 4-7.
	4.	
We generate input data, which consists of 1000 vectors of dimensionality 100.
Basically, 1000 examples with 100 features.
	5.	
We generate random target/output labels values between 0 and 1.
	6.	
We define the expressions for linear regression involving the data (denoted by x),
the outputs (denoted by y), the bias term (denoted by b) and the weight vector
(denoted by w). The weight vector and the bias term are shared variables.
	7.	
We compute the prediction, the error and the loss using squared error as
introduced in listing 4-7 earlier.
	8.	
Having defined these expressions, we can now use the grad construct in Theano
(introduced in listing (4-6) to compute the gradient.
	9.	
We define a train function based on the gradient function. The train function defines
the inputs, outputs and how the internal state (shared variables) are to be updated.
	10.	
The train function is invoked for a 1000 steps, in each step the gradient is
computed internally and the shared variables are updated.
	11.	
Root mean squared error (RMSE) is computed before and after the training steps
using sklearn.metrics

"""

import numpy
import theano
import theano.tensor as T
import sklearn.metrics

def l2(x):
    return T.sum(x**2)
def squared_error(x,y):
    return (x - y) ** 2

examples = 1000
features = 100
D = (numpy.random.randn(examples, features), numpy.random.randn(examples))
training_steps = 1000

x =T.dmatrix("x")
y =T.dvector("y")
w =theano.shared(numpy.random.randn(features), name="w")
b =theano.shared(0., name="b")

p = T.dot(x, w) + b
error = squared_error(p,y)
loss = error.mean() + 0.01 * l2(w)
gw, gb = T.grad(loss, [w, b])
train = theano.function(inputs=[x,y],outputs=[p, error], updates=((w, w - 0.1 * gw),
(b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=p)
print("RMSE before training:", sklearn.metrics.mean_squared_error(D[1],predict(D[0])))
for i in range(training_steps):
    prediction, error = train(D[0], D[1])
print("RMSE after training:", sklearn.metrics.mean_squared_error(D[1],predict(D[0])))