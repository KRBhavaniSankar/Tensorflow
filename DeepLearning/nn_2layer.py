# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:08:00 2018
@author: Bhavani
"""

"""
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
We define the expressions for the 2-layer neural network involving the data
(denoted by x), the outputs (denoted by y), the bias term of the first layer
(denoted by b1), the weight vector of the first layer (denoted by w1), the bias term
of the second layer (denoted by b2), and, the weight vector of the second layer
(denoted by w2). The weight vectors and the bias terms are shared variables.
	6.	
We use the tanh activation function as covered in listing 4-4 to encode the neural
network.
	7.	
We compute the prediction, the error, and the loss using binary cross entropy as
introduced in listing 4-7 earlier.
	8.	
Having defined these expressions, we can now use the grad construct in Theano
(introduced in listing (4-6)) to compute the gradient.
	9.	
We define a train function based on the gradient function. The train function
defines the inputs, outputs, and how the internal state (shared variables) are to
be updated.
10.	
The train function is invoked for 1000 steps; in each step the gradient is
computed internally and the shared variables are updated.
	11.	
Accuracy is computed before and after the training steps using sklearn.metrics
"""
import numpy
import theano
import theano.tensor as T
import sklearn.metrics

def l2(x):
    return T.sum(x ** 2)

examples = 1000
features = 100
hidden = 10

D = (numpy.random.randn(examples,features),numpy.random.randint(size=examples,low=0,high=2))

training_steps = 1000
x = T.dmatrix("x")
y = T.dvector("y")

w1 = theano.shared(numpy.random.randn(features,hidden),name="w1")
b1 = theano.shared(numpy.zeros(hidden),name="b1")

w2 = theano.shared(numpy.random.randn(hidden),name="w2")
b2 = theano.shared(0.,name="b2")

p1 = T.tanh(T.dot(x,w1) +b1)
p2 = T.tanh(T.dot(p1,w2)+b2)

prediction = p2 > 0.5
error = T.nnet.binary_crossentropy(p2,y)
loss = error.mean() + 0.01 *(l2(w1) + l2(w2))

gw1, gb1, gw2, gb2 = T.grad(loss, [w1, b1, w2, b2])
train = theano.function(inputs=[x,y],outputs=[p2, error], updates=((w1, w1 - 0.1 * gw1),
(b1, b1 - 0.1 * gb1), (w2, w2 - 0.1 * gw2), (b2, b2 - 0.1 * gb2)))
predict = theano.function(inputs=[x], outputs=[prediction])
print("Accuracy before Training:", sklearn.metrics.accuracy_score(D[1], numpy.array(predict(D[0])).ravel()))
for i in range(training_steps):
     prediction, error = train(D[0], D[1])

print("Accuracy after Training:",sklearn.metrics.accuracy_score(D[1],numpy.array(predict(D[0])).ravel()))