# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:08:00 2018
@author: Bhavani
"""
'''
1.	
A function needs to be defined using expressions before the gradient of the
function can be generated.
	2.	
The grad construct in Theano allows the user to generate the gradient of a
function (as an expression). Users can then define a function over the expression,
which gives them the gradient function.
	3.	
Gradients can be computed for any set of expressions/functions as in the earlier
examples. So, for instance, we could generate gradients for functions with a
shared state. As the shared state updates, so do the function and the gradient
function.
'''
import theano.tensor as T
from theano import function
from theano import shared

import numpy
x = T.dmatrix('x')
y= shared(numpy.array([[4,5,6]]))

z = T.sum(((x * x) + y) * x)

f = function(inputs=[x],outputs=[z])

g = T.grad(z,[x])
g_f = function([x],g)
print("Original:", f([[1, 2, 3]]))
print("Original Gradient:", g_f([[1, 2, 3]]))

y.set_value(numpy.array([[1, 1, 1]]))
print("Updated:", f([[1, 2, 3]]))
print("Updated Gradient", g_f([[1, 2, 3]]))

#Original: [array(68.0)]
#Original Gradient: [array([[  7.,  17.,  33.]])]
#Updated: [array(42.0)]
#Updated Gradient [array([[  4.,  13.,  28.]])]