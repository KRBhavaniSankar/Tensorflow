# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:01:00 2018
@author: Bhavani
"""

'''
1.	
All models (deep learning or otherwise) will involve defining functions with
internal state, which will typically be weights that need to be learned or fitted.
	2.	
A shared variable is defined using the shared construct in Theano.
	3.	
A shared variable can be initialized with Numpy constructs.
	4.	
Once the shared variable is defined and initialized, it can be used in the
definition of expressions and functions in a manner similar to scalars and
vectors/matrices, as we have seen earlier.
	5.	
A user can get the value of the shared variable using the get_value method

6.	
A user can set the value for the shared variable using the set_value method.
	7.	
A function defined using the shared variable computes its output based on
the current value of the shared variable. That is, as soon as the shared variable
updates, a function defined using the shared variable will produce a different
value for the same input.
	8.	
A shared variable allows a user to define a function with internal state, which can
be updated arbitrarily without needing to redefine the function defined using the
shared variable.

'''
import theano.tensor as T
from theano import function
from theano import shared

import numpy
x = T.dmatrix('x')
y = shared(numpy.array([[4,5,6]]))
z = x + y
f = function(inputs= [x],outputs=[z])

print('original shared value :',y.get_value())
print('original function evaluation :',f([[1,2,3]]))

y.set_value(numpy.array([[5,6,7]]))

print('original shared value',y.get_value())
print('original Function evaulation :',f([[1,2,3]]))