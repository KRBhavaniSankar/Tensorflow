# -*- coding: utf-8 -*-
"""
Created on Mon May 14 17:02:00 2018
@author: Bhavani
"""

'''
1.	
There are cases/situations where we want to define functions having a random
variable (for instance introducing minor corruptions in inputs).
	2.	
Such a random element in the function is different from having an internal state,
like in the case of shared variables.
3.	
Basically, the desired outcome in such cases/situations is that the user wants to
define a function with a random variable with a particular distribution.
	4.	
Theano provides a construct called RandomStreams, which allows the user to
define functions with a random variable. RandomStreams is initialized with a
seed.
	5.	
The user defines a variable using RandomStreams and specifies a distribution by
calling the appropriate function (in our case, normal).
	6.	
Once defined, the random variable can be used in the definition of expression or
functions in a manner similar to scalars and vectors/matrices.
	7.	
Every invocation of the function defined with a random variable will internally
draw a sample point from the set distribution (in our case, normal).

'''

import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import  RandomStreams
import numpy

random = RandomStreams(seed=42)
a = random.normal((1,3))
b = T.dmatrix('a')
f1 = a * b
g1 = function([b],f1)

print("Invocation 1:", g1(numpy.ones((1,3))))
print("Invocation 2:", g1(numpy.ones((1,3))))
print("Invocation 3:", g1(numpy.ones((1,3))))

# Invocation 1: [[ 1.25614218 -0.53793023 -0.10434045]]
# Invocation 2: [[ 0.66992188 -0.70813926  0.99601177]]
# Invocation 3: [[ 0.0724739  -0.66508406  0.93707751]]