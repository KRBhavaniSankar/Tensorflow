# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:46:00 2018
@author: Bhavani
"""

"""
1.	
Certain functions need an if-else (or switch) clause for their evaluation. For such
cases Theano provides an if-else and switch constructs.
	2.	
Expressions and functions can be defined using the if-else and switch constructs
and gradients can be generated as with other expressions/constructs.
	3.	
In the example we demonstrate the computation of the hinge lose using the
if-else and switch construct and verify that it matches to the one defined
with max.
"""

import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse

def hinge_a(x,y):
    return T.max([0 * x, 1-x *y])

def hinge_b(x,y):
    return ifelse(T.lt(1-x*y,0), 0 * x, 1-x*y)

def hinge_c(x,y):
    return T.switch(T.lt(1-x*y,0), 0 * x, 1-x*y)

x = T.dscalar("x")
y = T.dscalar("y")

z1 = hinge_a(x,y)
z2 = hinge_b(x,y)
z3 = hinge_c(x,y)

f1 = theano.function([x,y],z1)
f2 = theano.function([x,y],z2)
f3 = theano.function([x,y],z3)


print("f(-2, 1) =",f1(-2, 1), f2(-2, 1), f3(-2, 1))
print("f(-1,1 ) =",f1(-1, 1), f2(-1, 1), f3(-1, 1))
print("f(0,1) =",f1(0, 1), f2(0, 1), f3(0, 1))
print("f(1, 1) =",f1(1, 1), f2(1, 1), f3(1, 1))
print("f(2, 1) =",f1(2, 1), f2(2, 1), f3(2, 1))


"""
output
-------
('f(-2, 1) =', array(3.0), array(3.0), array(3.0))
('f(-1,1 ) =', array(2.0), array(2.0), array(2.0))
('f(0,1) =', array(1.0), array(1.0), array(1.0))
('f(1, 1) =', array(0.0), array(0.0), array(0.0))
('f(2, 1) =', array(0.0), array(0.0), array(0.0))

"""