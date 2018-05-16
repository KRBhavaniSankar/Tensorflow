
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:01:00 2018
@author: Bhavani
"""



"""
1.	
Computation of certain functions requires iterative constructs for which Theano
provides the scan construct.
	2.	
In our example we compute the power operation with the scan construct and
match the output with using the standard operator for power.
	3.	
Expressions and functions can be defined using the scan construct and gradients
can be generated as with other expressions/constructs.

"""
import theano
import theano.tensor as T
import theano.printing

k = T.iscalar("k")
a = T.dscalar("a")

result,update = theano.scan(fn= lambda prior_result , a:prior_result * a ,outputs_info=a,non_sequences=a,n_steps=k-1)
final_result = result[-1]

a_pow_k = theano.function(inputs=[a,k],outputs=final_result,updates=update)
print(a_pow_k(2,5), 2 ** 5)
print(a_pow_k(2,5),2 ** 5)
