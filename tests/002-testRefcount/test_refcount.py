#!/usr/bin/env python

# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import _refcount
import numpy as np
import sys

def refcount(obj):
    # Subtract 3 from the refcount, because each function call adds 1.
    # (+1 for this func, +1 for `sys.getrefcount`, +1 for some other reason?
    # Perhaps the 3rd +1 is for the function implementing the `-` operator?)
    return sys.getrefcount(obj) - 3


def check_refcount(varname, var, expected_rc):
    rc = refcount(var) - 2  # and -2 more for this function...
    print "Get refcount(%s) -> %d" % (varname, rc)
    assertion = "refcount(%s) == %d" % (varname, expected_rc)
    assert (rc == expected_rc), assertion
    print "TEST PASSED:", assertion


a = np.arange(20, dtype=np.int32).reshape((5, 4))
print "a ="
print a
check_refcount("a", a, 1)
print "\nb = _refcount.createsome(a)"
print "> Crossing into Nim..."
b = _refcount.createsome(a)
print "\n... and we're back in Python"
check_refcount("a", a, 1)
check_refcount("b", b, 1)
print "b ="
print b

print "\n------------\n"
c = np.arange(20, 40, dtype=np.int32).reshape((5, 4))
print "c ="
print c
check_refcount("c", c, 1)
print "\nd = _refcount.identity(c)"
print "> Crossing into Nim..."
d = _refcount.identity(c)
print "\n... and we're back in Python"
check_refcount("c", c, 2)
check_refcount("d", d, 2)

print "\n------------\n"
e = np.arange(20, 40, dtype=np.int32).reshape((5, 4))
print "e ="
print e
check_refcount("e", e, 1)
print "\n_refcount.identity(e)"
print "> Crossing into Nim..."
_refcount.identity(c)
print "\n... and we're back in Python"
check_refcount("e", e, 1)

print "\n------------\n"
print "f = _refcount.identity(np.arange(...))"
print "> Crossing into Nim..."
f = _refcount.identity(np.arange(40, 60, dtype=np.int32).reshape((5, 4)))
print "\n... and we're back in Python"
check_refcount("f", f, 1)

print "\n------------\n"
g = np.arange(60, 80, dtype=np.int32).reshape((5, 4))
h = np.arange(80, 100, dtype=np.int32).reshape((5, 4))
print "g ="
print g
check_refcount("g", g, 1)
print "\nh ="
print h
check_refcount("h", h, 1)
print "\ni = _refcount.twogoinonecomesout(g, h)"
print "> Crossing into Nim..."
i = _refcount.twogoinonecomesout(g, h)
print "\n... and we're back in Python"
check_refcount("g", g, 1)
check_refcount("h", h, 2)
check_refcount("i", i, 2)
print "i ="
print i

