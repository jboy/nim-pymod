#!/usr/bin/env python

# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import numpy as np
import _doresize as dr

a = np.arange(420).reshape(10, 6, 7)
print a.shape
b = dr.copyAndResizeAndPrintShape(a)
print b.shape, '\n'

a = np.arange(420, dtype=np.uint8).reshape(10, 6, 7)
print a.shape
b = dr.copyAndResizeAndPrintShape(a)
print b.shape, '\n'

a = np.arange(420, dtype=np.int16).reshape(10, 6, 7)
print a.shape
b = dr.copyAndResizeAndPrintShape(a)
print b.shape, '\n'

a = np.arange(420, dtype=np.float32).reshape(10, 6, 7)
print a.shape
b = dr.copyAndResizeAndPrintShape(a)
print b.shape, '\n'

print "--------------------------\n"

a = np.arange(6, dtype=np.int16).reshape(3, 2)
print a
print a.shape
b = dr.copyAndResizeAndPrintShape(a)
print b
print b.shape, '\n'

a = np.arange(6, dtype=np.float32).reshape(3, 2)
print a
print a.shape
b = dr.copyAndResizeAndPrintShape(a)
print b
print b.shape, '\n'

print "--------------------------\n"

a = np.arange(30, dtype=np.int16).reshape(3, 10)
print a
print a.shape
b = dr.copyAndResizeAndPrintShape(a)
print b
print b.shape, '\n'

a = np.arange(30, dtype=np.float32).reshape(3, 10)
print a
print a.shape
b = dr.copyAndResizeAndPrintShape(a)
print b
print b.shape, '\n'

print "==========================\n"

a = np.arange(6, dtype=np.int16).reshape(2, 3)
print a
print a.shape
b = dr.copyAndResizeNumRowsAndPrintShape(a, 5)
print b
print b.shape, '\n'

a = np.arange(6, dtype=np.float32).reshape(2, 3)
print a
print a.shape
b = dr.copyAndResizeNumRowsAndPrintShape(a, 5)
print b
print b.shape, '\n'

print "--------------------------\n"

a = np.arange(30, dtype=np.int16).reshape(10, 3)
print a
print a.shape
b = dr.copyAndResizeNumRowsAndPrintShape(a, 5)
print b
print b.shape, '\n'

a = np.arange(30, dtype=np.float32).reshape(10, 3)
print a
print a.shape
b = dr.copyAndResizeNumRowsAndPrintShape(a, 5)
print b
print b.shape, '\n'

