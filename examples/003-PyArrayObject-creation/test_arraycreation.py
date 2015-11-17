# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import numpy as np
import _arraycreation as ac

a = ac.arange_int32(123)
print("a = ac.arange_int32(123)\nprint(a)\n%s" % a)
print("\nprint(a.dtype)\n%s" % a.dtype)
print("\nprint(a.shape)\n%s" % str(a.shape))

z = ac.zeros_like(a)
print("\nz = ac.zeros_like(a)\nprint(z)\n%s" % z)
print("\nprint(z.dtype)\n%s" % z.dtype)
print("\nprint(z.shape)\n%s" % str(z.shape))

num_rows = 2
num_cols = 5
init_val = 17
img = ac.createRgbImage(num_rows, num_cols, init_val)
print("\nimg = ac.createRgbImage(%d, %d, %d)\nprint(img)\n%s" %
        (num_rows, num_cols, init_val, img))
print("\nprint(img.dtype)\n%s" % img.dtype)
print("\nprint(img.shape)\n%s" % str(img.shape))
