# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import numpy as np
import _addval as av

FUNCS_TO_RUN = [
        ("addVal1", av.addVal1),
        ("addVal2", av.addVal2),
        ("addVal3", av.addVal3),
]

for name, func in FUNCS_TO_RUN:
    print("\n%s:\nInput array =" % name)
    a = np.random.randint(0, 30, 10).astype(np.int32).reshape((2, 5))
    print(a)
    print("")
    val = 100
    func(a, val)
    print("\n=> After %s(a, %d), array is now =" % (name, val))
    print(a)
    print("\n---")


(name, func) = ("addValEachDelta", av.addValEachDelta)
print("\n%s:\nInput array =" % name)
a = np.random.randint(0, 30, 10).astype(np.int32).reshape((2, 5))
print(a)
print("")
val = 100
incDelta = 3
func(a, val, incDelta)
print("\n=> After %s(a, %d, %d), array is now =" % (name, val, incDelta))
print(a)
print("\n---")
