# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import numpy as np
import _addval as av

FUNCS_TO_RUN1 = [
        ("addVal1", av.addVal1),
        ("addVal2", av.addVal2),
        ("addVal3", av.addVal3),
]

for name, func in FUNCS_TO_RUN1:
    print("\n%s:\nInput array =" % name)
    a = np.random.randint(0, 30, 10).astype(np.int32).reshape((2, 5))
    print(a)
    print("")
    val = 100
    func(a, val)
    print("\n=> After %s(a, %d), array is now =" % (name, val))
    print(a)
    print("\n---")


FUNCS_TO_RUN2 = [
        ("addValEachDelta1", av.addValEachDelta1),
        ("addValEachDelta2", av.addValEachDelta2),
]

for name, func in FUNCS_TO_RUN2:
    print("\n%s:\nInput array =" % name)
    a = np.random.randint(0, 30, 10).astype(np.int32).reshape((2, 5))
    print(a)
    print("")
    val = 101
    incDelta = 3
    func(a, val, incDelta)
    print("\n=> After %s(a, %d, %d), array is now =" % (name, val, incDelta))
    print(a)
    print("\n---")


FUNCS_TO_RUN3 = [
        ("addValEachDeltaInitOffset", av.addValEachDeltaInitOffset),
]

for name, func in FUNCS_TO_RUN3:
    print("\n%s:\nInput array =" % name)
    a = np.random.randint(0, 30, 10).astype(np.int32).reshape((2, 5))
    print(a)
    print("")
    val = 101
    initOffset = 1
    incDelta = 3
    func(a, val, initOffset, incDelta)
    print("\n=> After %s(a, %d, %d), array is now =" % (name, val, incDelta))
    print(a)
    print("\n---")
