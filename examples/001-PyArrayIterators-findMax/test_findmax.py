# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import numpy as np
import _findmax as fm

FUNCS_TO_RUN = [
        ("findMax1", fm.findMax1),
        ("findMax2", fm.findMax2),
        ("findMax3", fm.findMax3),
        ("findMax4", fm.findMax4),
        ("findMax5", fm.findMax5),
        ("findMax6", fm.findMax6),
        ("findMax7", fm.findMax7),
        ("findMax8", fm.findMax8),
        ("findMax9", fm.findMax9),
]

for name, func in FUNCS_TO_RUN:
    print("\n%s:\nInput array =" % name)
    a = np.random.randint(0, 30, 10).astype(np.int32).reshape((2, 5))
    print(a)
    print("")
    m = func(a)
    print("\n=> Max is %d\n\n---" % m)


# Now, we'll demonstrate an exception being raised due to the wrong dtype.
print("\nAn exception will be raised due to the wrong dtype:\nInput array =")
a = np.random.randint(0, 30, 10).astype(np.float32).reshape((2, 5))
print(a)
print("")
m = fm.findMax1(a)  # Uh-oh!  ValueError will be raised here!
print("=> Max is %d" % m)
