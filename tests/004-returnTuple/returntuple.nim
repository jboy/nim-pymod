# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import pymod

proc foo*(a, b: int): tuple[c, d: int] {. exportpy .} =
  return (a, b)

proc bar*(a, b: int = 3): tuple[c, d: int] {. exportpy .} =
  return (a, b)

proc baz*(a: int, b: int = 3): tuple[c, d: int] {. exportpy .} =
  return (a, b)

initPyModule("_returntup", foo, bar, baz)
