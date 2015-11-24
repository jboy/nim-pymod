# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## Casting is bad, and unchecked pointer arithmetic is worse, and
## casting pointers so you can perform unchecked pointer arithmetic
## is just the worst of all.
##
## Don't use any of the functions in this module, ever, or you will be
## the worst sort of person.


proc offset_ptr*[T](p: ptr T, idx: int): ptr T {. noSideEffect, inline .} =
  ## Return a ``ptr T`` instance that is offset from the pointer value
  ## of argument ``p: ptr T`` by ``idx`` multiples of ``sizeof(T)``.
  ##
  ## ``idx`` can be any ``int`` value that is positive, zero or negative.
  ## No bounds checking will be performed.
  ##
  # Yay casting.  Everybody loves casting.
  # Nim defines its `int` type to be the same size as pointers:
  #  http://nim-lang.org/system.html#int
  let ip = cast[int](p)
  let ip_offset = ip +% idx * sizeof(T)
  return cast[ptr T](ip_offset)


proc offset_ptr*[T](p: ptr T): ptr T {. noSideEffect, inline .} =
  ## Return a ``ptr T`` instance that is offset from the pointer value
  ## of argument ``p: ptr T`` by exactly ``sizeof(T)``.
  ##
  ## No bounds checking will be performed.
  ##
  # Yay casting.  Everybody loves casting.
  # Nim defines its `int` type to be the same size as pointers:
  #  http://nim-lang.org/system.html#int
  let ip = cast[int](p)
  let ip_offset = ip +% sizeof(T)
  return cast[ptr T](ip_offset)


proc offset_var_ptr*[T](p: var ptr T, idx: int) {. noSideEffect, inline .} =
  ## Modify argument ``p: var ptr T`` in-place, to be offset by ``idx``
  ## multiples of ``sizeof(T)`` from its original pointer value.
  ##
  ## ``idx`` can be any ``int`` value that is positive, zero or negative.
  ## No bounds checking will be performed.
  ##
  # Yay casting.  Everybody loves casting.
  # Nim defines its `int` type to be the same size as pointers:
  #  http://nim-lang.org/system.html#int
  let ip = cast[int](p)
  let ip_offset = ip +% idx * sizeof(T)
  p = cast[ptr T](ip_offset)


proc offset_var_ptr*[T](p: var ptr T) {. noSideEffect, inline .} =
  ## Modify argument ``p: var ptr T`` in-place, to be offset by ``sizeof(T)``
  ## from its original pointer value.
  ##
  ## No bounds checking will be performed.
  ##
  # Yay casting.  Everybody loves casting.
  # Nim defines its `int` type to be the same size as pointers:
  #  http://nim-lang.org/system.html#int
  let ip = cast[int](p)
  let ip_offset = ip +% sizeof(T)
  p = cast[ptr T](ip_offset)


proc subtract_ptr*[T](p1, p2: ptr T): int {. noSideEffect, inline .} =
  ## Return an ``int`` value of the difference between the pointer values
  ## of the arguments ``p1: ptr T`` and ``p2: ptr T``.
  ##
  ## The return value will be a difference in multiples of ``sizeof(T)``,
  ## NOT in bytes.
  ##
  ## An ``int`` will always be sufficient to represent this difference,
  ## because Nim defines its `int` type to be the same size as pointers:
  ##  http://nim-lang.org/system.html#int
  ##
  # Yay casting.  Everybody loves casting.
  let ip1 = cast[int](p1)
  let ip2 = cast[int](p2)
  return (ip1 - ip2) div sizeof(T)


proc offset_ptr_in_bytes*[T](p: ptr T, num_bytes: int): ptr T {. noSideEffect, inline .} =
  ## Return a ``ptr T`` instance that is offset from the pointer value of
  ## argument ``p: ptr T`` by ``num_bytes`` bytes.
  ##
  ## ``num_bytes`` can be any ``int`` value that is positive, zero or negative.
  ## No bounds checking will be performed.
  ##
  # Yay casting.  Everybody loves casting.
  # Nim defines its `int` type to be the same size as pointers:
  #  http://nim-lang.org/system.html#int
  let ip = cast[int](p)
  let ip_offset = ip +% num_bytes
  return cast[ptr T](ip_offset)


template offset_void_ptr_in_bytes*(p: pointer, num_bytes: int): pointer =
  ## Return a ``pointer`` instance that is offset from the pointer value of
  ## argument ``p: pointer`` by ``num_bytes`` bytes.
  ##
  ## ``num_bytes`` can be any ``int`` value that is positive, zero or negative.
  ## No bounds checking will be performed.
  ##
  # Yay casting.  Everybody loves casting.
  # Nim defines its `int` type to be the same size as pointers:
  #  http://nim-lang.org/system.html#int
  let ip = cast[int](p)
  let ip_offset = ip +% num_bytes
  cast[pointer](ip_offset)


proc offset_var_ptr_in_bytes*[T](p: var ptr T, num_bytes: int) {. noSideEffect, inline .} =
  ## Modify argument ``p: var ptr T`` in-place, to be offset by ``num_bytes``
  ## bytes from its original pointer value.
  ##
  ## ``num_bytes`` can be any ``int`` value that is positive, zero or negative.
  ## No bounds checking will be performed.
  ##
  # Yay casting.  Everybody loves casting.
  # Nim defines its `int` type to be the same size as pointers:
  #  http://nim-lang.org/system.html#int
  let ip = cast[int](p)
  let ip_offset = ip +% num_bytes
  p = cast[ptr T](ip_offset)

