#!/usr/bin/env python

# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import sys
import numpy as np

import _copying


PYMOD_DTYPES = [
        np.bool,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64
]

SHAPES = [
        ((600, 800),    None),
        ((600, 800, 3), None),
        ((600, 800, 3), (None, None, 0)),
        ((600, 800, 3), (None, None, 1)),
        ((600, 800, 3), (None, None, 2)),
]

def getSliceFromTup(tup):
    return tuple([
            slice(None) if e is None else e
            for e in tup])

def testCopyArray():
    for sh, sli in SHAPES:
        for dt in PYMOD_DTYPES:
            in_arr = np.ones(sh, dtype=dt)
            if sli is None:
                in_arr_sliced = in_arr
                was_sliced = False
            else:
                in_arr_sliced = in_arr[getSliceFromTup(sli)]
                was_sliced = True
            out_arr = _copying.copyArray(in_arr_sliced)

            assert in_arr_sliced.dtype == out_arr.dtype
            assert in_arr_sliced.shape == out_arr.shape
            if was_sliced:
                assert in_arr.shape != out_arr.shape
                # TODO:  Assert `in_arr_sliced` is NOT contiguous
            else:
                assert in_arr.shape == out_arr.shape
                # TODO:  Assert `in_arr_sliced` is contiguous
            # TODO:  Assert `in_arr` is contiguous
            # TODO:  Assert `out_arr` is contiguous


def testAsTypeArray():
    for dt1 in PYMOD_DTYPES:
        target_dtype = np.arange(1, dtype=dt1).dtype
        target_typenum = np.arange(1, dtype=dt1).dtype.num

        for sh, sli in SHAPES:
            for dt in PYMOD_DTYPES:
                in_arr = np.ones(sh, dtype=dt)
                if sli is None:
                    in_arr_sliced = in_arr
                    was_sliced = False
                else:
                    in_arr_sliced = in_arr[getSliceFromTup(sli)]
                    was_sliced = True

                out_arr = _copying.asTypeArray(in_arr_sliced, target_typenum)

                assert out_arr.dtype == target_dtype
                assert out_arr.dtype.num == target_typenum
                if target_dtype == dt:
                    assert in_arr_sliced.dtype == out_arr.dtype
                else:
                    assert in_arr_sliced.dtype != out_arr.dtype

                assert in_arr_sliced.shape == out_arr.shape
                if was_sliced:
                    assert in_arr.shape != out_arr.shape
                    # TODO:  Assert `in_arr_sliced` is NOT contiguous
                else:
                    assert in_arr.shape == out_arr.shape
                    # TODO:  Assert `in_arr_sliced` is contiguous
                # TODO:  Assert `in_arr` is contiguous
                # TODO:  Assert `out_arr` is contiguous


TEST_FUNCTIONS = [
        ("testCopyArray", testCopyArray),
        ("testAsTypeArray", testAsTypeArray),
]

for func_name, test_func in TEST_FUNCTIONS:
    test_func()
    print("Test passed: %s" % func_name)
