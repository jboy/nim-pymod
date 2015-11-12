# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import _returntup as r

def run_test(curr_test, test_num, num_tests):
    (func, func_descr, args_tup, expected_result) = curr_test
    try:
        actual_result = func(*args_tup)
    except TypeError as e:
        if expected_result == TypeError:
            print "%02d / %d:  %s%s  ->  exception TypeError  (as expected).  TEST PASSED." % \
                    (test_num, num_tests, func_descr, args_tup)
        else:
            print "\n** %02d / %d:  %s%s  ->  exception TypeError  (which was NOT expected).  TEST FAILED.  :(\n" % \
                    (test_num, num_tests, func_descr, args_tup)
        return
    except Exception as e:
        print "\n** %02d / %d:  %s%s  ->  exception %s  (which was NOT expected).  TEST FAILED.  :(\n" % \
                (test_num, num_tests, func_descr, args_tup, e.__class__.__name__)
        return

    # If we got to here, there was no exception raised.  So, check the result.
    if actual_result == expected_result:
        print "%02d / %d:  %s%s  ->  %s  (as expected).  TEST PASSED." % \
                (test_num, num_tests, func_descr, args_tup, actual_result)
    else:
        print "\n** %02d / %d:  %s%s  ->  %s  (which was NOT expected).  TEST FAILED.  :(\n" % \
                (test_num, num_tests, func_descr, args_tup, actual_result)


TESTS = [
        (r.foo, "foo", (), TypeError),
        (r.foo, "foo", (1,), TypeError),
        (r.foo, "foo", (1, 2), (1, 2)),
        (r.foo, "foo", (1, 2, 4), TypeError),

        (r.bar, "bar", (), (3, 3)),
        (r.bar, "bar", (1,), (1, 3)),
        (r.bar, "bar", (1, 2), (1, 2)),
        (r.bar, "bar", (1, 2, 4), TypeError),

        (r.baz, "baz", (), TypeError),
        (r.baz, "baz", (1,), (1, 3)),
        (r.baz, "baz", (1, 2), (1, 2)),
        (r.baz, "baz", (1, 2, 4), TypeError),
]

for i, t in enumerate(TESTS):
    run_test(t, i+1, len(TESTS))
