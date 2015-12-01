#!/bin/sh
# Run all tests in the Pymod automated test suite.
#
# Usage:
#  sh run_all_tests.sh
#
# The test suite depends upon the Pytest framework: https://pytest.org/latest/
# The test suite will attempt to run using every major version of Python found
# on your system.
#
# To run just a single specific test, for example "001-return_1":
#  sh run_all_tests.sh 001-return_1
#
# To see all output produced by the tests runs (including from the Nim compiler
# and the Pymod "pmgen.py" script):
#  sh run_all_tests.sh -s


PYTEST_MODULE="pytest"
for pyver in python python2 python3
do
	if "$pyver" --version >/dev/null 2>/dev/null
	then
		echo "Found '$pyver' executable => We will run Pytest with $pyver"

		if ! "$pyver" -c "import $PYTEST_MODULE" >/dev/null 2>/dev/null
		then
			echo "Can't find Pytest testing framework: Python import '$PYTEST_MODULE' failed" >&2
			echo 'Aborting all tests.' >&2
			exit 1
		fi
		if ! "$pyver" -m "$PYTEST_MODULE" "$@"
		then
			echo
			echo "Test failure with '$pyver' executable" >&2
			echo 'Aborting all tests.' >&2
			exit 1
		fi
	fi
	echo
done
