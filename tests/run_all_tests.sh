#!/bin/sh

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
