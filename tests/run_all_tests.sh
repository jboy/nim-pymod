#!/bin/sh

PYTEST_COMPAT="000-return_string_HelloWorld"
PYTEST_MODULE="pytest"
#for pyver in python python2 python3
for pyver in python3
do
	if "$pyver" --version >/dev/null 2>/dev/null
	then
		echo "Found '$pyver' executable => We will run Pytest with $pyver"

		if ! "$pyver" -c "import $PYTEST_MODULE" >/dev/null 2>/dev/null
		then
			echo "Can't find Pytest testing framework: Python import '$PYTEST_MODULE' failed" >&2
			echo 'Aborting tests.' >&2
			exit 1
		fi
		"$pyver" -m "$PYTEST_MODULE" "$@" "$PYTEST_COMPAT"
	fi
done
