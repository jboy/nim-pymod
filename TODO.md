* Placate all the new Nim compiler warnings that have appeared recently.
* Allow the user to specify their own choice of build directory name rather than `pmgen`.
  * Ultimately, this should be something that can be specified in "pymod-extensions.cfg".
  * For now, start with a command-line option `--pmgen_dir=DIRPATH`.
  * Note: After the Python module `.so` file is built, the Makefile moves it to `..`.
  * So `pmgen.py` needs to write the current absolute path, rather than `..`, into the Makefile.
* Add a command-line option `--pymod_name=MODNAME` to override the modname that is specified in `initPyModule`.
* Enable `void` as a Nim type that is supported for params & return values.
* If the user returns a `nil` string, ensure it is converted to an empty string.
* If the user returns a `nil` ptr PyObject or ptr PyArrayObject, ensure it becomes PyNone.
* Enable `bool` as a Nim type that is supported for params & return values.
  * Note that the Python C-API doesn't actually offer `bool` as a conversion type.
  * https://docs.python.org/2/c-api/arg.html
  * https://docs.python.org/2/c-api/bool.html
  * http://stackoverflow.com/questions/9316179/what-is-the-correct-way-to-pass-a-boolean-to-a-python-c-extension
* After that, enable `bool` parameter to be a default parameter.
  * Start at proc `getDefaultValue` in "pymodpkg/private/impls.nim".
* When a `ptr PyArrayObject` is being returned from an `exportpy`-ed Nim proc, annotate it `not nil`.
  * http://nim-lang.org/docs/manual.html#types-not-nil-annotation
  * This will enable compile-time checking to ensure you haven't forgotten to assign a result.
  * Otherwise, the C-API conversion-back-to-Python code receives a NULL pointer, and fails.
* Do the tests in the "tests" directory properly.  Probably using Python's `unittest` or `nose2`?
  * http://www.drdobbs.com/testing/unit-testing-with-python/240165163
  * http://pythontesting.net/framework/unittest/unittest-introduction/
  * https://docs.python.org/release/2.7/library/unittest.html
  * https://docs.python.org/2/library/unittest.html
  * https://nose2.readthedocs.org/en/latest/index.html
  * http://docs.python-guide.org/en/latest/writing/tests/
* Implement the procs planned for implementation in "pymodpkg/pyarrayobject.nim".
* Enable Python types to be passed from Python into Nim, so we can pass `numpy.int32` as an argument for example.
* Implement reading of double-hash comments into docstrings using `nim jsondoc myfile.nim`?
  * http://nim-lang.org/docs/docgen.html#document-types-json
* Read & use `pymod-extensions.cfg`!
* Fix all the millions of `FIXME` notes scattered through the code...
  * In general, a `FIXME` is there because I did something in a hacky way, that I think I should re-do properly.
