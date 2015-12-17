# Pymod

Auto-generate a [Python](http://www.python.org) module that wraps
a [Nim programming language](http://nim-lang.org) module.

The **Pymod** software consists of Nim bindings & Python scripts to automate
the generation of
[Python C-API extension module](https://docs.python.org/2/extending/index.html)
boilerplate for Nim "procs" (procedures).
After the Pymod script has been run, there will be an auto-generated, auto-compiled
Python module that exposes the Nim procs in Python.

There's even a type (`PyArrayObject`) that provides a Nim interface to
[Numpy arrays](http://www.scipy-lectures.org/intro/numpy/array_object.html),
so you can pass Numpy arrays into your Nim procs from Python and access them
natively.

The auto-generated C-API boilerplate code handles the parsing & type-checking
of the function arguments passed from Python, including correct handling of Python
ref-counts if a type error occurs or an exception is raised.  The boilerplate
code also translates Nim exceptions (including back-traces) to Python exceptions.
The boilerplate code even includes auto-generated Python docstrings that have been
extracted from the Nim procs.

Pymod is definitely still in the **in-development** phase of software maturity,
and it's far from feature-complete, but it's been usable for our work for about
9 months now (and we've been using it regularly during that time).  There's a lot
of hacky code in there, but it gets the job done.

Table of contents
-----------------

1. [Motivation](#motivation)
2. [Nim](#nim)
3. [Example](#example)
4. [Usage](#usage)
5. [System requirements](#system-requirements)
6. [Per-project configuration](#per-project-configuration)
7. [Procedure parameter & return types](#procedure-parameter--return-types)
8. [Docstrings](#docstrings)
9. [PyArrayObject type](#pyarrayobject-type)
10. [PyArrayIterator types](#pyarrayiterator-types)
11. [PyArrayObject & PyArrayIterator usage example](#pyarrayobject--pyarrayiterator-usage-example)
12. [PyArrayIterator loop idioms](#pyarrayiterator-loop-idioms)
13. [Tips, warnings & gotchas](#tips-warnings--gotchas)
14. [What about calling Python from Nim?](#what-about-calling-python-from-nim)
15. [Implementation details](#implementation-details)

Motivation
----------

Perhaps you have a large body of existing Python code, that you can't or
don't want to rewrite.  Perhaps you want to use [Numpy](http://www.numpy.org/),
[Scipy](http://www.scipy.org/) or [Matplotlib](http://www.matplotlib.org).
Perhaps your program's main loop simply must be in Python.

However, you would like to write Nim code and then call your Nim procs from
your Python code.  (There are many more systems written in Python than in Nim,
but it would be great to start extending them in Nim!)
You can write
[Python C-API extension modules](https://docs.python.org/2/extending/extending.html)
to wrap your Nim procs, but all the
[C-API boilerplate](https://docs.python.org/2/c-api/)
is a huge drag, especially if you check types and manage reference counts
and handle Nim exceptions properly.

That's what Pymod is for.

Nim
---

If you'd like to learn more about the [Nim programming language](http://nim-lang.org),
we recommend:

* the official [Nim tutorial, part 1](http://nim-lang.org/docs/tut1.html)
  (followed by [part 2](http://nim-lang.org/docs/tut2.html))
* the very approachable [Nim by Example](https://nim-by-example.github.io/),
  which offers a series of short, simple lessons about the main Nim features
* the comprehensive [Nim manual](http://nim-lang.org/docs/manual.html)
* the Python-vs-Nim feature comparison table at
  [Nim for Python Programmers](https://github.com/nim-lang/Nim/wiki/Nim-for-Python-Programmers)

Example
-------

Here's a short "Hello world" example (assumed to be in a file called `greeting.nim`):

```Nimrod
## Compile this Nim module using the following command:
##   python path/to/pmgen.py greeting.nim

import strutils  # `%` operator

import pymod
import pymodpkg/docstrings

proc greet*(audience: string): string {.exportpy.} =
  docstring"""Greet the specified audience with a familiar greeting.

  The string returned will be a greeting directed specifically at that audience.
  """
  return "Hello, $1!" % audience

initPyModule("hw", greet)
```

Use the Python script `pmgen.py` to auto-generate & compile the boilerplate code:

    python path/to/pmgen.py greeting.nim

There will now be a compiled Python extension module `hw.so` in the current directory.
(It is called `hw` because that is the name that was specified in the `initPyModule()` macro).

In a Python interpreter, you can import the module and invoke the `greet` function:

```Python
>>> import hw
>>> hw.greet
<built-in function greet>
>>> hw.greet("World")
'Hello, World!'
>>>
```

You can also invoke the built-in Python interpreter `help` function about the `greet` function:

```Python
>>> help(hw.greet)
Help on built-in function greet in module hw:

greet(...)
    greet(audience: str) -> (str)

    Parameters
    ----------
    audience : str -> string

    Returns
    -------
    out : (str) <- (string)

    Greet the specified audience with a familiar greeting.

    The string returned will be a greeting directed specifically at that audience.
```

There is additional example code in the
[examples](https://github.com/jboy/nim-pymod/tree/master/examples) directory.

Usage
-----

Using Pymod is a 4-step process.  In brief:

1. `import pymod` at the top of your Nim module.
2. Add the `{.exportpy.}` pragma after each Nim proc.
3. Invoke the `initPyModule("modname", proc1, proc2, proc3)` macro at the bottom of your Nim module.
4. Run the `pmgen.py` Python script to compile everything.

In more detail:

1. At the top of your Nim module, import the module `pymod`.
  * **Tip:** You might additionally wish to import `pymodpkg/docstrings` (to
    enable Python-like docstrings) and/or `pymodpkg/pyarrayobject` (to enable
    the `PyArrayObject` type that corresponds to Numpy's array type).
2. In your Nim module, annotate the `{.exportpy.}` pragma onto each Nim proc
   to be exported to Python.  (This pragma is named by analogy with the
   standard Nim `{.exportc.}` pragma).
3. At the end of your Nim module, configure the Python module to be generated,
   using the `initPyModule()` macro:  Specify the desired Python module name as
   a string (without a filename suffix), followed by the names of the Nim procs
   that should be compiled into the Python module.
  * For example, if you supply the string `"foo"` as the first argument to
    `initPyModule()`, the generated Python module will be called `foo.so`.
  * **Tip:** You can use the `initPyModule()` macro multiple times at the end
    of your Nim module, with different Python module names & different
    combinations of Nim procs, to generate multiple Python modules.
  * **Tip:** If you specify the empty string `""` as the Python module name,
    the generated Python module will have the same name as the Nim module,
    but with an underscore `"_"` prepended.  So for example, a Nim module
    `bar.nim` would be compiled to a Python module `_bar.so`.
4. Invoke the supplied Python script `pmgen.py`, supplying the filename of
   your Nim module as a command-line argument, to auto-generate & invoke a
   set of Makefiles that will in turn initiate & run the Pymod process.

When the script `pmgen.py` is run, it will create a subdirectory `pmgen` in
the current directory.  All the source code auto-generated by Pymod will be
placed into this subdirectory and compiled.  At the end of the compilation
process, the new Python module, a `.so` (shared object) file, will be moved
back into the current directory.

**Note** that the `{.exportpy.}` pragma & `initPyModule()` macro are
**inert by default** (that is, they have no effect), so you can add them to
existing Nim code without changing the default operation of that Nim code.
It's only when you run the script `pmgen.py` (which, among other actions,
supplies the switch `--define:pmgen` to the Nim compiler) that the
`{.exportpy.}` pragma & `initPyModule()` macro are activated.

System requirements
-------------------

* The [latest Nim compiler from Github](http://nim-lang.org/download.html#installation-from-github)
  * either the `master` or `devel` branches
  * but **not the recent [Nim 0.12.0 release](http://nim-lang.org/news.html#Z2015-10-27-version-0-12-0-released)**. :(
  * (A [suggested fix for this Nim 0.12.0 packaging problem](http://forum.nim-lang.org/t/1797/2#11256)
    has been proposed on the Nim forums.)
* CPython 2.7 or CPython 3.2+
* Python C development header files & static library
* [Numpy](http://www.numpy.org)
* Numpy C development header files
* [Make](https://en.wikipedia.org/wiki/Make_%28software%29)
* a C compiler [for use by Nim](http://nim-lang.org/download.html)

Per-project configuration
-------------------------

If there is a file `pymod.cfg` in the same directory as the Nim module you want
to wrap, Pymod will read this as a configuration file for that project.

By default, Pymod runs the Nim compiler in **non-release** mode, and additionally
performs per-dereference bounds-checking of the `PyArrayObject` iterators.
This is safe (and catches all sorts of pesky bugs!) but slow.

If the file `pymod.cfg` in the current directory contains the following directives:

    [all]
    nimSetIsRelease: true

then the Nim compiler will be invoked in **release mode**, and bounds-checking
of the `PyArrayObject` iterators will be switched off.  Your code will now run
much faster!

Procedure parameter & return types
----------------------------------

The following Nim types are currently supported by Pymod:

| Type family      | Nim types | Python2 type | Python3 type |
| ---------------- | --------- | ------------ | -------------|
| floating-point   | `float`, `float32`, `float64`, `cfloat`, `cdouble` | `float` | `float` |
| signed integer   | `int`, `int16`, `int32`, `int64`, `cshort`, `cint`, `clong` | `int` | `int` |
| unsigned integer | `uint`, `uint8`, `uint16`, `uint32`, `uint64`, `cushort`, `cuint`, `culong`, `byte` | `int` | `int` |
| non-unicode character | `char`, `cchar` | `str` | `bytes` |
| string           | `string` | `str` | `str` |
| Numpy array      | `ptr PyArrayObject` | `numpy.ndarray` | `numpy.ndarray` |

Support for the following Nim types is in development:

| Type family      | Nim types | Python2 type | Python3 type |
| ---------------- | --------- | ------------ | -------------|
| signed integer   | `int8` | `int` | `int` |
| boolean          | `bool` | `bool` | `bool` |
| unicode code point (character) | `unicode.Rune` | `unicode` | `str` |
| non-unicode character sequence | `seq[char]` | `str` | `bytes` |
| unicode code point sequence    | `seq[unicode.Rune]` | `unicode` | `str` |
| sequence of a single type _T_  | `seq[T]` | `list` | `list` |


Procedure parameters may be any of the above supported Nim types.
Default parameters are supported to a limited extent, although the parameter
type must be specified explicitly, and is currently restricted to the `string`,
integer & floating-point types.

Procedure return values may be any of the above supported Nim types or
a Nim tuple of any of these types.  Nested tuples are currently not supported.
By default, named tuples in Nim are returned as raw tuples to Python:

    # Nim                   # Python
    tuple[ a, b: int ]  =>  (a_value, b_value)

If `{.exportpy.}` is specified as `{.exportpy returnDict.}` then the generated
code will instead return a Python dict containing the named properties:

    # Nim                   # Python
    tuple[ a, b: int ]  =>  { "a": a_value, "b": b_value }

You can tell Pymod about additional Nim types using the `definePyObjectType()`
macro.  This will include your additional type-mapping in Pymod's type-mapping
registry, similar to how Pymod maps its own `PyArrayObject` type to Numpy's
array type.  This provides Pymod with a mapping from an already-defined Nim
type to the corresponding Python & C-API types, enabling Pymod to generate
the Nim-Python conversions & type-checking boilerplate for additional types.

Docstrings
----------

Pymod will also auto-generate a Python docstring for each function in the
extension module, specifying the function's parameter types & return type,
based upon the parameter types & return types of the exported Nim proc.
You can embed additional documentation in each Nim proc you want to export,
using the supplied `docstring"""Text goes here."""` string type.  Any docstrings
in the proc will be extracted automatically and included in the generated Python
docstring.  There is an example of docstring usage in the code sample above.

PyArrayObject type
------------------

Pymod provides the `PyArrayObject` type to allow Python code to pass
[Numpy ndarrays](http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html)
into Nim procs with appropriate type-safety.  To access the `PyArrayObject`
Nim type definition, import `pymodpkg/pyarrayobject` in your Nim module (after
you have already imported `pymod`).

Because the Numpy array object was allocated in Python, the type of the Nim
proc parameter or return value is `ptr PyArrayObject`.  **Note** that it is a
Nim `ptr`, not a Nim `ref`.  Your code should pass around `ptr PyArrayObject`.

Pymod also wraps many C functions from the
[Numpy C-API](http://docs.scipy.org/doc/numpy/reference/c-api.html)
for Numpy array manipulation & attribute access.
To review the full list of `PyArrayObject` procs that Pymod provides, browse
the Pymod source file `pymodpkg/pyarrayobject.nim`.

Here are some of the Numpy array attributes that Pymod exposes:

* `.data` (returns `pointer`)
* `.data(T)` (returns `ptr T`)
* `.descr`
* `.dimensions`
* `.dtype`
* `.nd`
* `.ndim` (an alias for `.nd`)
* `.shape` (an alias for `.dimensions`)
* `.strides`

Here are some of the Numpy functions for array creation & manipulation that Pymod wraps:

* `createSimpleNew(dims, npType)`
* `createNewCopyNewData(oldArray, order)`
* `copy(oldArray)`  (an alias for `createNewCopyNewData`)
* `createAsTypeNewData(oldArray, newType)`
* `doCopyInto(destArray, srcArray)`
* `doFILLWBYTE(destArray, val)`
* `doResizeDataInplace(oldArray, newShape, doRefCheck)`

PyArrayIterator types
---------------------

**Note** that, due to its typeless Pythonic origin, `PyArrayObject` is not a
Nim generic type.  So the element data-type of a `PyArrayObject` instance
is unknown to Nim.  The Nim code must **specify the correct element data-type**
for the `PyArrayObject` elements.  The preferred method of accessing the
(appropriately-typed) elements of a `PyArrayObject` instance is to use one of
the two supplied `PyArrayIterator` types:

* `PyArrayForwardIter[T]`
  * returned by `.iterateFlat(T)`
  * can only be incremented & dereferenced
  * the fastest & safest iteration style
* `PyArrayRandAccIter[T]`
  * returned by `.accessFlat(T)`
  * can be incremented or decremented by any integer; offset (using `+` or `-`) by any integer; indexed by any integer; & dereferenced
  * basically a C pointer with bounds-checking

Both of the `PyArrayIterator` types offer **1-D iteration & indexing** over
a "flat" interpretation of the Numpy N-D array data.  These two iterator types
are inspired by the
[C++ iterator category model](http://www.cplusplus.com/reference/iterator/).
By default, the iterators implement per-dereference bounds-checking.
This bounds-checking can be disabled, as described above in the section
[Per-project configuration](#per-project-configuration).

**Note** that the `PyArrayIterator` types can't handle any of the following
usage scenarios:

 * non-C-contiguous array data
 * strides
 * multi-dimensional indexing

If you attempt to iterate over a Numpy array with non-C-contiguous data,
an `AssertionError` will be raised (even in release mode).  If you supply
the incorrect array element data-type when invoking `.iterateFlat(T)`
or `.accessFlat(T)`, an `ObjectConversionError` will be raised (even in
release mode).

PyArrayObject & PyArrayIterator usage example
---------------------------------------------

Here is a simple example of how to use `PyArrayObject` & `PyArrayForwardIter[T]`:

```Nimrod
import strutils  # `%`
import pymod
import pymodpkg/docstrings
import pymodpkg/pyarrayobject

proc addVal*(arr: ptr PyArrayObject, val: int32) {.exportpy} =
  docstring"""Add `val` to each element in the supplied Numpy array.

  The array is assumed to have dtype `int32`; otherwise, a ValueError will be
  raised.  The elements in the array will be modified in-place.
  """
  let dt = arr.dtype
  echo "PyArrayObject has shape $1 and dtype $2" % [$arr.shape, $dt]
  if dt == np_int32:
    let bounds = arr.getBounds(int32)  # Iterator bounds
    var iter = arr.iterateFlat(int32)  # Forward iterator
    while iter in bounds:
      iter[] += val
      inc(iter)  # Increment the iterator manually.
  else:
    let msg = "expected array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)

initPyModule("_myModule", addVal)
```

You can test the Pymod-wrapped Nim proc `addVal` using a Python script like this:

```Python
import numpy as np
import _myModule as mm

int32arr = np.arange(10, dtype=np.int32).reshape((2, 5))
print(int32arr)
mm.addVal(int32arr, 101)
print(int32arr)

print("")

float32arr = np.arange(10, dtype=np.float32).reshape((2, 5))
print(float32arr)
mm.addVal(float32arr, 101)  # Uh-oh!  Our `addVal` proc wants an array with dtype == `np.int32`!
print(float32arr)
```

The output from running this script will look something like this:

```Python
[[0 1 2 3 4]
 [5 6 7 8 9]]
PyArrayObject has shape @[2, 5] and dtype numpy.int32
[[101 102 103 104 105]
 [106 107 108 109 110]]

[[ 0.  1.  2.  3.  4.]
 [ 5.  6.  7.  8.  9.]]
PyArrayObject has shape @[2, 5] and dtype numpy.float32
Traceback (most recent call last):
  File "test_addvalmod.py", line 13, in <module>
    mm.addVal(float32arr, 101)  # Uh-oh!  Our `addVal` proc wants an array with dtype == `np.int32`!
ValueError: expected array of dtype numpy.int32, received dtype numpy.float32
Nim traceback (most recent call last):
  File "pmgen_myModule_wrap.nim", line 26, in exportpy_addVal
  File "addvalmod.nim", line 22, in addVal
```

PyArrayIterator loop idioms
---------------------------

Observe the `while` loop that was used in `addVal` to iterate over the array.
This is the most flexible loop idiom for forward-iterating over an array,
since you are able to control where, and how many times, the forward iterator
will be incremented within the body of the loop:

```Nimrod
let bounds = arr.getBounds(int32)  # Iterator bounds
var iter = arr.iterateFlat(int32)  # Forward iterator
while iter in bounds:
  iter[] += val
  inc(iter)  # Increment the iterator manually
```

However, this `while` loop idiom is more verbose than it often needs to be.
Often, you will only need to increment the forward iterator once per iteration,
at the end of the body of the loop; if this is all you need, there is a shorter
`for` loop idiom that you can use:

```Nimrod
for iter in arr.iterateFlat(int32):
  iter[] += val
```

And if you don't need to modify the array data at all, there is an even shorter
`for` loop idiom that yields a succession of (read-only) array values:

```Nimrod
var maxVal: int32 = low(int32)
for val in arr.values(int32):
  if val > maxVal:
    maxVal = val
```

Likewise for `PyArrayRandAccIter[T]`, there are 4 loop idioms, which
offer different levels of control & convenience.  For the most flexibility,
use a `while` loop:

```Nimrod
let bounds = arr.getBounds(int32)  # Iterator bounds
var iter = arr.accessFlat(int32)  # Random access iterator
while iter in bounds:
  iter[] += val
  inc(iter, incDelta)  # Increment the iterator manually
```

There are 3 different `for` loop forms available for `PyArrayRandAccIter[T]`,
to make it convenient to iterate over C-contiguous N-dimensional arrays.  If you want to
visit every iterator position in turn, but retain the ability to index/offset arbitrarily,
use the 1-argument form of `accessFlat`:

```Nimrod
for iter in arr.accessFlat(int32):
  iter[] += val
```

If you want to increment by a certain specific delta each time, use the
2-argument form of `accessFlat`:

```Nimrod
for iter in arr.accessFlat(int32, incDelta):
  iter[] += val
```

And finally, if you want the iteration to begin at a certain initial offset,
then increment by a certain specific delta each time, use the 3-argument form
of `accessFlat`:

```Nimrod
for iter in arr.accessFlat(int32, initialOffset, incDelta):
  iter[] += val
```

For example, if you want to visit just the "green" channel of an RGB image,
you might use a loop like this:

```Nimrod
let greenIdx = 1
let numChans = img.shape[2]
for iter in img.accessFlat(uint8, greenIdx, numChans):
  processGreenComponent(iter[])
```

These code examples are all available in full in the
[examples](https://github.com/jboy/nim-pymod/tree/master/examples) directory.

Tips, warnings & gotchas
------------------------

Here are some helpful hints about a few sharp edges of Pymod (some of them due
to sharp edges in Nim that we haven't been able to cover over completely) that
can trip you up (and then confuse you with obscure compiler error messages):

* If you want to `exportpy` a proc using Pymod, **don't** give your proc the
  same name as the Nim module that contains the proc (or in fact, the same name
  as any other procs in that same module).
* If you want to `exportpy` a proc using Pymod, ensure that the proc is also
  [exported in Nim](http://nim-lang.org/docs/manual.html#procedures-export-marker)
  by marking it with an asterisk after the proc-name.

What about calling Python from Nim?
-----------------------------------

Pymod enables you to wrap your Nim procs so they can be called from Python.

If instead you want to call Python functions (maybe even the interpreter)
from Nim (ie, the control flows in the opposite direction), Pymod is not
what you're looking for.

In a situation like this, [python.nim](https://github.com/nim-lang/python)
or the [NimBorg project](https://github.com/micklat/NimBorg) might be what
you're looking for.

Implementation details
----------------------

We want to make existing Python types (and extended types like
[Numpy arrays](https://scipy-lectures.github.io/intro/numpy/array_object.html))
available in Nim procs.  The idea is that objects of these Python types will
be created in the existing Python code, then passed through to Nim procs for
processing, and then the results will be returned to the Python code for the
pre-existing processing or viewing in Python.

For this reason, we decided not to use the
[Python ctypes module](https://docs.python.org/2/library/ctypes.html);
the focus of `ctypes` seems to be on propagating types in the opposite
direction.  Instead of making fully-fledged Python types easily available
in C, `ctypes` wraps lowest-common-denominator C types and structs for
basic access in Python.

The way
[ctypes handles pointers](https://docs.python.org/2/library/ctypes.html#pointers)
& [structs](https://docs.python.org/2/library/ctypes.html#structures-and-unions)
requires you to build your object types up in terms of C primitive types,
so they can be accessed as method-less, field-only objects in Python.
But the Python types we want to use in Nim are already defined in Python!
So `ctypes` would be more applicable to us if we wanted to make existing
Nim data-structures available in Python, rather than the other way around.

On an initial reading,
[Python CFFI](http://cffi.readthedocs.org/en/release-0.8/) &
[cffiwrap](http://cffiwrap.readthedocs.org/en/latest/) appear to be more
useful for exposing
[existing Python types in C](http://cffi.readthedocs.org/en/latest/using.html#reference-conversions)
than `ctypes` is.

However, for the initial implementation of Pymod, we chose the
[CPython C-API](https://docs.python.org/2/extending/index.html) &
[Numpy C-API](http://docs.scipy.org/doc/numpy-1.10.0/reference/c-api.html)
over `cffi` and/or `cffiwrap`.
We reasoned that Nim is going to produce & auto-compile its own C code anyway,
so if we generate C-API C code for our Python integration, this C code can be
compiled & linked into a shared library by the Nim compiler itself -- thus
ensuring that the same compiler settings are used for both the compilation &
linking stages.

We are considering implementing an additional `cffi` back-end for
Pymod in the future.  This would be a significant step towards compatibility
with the [PyPy Python interpreter](http://pypy.org/).

