import importlib
import subprocess
import sys

import pytest


_PMGEN_PY_DIRPATH_DEFAULT = ".."
_PMGEN_PY_FNAME_DEFAULT = "pmgen.py"

_PMGEN_DIRNAME_DEFAULT = "pmgen"
_COMPILED_MOD_FNAME_SUFFIX = ".so"
_NIM_MOD_FNAME_SUFFIX = ".nim"
_RENAME_PREFIX_FOR_FAILED_TEST = "failed_"


##
## Add extra information to the test report header.
##

def pytest_report_header(config):
    # https://pytest.org/latest/example/simple.html#adding-info-to-test-report-header

    nim_version_output = subprocess.check_output(["nim", "--version"], stderr=subprocess.STDOUT)
    if sys.version_info.major >= 3:
        # Python 3: `lines` has type `bytes`.
        lines = nim_version_output.split(b"\n")
    else:
        # Python 2: `lines` has type `str`.
        lines = nim_version_output.split("\n")
    return lines[0]


##
## Configure the command-line parser.
##

def pytest_addoption(parser):
    """Add custom command-line options to the command-line parser."""
    # http://pytest.readthedocs.org/en/2.0.3/plugins.html#_pytest.hookspec.pytest_addoption

    # Pytest's option parser is similar to Python's stdlib `argparse`:
    #  https://docs.python.org/2/library/argparse.html#argparse.ArgumentParser.add_argument
    pmgen_py_dirpath_help = \
            "override the directory path to the 'pmgen.py' script " + \
            "from the default of '%(default)s'. The path can be specified " + \
            "as a relative path or absolute path."
    parser.addoption("--pmgen_py_dirpath", type="string", metavar="DIRPATH",
            default=_PMGEN_PY_DIRPATH_DEFAULT, action="store", dest="pmgen_py_dirpath",
            help=pmgen_py_dirpath_help)

    pmgen_py_fname_help = \
            "override the filename of the 'pmgen.py' script " + \
            "from the default of '%(default)s'."
    parser.addoption("--pmgen_py_fname", type="string", metavar="FNAME",
            default=_PMGEN_PY_FNAME_DEFAULT, action="store", dest="pmgen_py_fname",
            help=pmgen_py_fname_help)


##
## Utility functions
##

def _get_option_value(config, option_name, option_default_value):
    cmdline_option_value = config.getoption(option_name)
    if cmdline_option_value and (cmdline_option_value != option_default_value):
        return cmdline_option_value

    inicfg_option_value = config.inicfg.get(option_name)
    if inicfg_option_value and (inicfg_option_value != option_default_value):
        return inicfg_option_value

    return option_default_value


def _localpath_exists(localpath):
    return (localpath.stat(raising=False) is not None)


def _get_test_dir_that_contains_module(request):
    # Note that `request.fspath` returns a LocalPath instance, not a string.
    #  http://py.readthedocs.org/en/latest/path.html#py._path.local.LocalPath
    #
    # But, dumb API design: `.dirname` returns `str` rather than `LocalPath`.
    # So we have to use `.parts(reverse=True)[1]` instead.
    #test_dir = request.fspath.dirname
    test_dir = request.fspath.parts(reverse=True)[1]
    return test_dir


def _get_pymod_test_mod_name(request):
    return "_%s" % request.module.__name__


def _get_pymod_test_mod_fname(request):
    return "_%s%s" % (request.module.__name__, _COMPILED_MOD_FNAME_SUFFIX)


##
## Session fixtures
##

@pytest.fixture(scope="session")
def pmgen_py_fullpath(request):
    """Return the correct fullpath of the "pmgen.py" script for this session.
    This checks for user-specified values in the "pytest.ini" file & in the
    command-line options.  It will double-check that the specified fullpath
    actually exists.
    """
    pmgen_py_dirpath = _get_option_value(request.config, "pmgen_py_dirpath", _PMGEN_PY_DIRPATH_DEFAULT)
    pmgen_py_fname = _get_option_value(request.config, "pmgen_py_fname", _PMGEN_PY_FNAME_DEFAULT)
    print("\nInitializing test session...")
    print(" - pmgen.py dirpath specified: %s" % pmgen_py_dirpath)
    print(" - pmgen.py fname specified: %s" % pmgen_py_fname)

    # Note that `request.config.invocation_dir` returns a LocalPath instance,
    # not a string.
    #  http://py.readthedocs.org/en/latest/path.html#py._path.local.LocalPath
    invocation_dir = request.config.invocation_dir
    pmgen_py_fullpath = invocation_dir.join(pmgen_py_dirpath, pmgen_py_fname, abs=1)
    print(" > pmgen.py fullpath to use: %s\n" % pmgen_py_fullpath)

    # Ensure the specified fullpath actually exists.
    if not _localpath_exists(pmgen_py_fullpath):
        request.raiseerror("file does not exist: %s" % pmgen_py_fullpath)

    return str(pmgen_py_fullpath)


@pytest.fixture(scope="session")
def python_exe_fullpath(request):
    """Return the fullpath of the Python executable to run for this session.
    This also tells us the version of Python (ie, 2 or 3) that is being used.
    """
    # https://docs.python.org/2/library/sys.html#sys.executable
    python_exe_path = sys.executable
    if not python_exe_path:
        # "If Python is unable to retrieve the real path to its executable,
        # `sys.executable` will be an empty string or None."
        request.raiseerror("unable to determine real path to Python executable")

    return python_exe_path


@pytest.fixture(scope="session")
def python_major_version(request):
    """Return the (int) major version of Python being used for this session."""
    # https://docs.python.org/2/library/sys.html#sys.version_info
    return sys.version_info.major


##
## Module fixtures
##

@pytest.fixture(scope="module", autouse=True)
def chdir_into_test_dir(request):
    """Change directory into the correct test directory for this module.
    This also sets a finalizer that will change back to the previous directory
    when the tests in this module are complete (or have failed).  If the tests
    in this module have all passed, the finalizer will also clean up any files
    or subdirectories that were created by Pymod in this test directory.
    """
    print("\nInitializing test module: %s" % request.module.__name__)

    # Note that `request.config.invocation_dir` returns a LocalPath instance,
    # not a string.
    #  http://py.readthedocs.org/en/latest/path.html#py._path.local.LocalPath
    invocation_dir = request.config.invocation_dir
    print("Pytest invocation directory: %s" % invocation_dir)

    test_dir = _get_test_dir_that_contains_module(request)
    print("Chdir into test directory: %s" % test_dir)
    prev_dir = test_dir.chdir()

    def chdir_back_to_starting_dir():
        # Test whether there are any files we should delete before we leave.
        pymod_test_mod_fullpath = test_dir.join(_get_pymod_test_mod_fname(request))
        if _localpath_exists(pymod_test_mod_fullpath):
            print("\nDelete file: %s" % pymod_test_mod_fullpath)
            pymod_test_mod_fullpath.remove()

        pmgen_dir_fullpath = test_dir.join(_PMGEN_DIRNAME_DEFAULT)
        if _localpath_exists(pmgen_dir_fullpath):
            print("Delete directory: %s" % pmgen_dir_fullpath)
            pmgen_dir_fullpath.remove(rec=1, ignore_errors=True)

        print("\nChdir back to previous directory: %s" % prev_dir)
        prev_dir.chdir()
    request.addfinalizer(chdir_back_to_starting_dir)


##
## Function fixtures
##

# This function is copy-pasted from the example of how to
# "[make] test result information available in fixtures" at:
#  https://pytest.org/latest/example/simple.html#making-test-result-information-available-in-fixtures
# and:
#  https://github.com/pytest-dev/pytest/issues/288
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Execute all other hooks to obtain the report object.
    outcome = yield
    rep = outcome.get_result()

    # Set an report attribute for each phase of a call, which can be
    # "setup", "call", "teardown".
    setattr(item, "rep_" + rep.when, rep)


# This function is based upon the example of how to
# "[make] test result information available in fixtures" at:
#  https://pytest.org/latest/example/simple.html#making-test-result-information-available-in-fixtures
# and:
#  https://github.com/pytest-dev/pytest/issues/288
#
# This fixture relies upon setup performed by hook `pytest_runtest_makereport`.
@pytest.fixture(autouse=True)
def act_upon_test_result(request):
    """Set a finalizer that will act upon the result of test success/failure."""
    test_dir = _get_test_dir_that_contains_module(request)
    def fin():
        # Because this is a function-scope fixture, `request.node` corresponds
        # to a single test function.
        if request.node.rep_setup.failed:
            print("Test setup failed: %s" % request.node.nodeid)
            pass
        elif request.node.rep_setup.passed:
            if request.node.rep_call.failed:
                print("\n\nTest failed: %s\nPreserving temporary files for review..." %
                        request.node.nodeid)

                pymod_test_mod_fname = _get_pymod_test_mod_fname(request)
                pymod_test_mod_fullpath = test_dir.join(pymod_test_mod_fname)
                if _localpath_exists(pymod_test_mod_fullpath):
                    failed_fname = _RENAME_PREFIX_FOR_FAILED_TEST + pymod_test_mod_fname
                    pymod_test_mod_failed_fullpath = test_dir.join(failed_fname)
                    print("Rename file: %s -> %s" %
                            (pymod_test_mod_fullpath, pymod_test_mod_failed_fullpath))
                    pymod_test_mod_fullpath.rename(pymod_test_mod_failed_fullpath)

                pmgen_dir_fullpath = test_dir.join(_PMGEN_DIRNAME_DEFAULT)
                if _localpath_exists(pmgen_dir_fullpath):
                    failed_fname = _RENAME_PREFIX_FOR_FAILED_TEST + _PMGEN_DIRNAME_DEFAULT
                    pmgen_dir_failed_fullpath = test_dir.join(failed_fname)
                    print("Rename directory: %s -> %s" %
                            (pmgen_dir_fullpath, pmgen_dir_failed_fullpath))
                    pmgen_dir_fullpath.rename(pmgen_dir_failed_fullpath)

    request.addfinalizer(fin)


@pytest.fixture
def pmgen_py_compile(python_exe_fullpath, pmgen_py_fullpath, request):
    """Return a closure that can be invoked to compile a Pymod Nim module."""
    def compile_py_mod(py_mod_name):
        nim_mod_fname = py_mod_name + _NIM_MOD_FNAME_SUFFIX
        subprocess.check_call([python_exe_fullpath, pmgen_py_fullpath, nim_mod_fname])
    return compile_py_mod


@pytest.fixture
def pymod_test_mod(request):
    """Import & return the Pymod test module that has been compiled."""
    return importlib.import_module(_get_pymod_test_mod_name(request))

