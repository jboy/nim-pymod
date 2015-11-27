def test_0_compile_pymod_test_mod(pmgen_py_compile):
    pmgen_py_compile(__name__)


def test_Hello_World(pymod_test_mod):
    assert pymod_test_mod.returnHelloWorld() == "Hello World!"
