def test_0_compile_and_import(python_exe_fullpath, pmgen_py_fullpath):
    import subprocess
    nim_mod_fname = __name__ + ".nim"
    subprocess.check_call([python_exe_fullpath, pmgen_py_fullpath, nim_mod_fname])


def test_Hello_World(pymod_test_mod):
    assert pymod_test_mod.returnHelloWorld() == "Hello World!"
