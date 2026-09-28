from setuptools import Extension, setup

setup(ext_modules=[Extension("call_trace", ["call_trace.c"], extra_compile_args=["-O2", "-Wall"])])
