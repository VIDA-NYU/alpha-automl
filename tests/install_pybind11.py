import os
import setuptools
import sys
import subprocess

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        try:
            import pybind11
        except ImportError:
            if subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11']):
                raise RuntimeError('pybind11 install failed.')

        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)
    

if __name__ == '__main__':

        ext_modules=[
            setuptools.Extension(
                'your_extension_name',
                sources=['your_extension_source.cpp'],
                include_dirs=[get_pybind_include()],
                language='c++'
            )
        ]
