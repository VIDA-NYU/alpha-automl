import os
import setuptools
import sys
import subprocess

package_name = 'alpha-automl'
package_dir = 'alpha_automl'


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


def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf8') as file:
        return file.read()


def read_version():
    module_path = os.path.join(package_dir, '__init__.py')
    with open(module_path) as file:
        for line in file:
            parts = line.strip().split(' ')
            if parts and parts[0] == '__version__':
                return parts[-1].strip("'")

    raise KeyError('Version not found in {0}'.format(module_path))


long_description = read_readme()
version = read_version()

with open('requirements.txt') as fp:
    req = [line for line in fp if line and not line.startswith('#')]

setuptools.setup(
    name=package_name,
    version=version,
    packages=setuptools.find_packages(),
    install_requires=req,
    description="Alpha-AutoML: NYU's AutoML System",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VIDA-NYU/alpha-automl',
    include_package_data=True,
    author='Roque Lopez, Remi Rampin',
    author_email='rlopez@nyu.edu, remi.rampin@nyu.edu',
    maintainer='Roque Lopez, Remi Rampin',
    maintainer_email='rlopez@nyu.edu, remi.rampin@nyu.edu',
    keywords=['datadrivendiscovery', 'automl', 'nyu'],
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
    ],
    ext_modules=[
        setuptools.Extension(
            'your_extension_name',
            sources=['your_extension_source.cpp'],
            include_dirs=[get_pybind_include()],
            language='c++'
        )
    ]
)
