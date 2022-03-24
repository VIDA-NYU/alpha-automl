import os
import setuptools

package_name = 'alphad3m'


def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf8') as file:
        return file.read()


def read_version():
    module_path = os.path.join(package_name, '__init__.py')
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
    entry_points={
        'console_scripts': [
            'alphad3m_serve = alphad3m.main:main_serve',
            'alphad3m_search = alphad3m.main:main_search'
          ]},
    install_requires=req,
    description="AlphaD3M: NYU's AutoML System",
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    author='Remi Rampin, Roque Lopez, Raoni Lourenco',
    author_email='remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu',
    maintainer='Remi Rampin, Roque Lopez, Raoni Lourenco',
    maintainer_email='remi.rampin@nyu.edu, rlopez@nyu.edu, raoni@nyu.edu',
    keywords=['datadrivendiscovery', 'automl', 'd3m', 'ta2', 'nyu'],
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
    ])
