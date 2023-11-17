import os
import re
import setuptools
from collections import defaultdict

package_name = 'alpha-automl'
package_dir = 'alpha_automl'


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


def get_requires():
    with open('requirements.txt') as fp:
        dependencies = [line for line in fp if line and not line.startswith('#')]

        return dependencies


def get_extra_requires():
    with open('extra_requirements.txt') as fp:
        extra_dependencies = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith('#'):
                tags = set()
                if ':' in k:
                    k, v = k.split(':')
                    tags.update(vv.strip() for vv in v.split(','))
                tags.add(re.split('[<=>]', k)[0])
                for t in tags:
                    extra_dependencies[t].add(k)

        # add tag `full` at the end
        extra_dependencies['full'] = set(vv for v in extra_dependencies.values() for vv in v)

    return extra_dependencies


long_description = read_readme()
version = read_version()
requires = get_requires()
extra_requires = get_extra_requires()

setuptools.setup(
    name=package_name,
    version=version,
    packages=setuptools.find_packages(),
    install_requires=requires,
    extras_require=extra_requires,
    description="Alpha-AutoML: NYU's AutoML System",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VIDA-NYU/alpha-automl',
    include_package_data=True,
    author='Roque Lopez, Eden Wu, Aécio Santos, Remi Rampin',
    author_email='rlopez@nyu.edu, eden.wu@nyu.edu, aecio.santos@nyu.edu, remi.rampin@nyu.edu',
    maintainer='Roque Lopez, Eden Wu, Aécio Santos, Remi Rampin',
    maintainer_email='rlopez@nyu.edu, eden.wu@nyu.edu, aecio.santos@nyu.edu, remi.rampin@nyu.edu',
    keywords=['datadrivendiscovery', 'automl', 'nyu'],
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
    ])
