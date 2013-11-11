#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Sun Jul  8 20:35:55 CEST 2012

from setuptools import setup, find_packages

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='parallel_trainers',
    version='0.0.0a1',
    description='Paralll',
    url='http://pypi.python.org/pypi/',
    license='GPLv3',
    author='Tiago de Freitas Pereira',
    author_email='tiagofrepereira@gmail.com',
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data = True,

    install_requires=[
        "setuptools",
        "bob >= 1.1.0",      # base signal proc./machine learning library
    ],

    namespace_packages = [
      'parallel_trainers'
      ],

    entry_points={
      'console_scripts': [
        'ubm_trainer.py = parallel_trainers.trainers.script.ubm_trainer:main'
      ],
    },

    classifiers = [
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],


)
