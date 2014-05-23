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
        "facereclib",
        "mpi4py"
    ],

    namespace_packages = [
      'parallel_trainers'
      ],

    entry_points={
      'console_scripts': [
        'ubm_trainer.py   = parallel_trainers.trainers.script.ubm_trainer:main',
        'isv_U_trainer.py = parallel_trainers.trainers.script.isv_U_trainer:main',
        'isv_U_trainer_dir.py = parallel_trainers.trainers.script.isv_U_trainer_dir:main'
      ],

      # registered database short cuts
      'databases': [
        'arface                  =  parallel_trainers.trainers.configurations.databases.arface:database',
        'atnt                    = parallel_trainers.trainers.configurations.databases.atnt:database',
        'banca                   = parallel_trainers.trainers.configurations.databases.banca:database',
        'caspeal                 = parallel_trainers.trainers.configurations.databases.caspeal:database',
        'frgc                    = parallel_trainers.trainers.configurations.databases.frgc:database',
        'gbu                     = parallel_trainers.trainers.configurations.databases.gbu:database',
        'lfw                     = parallel_trainers.trainers.configurations.databases.lfw_unrestricted:database',
        'mobio                   = parallel_trainers.trainers.configurations.databases.mobio:database',
        'multipie                = parallel_trainers.trainers.configurations.databases.multipie:database',
        'scface                  = parallel_trainers.trainers.configurations.databases.scface:database',
        'xm2vts                  = parallel_trainers.trainers.configurations.databases.xm2vts:database',


        'cpqd_smartphone_male    = parallel_trainers.trainers.configurations.databases.cpqd_smartphone_male:database',
        'cpqd_notebook_male      = parallel_trainers.trainers.configurations.databases.cpqd_notebook_male:database',
        'cpqd_n2s_male           = parallel_trainers.trainers.configurations.databases.cpqd_n2s_male:database',
        'cpqd_s2n_male           = parallel_trainers.trainers.configurations.databases.cpqd_s2n_male:database',

        'cpqd_notebook_female    = parallel_trainers.trainers.configurations.databases.cpqd_notebook_female:database',
        'cpqd_smartphone_female  = parallel_trainers.trainers.configurations.databases.cpqd_smartphone_female:database',
        'cpqd_n2s_female         = parallel_trainers.trainers.configurations.databases.cpqd_n2s_female:database',
        'cpqd_s2n_female         = parallel_trainers.trainers.configurations.databases.cpqd_s2n_female:database',
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
