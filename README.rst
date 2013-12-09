===============================================================================
Parallel trainers
===============================================================================
This package trains a parallel version of the following models, using the python binds for MPI (mpi4py - https://pypi.python.org/pypi/mpi4py/), and the Trainers and Machines from Bob:
 - Universal Background Model (UBM)
 - The within client variation matrix (U Matrix) for the Intersession Variability Modeling (ISV)


With these codes you can split all the work in a grid system that has MPI available.


If you use this package, please cite the following publications:

1. The original paper for the UBM system::

    @article{reynolds2000speaker,
      title={Speaker verification using adapted Gaussian mixture models},
      author={Reynolds, Douglas A and Quatieri, Thomas F and Dunn, Robert B},
      journal={Digital signal processing},
      volume={10},
      number={1},
      pages={19--41},
      year={2000},
      publisher={Elsevier}
    }

2. Paper describing the use of Session Variability Modelling for face authentication::

    @article{mccool2013session,
      title={Session variability modelling for face authentication},
      author={McCool, Christopher and Wallace, Roy and McLaren, Mitchell and El Shafey, Laurent and Marcel, S{\'e}bastien},
      journal={IET biometrics},
      volume={2},
      number={3},
      pages={117--129},
      year={2013},
      publisher={IET}
    }


3. Bob as the core framework used to train the models::

    @inproceedings{Anjos_ACMMM_2012,
      author = {A. Anjos AND L. El Shafey AND R. Wallace AND M. G\"unther AND C. McCool AND S. Marcel},
      title = {Bob: a free signal processing and machine learning toolbox for researchers},
      year = {2012},
      month = oct,
      booktitle = {20th ACM Conference on Multimedia Systems (ACMMM), Nara, Japan},
      publisher = {ACM Press},
    }


Installation
------------

Using ``zc.buildout``
=====================

Download the latest version of this package from `Github
<https://github.com/tiagofrepereira2012/parallel_trainers>`_ and unpack it in your
working area. The installation of the toolkit itself uses `buildout
<http://www.buildout.org/>`_. You don't need to understand its inner workings
to use this package. Here is a recipe to get you started::
  
  $ python bootstrap.py 
  $ ./bin/buildout

These 2 commands should download and install all non-installed dependencies and
get you a fully operational test and development environment.

.. note::

  The python shell used in the first line of the previous command set
  determines the python interpreter that will be used for all scripts developed
  inside this package. Because this package makes use of `Bob
  <http://idiap.github.com/bob>`_, you must make sure that the ``bootstrap.py``
  script is called with the **same** interpreter used to build Bob, or
  unexpected problems might occur.

  If Bob is installed by the administrator of your system, it is safe to
  consider it uses the default python interpreter. In this case, the above 3
  command lines should work as expected. If you have Bob installed somewhere
  else on a private directory, edit the file ``buildout.cfg`` **before**
  running ``./bin/buildout``. Find the section named ``external`` and edit the
  line ``egg-directories`` to point to the ``lib`` directory of the Bob
  installation you want to use. For example::

    [external]
    recipe = xbob.buildout:external
    egg-directories=/Users/crazyfox/work/bob/build/lib

User Guide
----------

Universal Background Training
==============================

Type the following command to see all the available options for the UBM trainer::

   $ ./bin/ubm_trainer.py --help

In order to run this script in the MPI environment run the following code::

   $ mpiexec --np <number_of_nodes> --hosts=<available_hosts (comma separated)> ./bin/ubm_trainer.py <options>

It is possible to use an xbob.db package as input or a file containing the list of features to train. To use a database, run the following code::

   $ mpiexec --np <number_of_nodes> --hosts=<available_hosts (comma separated)> ./bin/ubm_trainer.py <options> database -d <database_name>

To use a regular file list, run the following code::
   $ mpiexec --np <number_of_nodes> --hosts=<available_hosts (comma separated)> ./bin/ubm_trainer.py <options> list -f <file_name>





Within client variation matrix (U Matrix) for the ISV
======================================================

Type the following command to see all the available options for the UBM trainer::

   $ ./bin/isv_U_trainer.py --help

In order to run this script in the MPI environment run the following code::

   $ mpiexec --np <number_of_nodes> --hosts=<available_hosts (comma separated)> ./bin/isv_U_trainer.py <options>


How to configure the MPI in my grid system?
============================================

You can see all the details of how to configure the MPI and how to setup the python bindings in the following page: `http://mpi4py.scipy.org/ <http://mpi4py.scipy.org/>`_.



Problems
--------

In case of problems, please contact any of the authors of the package.



