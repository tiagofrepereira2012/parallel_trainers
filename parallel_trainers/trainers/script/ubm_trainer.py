#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Fri Dec 08 14:22 BRST 2013

import bob
import numpy
import argparse
from .. import utils
from mpi4py import MPI

from parallel_trainers.trainers.utils.file_loader import FileLoader


def train_kmeans(data, gaussians, dim):
  #Starting the kmeans (MUST BE SERIAL)
  kmeans = bob.machine.KMeansMachine(gaussians, dim)
  kmeansTrainer                       = bob.trainer.KMeansTrainer()
  kmeansTrainer.max_iterations        = 500
  kmeansTrainer.convergence_threshold = 0.0001
  kmeansTrainer.rng                   = bob.core.random.mt19937(5489)
  kmeansTrainer.train(kmeans, data)

  #GMM Trainer and machine
  [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(data)
  means = kmeans.means

  return means,variances,weights



def main():

  DATABASES_RESOURCE_NAME = "databases"

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-i', '--input-dir', metavar='DIR', type=str, dest='input_dir', default='', help='Base directory that will be used to save the results.')
  parser.add_argument('-o', '--output-dir', metavar='DIR', type=str, dest='output_dir', default='', help='Base directory with the features.')

  parser.add_argument('-g', '--gaussians', type=int, dest='gaussians', default=16, help='Number of Gaussians.')

  parser.add_argument('-t', '--convergence_threshold', type=float, dest='convergence_threshold', default=0.0001, help='Convergence threshold.')

  parser.add_argument('-d','--databases', nargs = '+', required = True,
        help = 'Database and the protocol; registered databases are: %s'%utils.resources.get_available_resources(DATABASES_RESOURCE_NAME))

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')


  args = parser.parse_args()

  input_dir  = args.input_dir
  output_dir = args.output_dir
  gaussians  = args.gaussians
  convergence_threshold = args.convergence_threshold
  verbose = args.verbose
  databases = args.databases
  
  #Loading the database resources
  v = None
  for d in databases:
    db = utils.resources.load_resource(DATABASES_RESOURCE_NAME,d)
    file_loader = FileLoader(db)
    print("Loading features .... ")
    v = file_loader()

  
  #SETTING UP MPI
  comm        = MPI.COMM_WORLD #Starting MPI
  rank        = comm.Get_rank() #RANK i.e. How am I?
  size        = comm.Get_size() #Number of parallel process for T-matrix calculation

  print("Kmeans ... ")
  means,variances,weights = train_kmeans(v,gaussians, v.shape[1])
  print(weights)




if __name__ == "__main__":
  main()

