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
from ..mpi_trainers import *


def main():

  DATABASES_RESOURCE_NAME       = "databases"
  K_MEANS_MAX_ITERATIONS        = 50
  K_MEANS_CONVERGENCE_THRESHOLD = 0.0001


  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-o', '--output-file', metavar='DIR', type=str, dest='output_file', default='UBM.hdf5', help='Output file.')

  parser.add_argument('-g', '--gaussians', type=int, dest='gaussians', default=16, help='Number of Gaussians.')

  parser.add_argument('-t', '--convergence_threshold', type=float, dest='convergence_threshold', default=0.0001, help='Convergence threshold.')

  parser.add_argument('-r', '--iterations', type=int, dest='iterations', default=10, help='Number of iterations.')

  parser.add_argument('-d','--databases', nargs = '+', required = True,
        help = 'Database and the protocol; registered databases are: %s'%utils.resources.get_available_resources(DATABASES_RESOURCE_NAME))

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')

  parser.add_argument('-u', '--ubm-initialization-file', type=str, dest='ubm_initialization', default='', help='Use the paramenters of this UBM for initialization instead of use kmeans')


  args = parser.parse_args()

  output_file           = args.output_file
  gaussians             = args.gaussians
  convergence_threshold = args.convergence_threshold
  verbose               = args.verbose
  databases             = args.databases
  ubm_initialization    = args.ubm_initialization
  iterations            = args.iterations
  

  ###############
  #SETTING UP MPI
  ##############
  comm        = MPI.COMM_WORLD #Starting MPI
  rank        = comm.Get_rank() #RANK i.e. Who am I?
  size        = comm.Get_size() #Number of parallel process for T-matrix calculation

  
  base_ubm     =  None
  partial_data = None

  ####
  # Loading features
  ####
  if(rank==0):
    print("Loading features...")
    whole_data = utils.load_features_from_resources(databases, DATABASES_RESOURCE_NAME)
    #whole_data = whole_data[0:50000,:]
    #whole_data = numpy.random.rand(50000, 44)

    #sending the proper data for each node
    print("Transmitting proper data to each node...")
    for i in range(1,size):
      partial_data = utils.select_data(whole_data, size, i)
      comm.send(partial_data,dest=i,tag=11)
    partial_data = utils.select_data(whole_data, size, 0)#data for the rank 0

  else: #if rank!=0
    #receiving the data for each node
    partial_data = comm.recv(source=0,tag=11)
    

  dim = partial_data.shape[1] #Fetching the feature dimensionality
  ####
  # K-Means
  ###
  #Setting UP the initialization
  if(ubm_initialization != ""):
    #print("Load UBM from: " + ubm_initialization)
    base_ubm = utils.load_gmm_from_file(ubm_initialization, gaussians, dim)
  else:


    mpi_kmeans_trainer = MPIKmeansTrainer(comm, gaussians, dim)
    mpi_kmeans_trainer.train(partial_data)

    if(rank == 0):
      print("Waiting UBM trainer")
      kmeans_machine = mpi_kmeans_trainer.get_machine()

      base_ubm       = bob.machine.GMMMachine(gaussians, dim)
      base_ubm.means       = kmeans_machine.means
      [variances, weights] = kmeans_machine.get_variances_and_weights_for_each_cluster(whole_data)
      base_ubm.variances   = variances
      base_ubm.weights     = weights
      utils.save_gmm(base_ubm, output_file)

    comm.Barrier() #Synchronization barrier in order to all process load the base_ubm
    base_ubm = utils.load_gmm_from_file(output_file, gaussians, dim) #loading the kmeans UBM for each node 


  if(rank == 0):
    print("Training UBM")

  mpi_ubm_trainer = MPIUBMTrainer(comm, base_ubm, iterations=iterations, convergence_threshold=convergence_threshold)
  mpi_ubm_trainer.train(partial_data)

  if(rank==0):
    print("Saving GMM ...")
    machine = mpi_ubm_trainer.get_machine()
    utils.save_gmm(machine, output_file)
    print("End!")



if __name__ == "__main__":
  main()

