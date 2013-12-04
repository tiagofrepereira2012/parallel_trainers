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

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-o', '--output-file', metavar='DIR', type=str, dest='output_file', default='ISV.hdf5', help='Output file.')

  parser.add_argument('-r', '--iterations', type=int, dest='iterations', default=10, help='Number of iterations.')

  parser.add_argument('-d','--databases', nargs = '+', required = True,
        help = 'Database and the protocol; registered databases are: %s'%utils.resources.get_available_resources(DATABASES_RESOURCE_NAME))

  parser.add_argument('-v', '--verbose', action='count', dest='verbose', help='Increases this script verbosity')

  parser.add_argument('-u', '--ubm', type=str, dest='ubm', default='', help='Use the paramenters of this UBM for initialization instead of use kmeans')

  parser.add_argument('-s', '--u-subspace', type=int, dest='u_subspace', default=160, help='The subspace U for within-class variations')

  parser.add_argument('-g', '--gaussians', type=int, dest='gaussians', default=512, help='Number of Gaussians.')

  parser.add_argument('-e', '--relevance-factor', type=float, dest='relevance_factor', default=4, help='Relevance factor for the user offset')

  parser.add_argument('-f', '--facereclib', action='store_true', dest='facereclib', default=False, help='Compatible with facereclib?')

  args = parser.parse_args()

  output_file           = args.output_file
  verbose               = args.verbose
  databases             = args.databases
  ubm                   = args.ubm
  iterations            = args.iterations
  u_subspace            = args.u_subspace
  relevance_factor      = args.relevance_factor
  facereclib            = args.facereclib
  gaussians             = args.gaussians
  

  #Setting the log
  logging.basicConfig(filename='mpi_isvU_trainer.log',level=verbose, format='%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')


  ###############
  #SETTING UP MPI
  ##############
  comm        = MPI.COMM_WORLD #Starting MPI
  rank        = comm.Get_rank() #RANK i.e. Who am I?
  size        = comm.Get_size() #Number of parallel process for T-matrix calculation

  if(rank==0):
    logging.info("Loading features...")
    whole_data   = numpy.array(utils.load_features_from_resources(databases, DATABASES_RESOURCE_NAME, arrange_by_client = True))

    #sending the proper data for each node
    logging.info("Transmitting proper data to each node...")
    for i in range(1,size):
      partial_data = utils.select_data(whole_data, size, i)
      comm.send(partial_data,dest=i,tag=11)

    partial_data = utils.select_data(whole_data, size, 0)#data for the rank 0

  else: #if rank!=0
    #receiving the data for each node
    partial_data = comm.recv(source=0,tag=11)


  dim          = partial_data[0].shape[2]
  base_ubm     = utils.load_gmm_from_file(ubm, gaussians, dim)


  #Computing the statistics per client
  gmm_stats = []
  if(rank==0):
    logging.info("Computing statistics")

  client_stats = []
  i = 0
  for client_data in partial_data:
    i = i + 1

    feature_stats = []   
    for feature in client_data:
      stats_client = bob.machine.GMMStats(gaussians,dim)
      base_ubm.acc_statistics(feature,stats_client)
      feature_stats.append(stats_client)

    client_stats.append(feature_stats)


  #################
  # Training U
  ##############

  if(rank == 0):
    logging.info("Training ISV U")

  mpi_isvU_trainer = MPIISVUTrainer(comm, base_ubm, U_dimension = u_subspace, relevance_factor = relevance_factor, iterations=iterations)
  mpi_isvU_trainer.train(client_stats)

  if(rank==0):
    logging.info("Saving isv_base ...")
    machine = mpi_isvU_trainer.get_machine()
    if(facereclib):
      utils.save_isvbase(machine, output_file, ubm=base_ubm)
    else:
      utils.save_isvbase(machine, output_file)

    logging.info("End!")




if __name__ == "__main__":
  main()

