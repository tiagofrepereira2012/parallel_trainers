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
import os



def main():

  DATABASES_RESOURCE_NAME       = "databases"

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-o', '--output-file', metavar='DIR', type=str, dest='output_file', default='ISV.hdf5', help='Output file.')

  parser.add_argument('-r', '--iterations', type=int, dest='iterations', default=10, help='Number of iterations.')

  parser.add_argument('-d','--database', dest='databases', type=str, required = True, help = 'Database directory')

  parser.add_argument('-v', '--verbose', action='count', dest='verbose', help='Increases this script verbosity')

  parser.add_argument('-u', '--ubm', type=str, dest='ubm', default='', help='Use the paramenters of this UBM for initialization instead of use kmeans')

  parser.add_argument('-s', '--u-subspace', type=int, dest='u_subspace', default=160, help='The subspace U for within-class variations')

  parser.add_argument('-g', '--gaussians', type=int, dest='gaussians', default=512, help='Number of Gaussians.')

  parser.add_argument('-e', '--relevance-factor', type=float, dest='relevance_factor', default=4, help='Relevance factor for the user offset')

  parser.add_argument('-f', '--facereclib', action='store_true', dest='facereclib', default=False, help='Is the output compatible with with the IDIAP Facereclib?')



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
  dim                   = 44
  

  #Setting the log
  logging.basicConfig(filename='mpi_isvU_trainer.log',level=verbose, format='%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')


  ###############
  #SETTING UP MPI
  ##############
  comm        = MPI.COMM_WORLD #Starting MPI
  rank        = comm.Get_rank() #RANK i.e. Who am I?
  size        = comm.Get_size() #Number of parallel process for T-matrix calculation


  users = os.listdir(databases)
  for i in range(len(users)):
    users[i] = os.path.join(databases,users[i])

  users = utils.split_files(users,rank,size)


  #dim          = partial_data[0].shape[2]
  base_ubm     = utils.load_gmm_from_file(ubm, gaussians, dim)


  #Computing the statistics per client
  gmm_stats = []
  if(rank==0):
    logging.info("Computing statistics")

  client_stats = []
  i = 0
  for u in users:
    i = i + 1

    #features = utils.load_features_from_dir(u, dim)
    user_file_names = os.listdir(u)
    feature_stats = []
    for f in user_file_names:
      if(f[len(f)-4:len(f)] != "hdf5"):  
        continue

      #print("{0} - {1}".format(u,f))

      user_file_name = os.path.join(u,f)
      #features = utils.load_features_from_file(user_file_name,dim)
      #print(user_file_name)
      features  = bob.io.load(user_file_name)

      stats_client = bob.machine.GMMStats(gaussians,dim)
      base_ubm.acc_statistics(features,stats_client)
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

