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

  parser.add_argument('-o', '--output-file', metavar='DIR', type=str, dest='output_file', default='T.hdf5', help='Output file.')

  parser.add_argument('-r', '--iterations', type=int, dest='iterations', default=10, help='Number of iterations.')

  parser.add_argument('-v', '--verbose', action='count', dest='verbose', help='Increases this script verbosity')

  parser.add_argument('-u', '--ubm', type=str, dest='ubm', default='', help='Use the paramenters of this UBM for initialization instead of use kmeans')

  parser.add_argument('-s', '--t-subspace', type=int, dest='t_subspace', default=200, help='The TV Matrix dimension')

  parser.add_argument('-d','--dimensionality',dest='dim', type=int, required=True, default=40, help = 'Dimensionality of the feature vector')

  parser.add_argument('-c', '--compute-statistics', action='store_true', dest='compute_statistics', default=False, help='?')

  #create a subparser
  subparsers = parser.add_subparsers(help='Input type: Database querying or list file')

  #Database
  parser_database = subparsers.add_parser('database', help='Querying parameters using the database plugins')
  parser_database.add_argument('-d','--database_names', dest='databases', nargs = '+', required = True, help = 'Database and the protocol; registered databases are: %s'%utils.resources.get_available_resources(DATABASES_RESOURCE_NAME))

  #List files
  parser_files = subparsers.add_parser('list', help='Querying the parameters using a file list')
  parser_files.add_argument('-f','--file_name',dest='file_name', type=str, required=True, help = 'List of parameters (Separated by line)')

  args = parser.parse_args()

  output_file           = args.output_file
  verbose               = args.verbose
  ubm                   = args.ubm
  iterations            = args.iterations
  t_subspace            = args.t_subspace  
  dim                   = args.dim
  compute_statistics    = args.compute_statistics
  

  databases = None
  file_name = None
  base_ubm  = utils.load_gmm_from_file(ubm)
  gaussians = base_ubm.dim_c


  dict_args = vars(args)
  for k in dict_args.keys():
    if(k=="databases"):
      databases = args.databases
    elif(k=="file_name"):
      file_name = args.file_name


  #Setting the log
  logging.basicConfig(filename='mpi_ivectorT_trainer.log',level=verbose, format='%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')


  ###############
  #SETTING UP MPI
  ##############
  comm        = MPI.COMM_WORLD #Starting MPI
  rank        = comm.Get_rank() #RANK i.e. Who am I?
  size        = comm.Get_size() #Number of parallel process for T-matrix calculation


  file_loader = FileLoader(dim=dim)

  #Load a database or a file dir
  if(databases!=None):
    files = utils.load_list_from_resources(databases, DATABASES_RESOURCE_NAME, file_loader, arrange_by_client=False)
  else:
    files        = open(file_name).readlines()
    for i in range(len(files)):
      files[i] = files[i].rstrip("\n")

  subset_files = utils.split_files(files,rank,size)

  #Computing the statistics per client
  if(rank==0): 
    if(compute_statistics):
      logging.info("Computing statistics")
    else:
      logging.info("Loading statistics")

  gmm_stats = []
  for f in subset_files: #for each user file, acc statistics
    if(not compute_statistics):
      gmm_stats.append(bob.machine.GMMStats(bob.io.HDF5File(f)))
    else:
      features = file_loader.load_features_from_file(f)
      stats = bob.machine.GMMStats(gaussians,dim)
      base_ubm.acc_statistics(features,stats)
      gmm_stats.append(stats)


  #################
  # Training T
  ##############

  if(rank == 0):
    logging.info("Training IVector T")

  mpi_tv_trainer = MPITVTrainer(comm, base_ubm, T_dimension = t_subspace,  iterations=iterations)
  mpi_tv_trainer.train(gmm_stats)

  if(rank==0):
    logging.info("Saving TV Matrix ...")
    mpi_tv_trainer.save(output_file)
    logging.info("End!")




if __name__ == "__main__":
  main()

