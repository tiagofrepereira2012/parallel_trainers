#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Fri Dec 08 14:22 BRST 2013
mtklhmlkftmhlkfgm

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

  parser.add_argument('-v', '--verbose', action='count', dest='verbose', help='Increases this script verbosity')

  parser.add_argument('-u', '--ubm', type=str, dest='ubm', default='', help='Use the paramenters of this UBM for initialization instead of use kmeans')

  parser.add_argument('-s', '--u-subspace', type=int, dest='u_subspace', default=160, help='The subspace U for within-class variations')

  parser.add_argument('-g', '--gaussians', type=int, dest='gaussians', default=512, help='Number of Gaussians.')

  parser.add_argument('-e', '--relevance-factor', type=float, dest='relevance_factor', default=4, help='Relevance factor for the user offset')

  parser.add_argument('-f', '--facereclib', action='store_true', dest='facereclib', default=False, help='Is the output compatible with with the IDIAP Facereclib?')

  parser.add_argument('-d','--dimensionality',dest='dim', type=int, required=True, default=40, help = 'Dimensionality of the feature vector')

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
  u_subspace            = args.u_subspace
  relevance_factor      = args.relevance_factor
  facereclib            = args.facereclib
  gaussians             = args.gaussians
  dim                   = args.dim 
  

  databases = None
  file_name = None
  base_ubm  = utils.load_gmm_from_file(ubm, gaussians, dim)


  dict_args = vars(args)
  for k in dict_args.keys():
    if(k=="databases"):
      databases = args.databases
    elif(k=="file_name"):
      file_name = args.file_name


  #Setting the log
  logging.basicConfig(filename='mpi_isvU_trainer.log',level=verbose, format='%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')


  ###############
  #SETTING UP MPI
  ##############
  comm        = MPI.COMM_WORLD #Starting MPI
  rank        = comm.Get_rank() #RANK i.e. Who am I?
  size        = comm.Get_size() #Number of parallel process for T-matrix calculation


  file_loader = FileLoader(dim=dim)

  #Load a database or a file dir
  if(databases!=None):
    files = utils.load_list_from_resources(databases, DATABASES_RESOURCE_NAME, file_loader, arrange_by_client=True)
  else:
    files = os.listdir(file_name)
    for i in range(len(files)):
      files[i] = os.path.join(file_name, files[i])

  subset_files = utils.split_files(files,rank,size)

  #Computing the statistics per client
  gmm_stats = []
  if(rank==0):
    logging.info("Computing statistics")

  client_stats = []
  for client_files in subset_files: #for each user, accumulate a list of statistics of this user
    
    stats = []
    #If was not a database resource we must execute a listdir
    if(databases == None):
      listdir = os.listdir(client_files)
      for i in range(len(listdir)):
        listdir[i] = os.path.join(client_files,listdir[i])
      client_files = listdir

    for f in client_files: #for each user file, acc statistics
      features = file_loader.load_features_from_file(f)      
      stats_client = bob.machine.GMMStats(gaussians,dim)
      base_ubm.acc_statistics(features,stats_client)
      stats.append(stats_client)

    client_stats.append(stats)


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

