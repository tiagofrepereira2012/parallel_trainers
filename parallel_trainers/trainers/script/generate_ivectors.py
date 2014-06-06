#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Wed Jun 04 13:39 BRT 2014

import bob
import numpy
import argparse
from .. import utils
from mpi4py import MPI
from parallel_trainers.trainers.utils.file_loader import FileLoader
import os

import logging

def main():

  DATABASES_RESOURCE_NAME       = "databases"


  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('input_dir', metavar='DIR', type=str, default='', help='Input directory with the GMM Stats')

  parser.add_argument('-o', '--output-dir', metavar='DIR', type=str, dest='output_dir', default='./ivectors/', help='Output file.')

  parser.add_argument('-u', '--ubm', type=str, dest='ubm', default='', help='Use the paramenters of this UBM for initialization instead of use kmeans')

  parser.add_argument('-t', '--tv-matrix-file', type=str, dest='tv_file', default='', help='TV Matrix File') 

  parser.add_argument('-v', '--verbose', action='count', dest='verbose', help='Increases this script verbosity')

  #create a subparser
  subparsers = parser.add_subparsers(help='Input type: Database querying or list file with the GMM Statistics')

  #Database
  parser_database = subparsers.add_parser('database', help='Querying the GMM Statistics using the database plugins')
  parser_database.add_argument('-d','--database_names', dest='databases', nargs = '+', required = True, help = 'Database and the protocol; registered databases are: %s'%utils.resources.get_available_resources(DATABASES_RESOURCE_NAME))

  #List filesi
  parser_files = subparsers.add_parser('list', help='Querying the GMM Statistics using a file list')
  parser_files.add_argument('-f','--file_name',dest='file_name', type=str, required=True, help = 'List with relative paths (Separated by line)')


  args = parser.parse_args()

  input_dir             = args.input_dir
  output_dir            = args.output_dir
  verbose               = args.verbose
  ubm_machine           = bob.machine.GMMMachine(bob.io.HDF5File(args.ubm))
  ivector_machine       = bob.machine.IVectorMachine(bob.io.HDF5File(args.tv_file))
  ivector_machine.ubm   = ubm_machine

  databases = None
  file_name = None

  dict_args = vars(args)
  for k in dict_args.keys():
    if(k=="databases"):
      databases = args.databases
    elif(k=="file_name"):
      file_name = args.file_name
    

  ###############
  #SETTING UP MPI
  ##############
  comm        = MPI.COMM_WORLD #Starting MPI
  rank        = comm.Get_rank() #RANK i.e. Who am I?
  size        = comm.Get_size() #Number of parallel process for T-matrix calculation

  #Setting the log
  logging.basicConfig(filename='mpi_ivectors.log',level=verbose, format='%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')

  if(rank==0):
    logging.info("Computing and iVectors")

  #################
  #Load a database or a file dir
  #################
  file_loader = FileLoader(dim=0)
  if(databases!=None):
    files = utils.load_list_from_resources(databases, DATABASES_RESOURCE_NAME, file_loader, arrange_by_client=False)
  else:
    files        = open(file_name).readlines()
    for i in range(len(files)):
      files[i] = files[i].rstrip("\n")

  subset_files = utils.split_files(files,rank,size)
  
  for f in subset_files:
    i_file = os.path.join(input_dir,f)
    o_file = os.path.join(output_dir,f)
    utils.ensure_dir(os.path.dirname(o_file))

    gmm_stats = bob.machine.GMMStats(bob.io.HDF5File(i_file))
    ivector   = ivector_machine.forward(gmm_stats)
    hdf5file = bob.io.HDF5File(o_file, 'w')
    hdf5file.set('ivec', ivector)
    del hdf5file

  if(rank==0):
    logging.info("End!!!")
  

if __name__ == "__main__":
  main()

