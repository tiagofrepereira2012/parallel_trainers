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


def main():

  DATABASES_RESOURCE_NAME       = "databases"
  K_MEANS_MAX_ITERATIONS        = 50
  K_MEANS_CONVERGENCE_THRESHOLD = 0.0001


  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-i', '--input-dir', metavar='DIR', type=str, dest='input_dir', default='', help='Base directory for the features.')
  parser.add_argument('-o', '--output-file', metavar='DIR', type=str, dest='output_file', default='UBM.hdf5', help='Output file.')

  parser.add_argument('-t', '--convergence_threshold', type=float, dest='convergence_threshold', default=0.0001, help='Convergence threshold.')

  parser.add_argument('-r', '--iterations', type=int, dest='iterations', default=10, help='Number of iterations.')

  parser.add_argument('-d','--databases', nargs = '+', required = True,
        help = 'Database and the protocol; registered databases are: %s'%utils.resources.get_available_resources(DATABASES_RESOURCE_NAME))

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')

  parser.add_argument('-u', '--ubm', type=str, dest='ubm', default='', help='Use the paramenters of this UBM for initialization instead of use kmeans')


  args = parser.parse_args()

  input_dir             = args.input_dir
  output_file           = args.output_file
  convergence_threshold = args.convergence_threshold
  verbose               = args.verbose
  databases             = args.databases
  ubm                   = args.ubm
  iterations            = args.iterations
  

  ###############
  #SETTING UP MPI
  ##############
  comm        = MPI.COMM_WORLD #Starting MPI
  rank        = comm.Get_rank() #RANK i.e. How am I?
  size        = comm.Get_size() #Number of parallel process for T-matrix calculation

  
  #means        = None
  #variances    = None
  #weights      = None
  base_ubm     =  None
  partial_data = None
  fresh_means  = None

  
  whole_data = utils.load_features_from_resources(databases, DATABASES_RESOURCE_NAME, arrange_by_client = True)
  print(whole_data[0].shape)

  ####
  # Loading features
  ####
  #if(rank==0):
    #print("Loading features...")
    #whole_data = utils.load_features_from_resources(databases, DATABASES_RESOURCE_NAME, arrange_by_client=True)

    ##sending the proper data for each node
    #print("Transmitting proper data to each node...")
    #for i in range(1,size):
      #partial_data = utils.select_data(whole_data, size, i)
      #comm.send(partial_data,dest=i,tag=11)
    #partial_data = utils.select_data(whole_data, size, 0)#data for the rank 0

  #else: #if rank!=0
    ##receiving the data for each node
    #partial_data = comm.recv(source=0,tag=11)

  #dim = partial_data.shape[1] #Fetching the feature dimensionality


if __name__ == "__main__":
  main()

