#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Fri Dec 08 14:22 BRST 2013

import bob
import numpy
import argparse
from .. import utils

from parallel_trainers.trainers.utils.file_loader import FileLoader

def main():

  DATABASES_RESOURCE_NAME = "databases"

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-i', '--input-dir', metavar='DIR', type=str, dest='input_dir', default='', help='Base directory that will be used to save the results.')
  parser.add_argument('-o', '--output-dir', metavar='DIR', type=str, dest='output_dir', default='', help='Base directory with the features.')

  parser.add_argument('-x', '--mixtures', type=int, dest='mixtures', default=16, help='Number of mixtures.')

  parser.add_argument('-t', '--convergence_threshold', type=float, dest='convergence_threshold', default=0.0001, help='Convergence threshold.')

  parser.add_argument('-d','--databases', nargs = '+', required = True,
        help = 'Database and the protocol; registered databases are: %s'%utils.resources.get_available_resources(DATABASES_RESOURCE_NAME))

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')


  args = parser.parse_args()

  input_dir  = args.input_dir
  output_dir = args.output_dir
  mixtures   = args.mixtures
  convergence_threshold = args.convergence_threshold
  verbose = args.verbose
  databases = args.databases
  
  #Loading the database resources
  for d in databases:
    db = utils.resources.load_resource(DATABASES_RESOURCE_NAME,d)
    file_loader = FileLoader(db)







if __name__ == "__main__":
  main()

