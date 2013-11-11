#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Fri Dec 14 14:22 BRST 2012

import bob
import numpy
import argparse


def main():

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-i', '--input-dir', metavar='DIR', type=str, dest='input_dir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')
  parser.add_argument('-o', '--output-dir', metavar='DIR', type=str, dest='output_dir', default=OUTPUT_DIR, help='Base directory with the features.')

  parser.add_argument('-x', '--mixtures', type=int, dest='mixtures', default=16, help='Number of mixtures.')

  parser.add_argument('-t', '--convergence_threshold', type=float, dest='convergence_threshold', default=0.0001, help='Convergence threshold.')

  parser.add_argument('-d','--database', nargs = '+', required = True,
        help = 'Database and the protocol; registered databases are: %s'%utils.resources.resource_keys('database'))

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')


  args = parser.parse_args()


  input_dir = args.input_dir


if __name__ == "__main__":
  main()

