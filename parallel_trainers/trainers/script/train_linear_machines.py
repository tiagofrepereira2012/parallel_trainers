#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Wed Jun 04 13:39 BRT 2014

import bob
import numpy
import argparse
from .. import utils
#from mpi4py import MPI
from parallel_trainers.trainers.utils.file_loader import FileLoader
import os

import logging

def precompute_ivector_list(files,dim):
  ivectors = []
  for f in files:
    ivectors.append(numpy.zeros(shape=(len(f),dim)))
  return ivectors


def flat_list_numpy(list_numpy):
  
  elements = 0
  dim = 0
  #precomputing
  for l in list_numpy:  
    elements += l.shape[0]
    dim = l.shape[1]

  #allocating
  flat_numpy = numpy.zeros(shape=(elements, dim))
  offset = 0
  for l in list_numpy:
    flat_numpy[offset:offset+l.shape[0],:] = l
    offset += l.shape[0]

  return flat_numpy


def main():

  DATABASES_RESOURCE_NAME       = "databases"


  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-o', '--output-dir', metavar='DIR', type=str, dest='output_dir', default='./linear_trainers/', help='Output Directory')

  parser.add_argument('-i', '--whitening', type=str, dest='whitening_file', default='', help='Whitening Linear machine file. If \'\', will be trained!!!')

  parser.add_argument('-l', '--lda', type=str, dest='lda_file', default='', help='LDA Linear machine file. If \'\', will be trained!!!')

  parser.add_argument('-d', '--dim_lda', type=int, dest='dim_lda', default='200', help='Dimension LDA.')

  parser.add_argument('-w', '--wccn', type=str, dest='wccn_file', default='', help='WCCN Linear machine file. If \'\', will be trained!!!')

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

  #input_dir             = args.input_dir
  output_dir           = args.output_dir
  whitening_file       = args.whitening_file
  lda_file             = args.lda_file
  wccn_file            = args.wccn_file
  dim_lda              = args.dim_lda
  verbose              = args.verbose

  databases = None
  file_name = None

  dict_args = vars(args)
  for k in dict_args.keys():
    if(k=="databases"):
      databases = args.databases
    elif(k=="file_name"):
      file_name = args.file_name
    

  #Setting the log
  logging.basicConfig(filename='mpi_linear_machines.log',level=verbose, format='%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')

  #################
  #Load a database or a file dir
  #################
  file_loader = FileLoader(dim=0)
  if(databases!=None):
    files = utils.load_list_from_resources(databases, DATABASES_RESOURCE_NAME, file_loader, arrange_by_client=True)
  else:
    files        = open(file_name).readlines()
    for i in range(len(files)):
      files[i] = files[i].rstrip("\n")

  utils.ensure_dir(output_dir)
  dim = bob.io.load(files[0][0]).shape[0]

  logging.info("Loading iVectors")      
  ivectors = precompute_ivector_list(files,dim)

  for i in range(len(files)):
    for j in range(len(files[i])):
      f = files[i][j]
      ivectors[i][j,:] = bob.io.load(f)


  
  ########################
  #Training Whitening
  ##########################
  if(whitening_file != ""):
    logging.info("Loading Whitening")
    whitening_machine = bob.machine.LinearMachine(bob.io.HDF5File(whitening_file))
  else:
    logging.info("Training Whitening")

    flat_numpy = flat_list_numpy(ivectors)
    whitening_trainer = bob.trainer.WhiteningTrainer()
    whitening_machine = bob.machine.LinearMachine(dim, dim)
    whitening_trainer.train(whitening_machine, flat_numpy)
    hdf5file = bob.io.HDF5File(os.path.join(output_dir, "Whitening.hdf5"),"w")
    whitening_machine.save(hdf5file)
    del hdf5file
   
  logging.info("Whitening iVectors")
  for i in range(len(files)):
    for j in range(len(files[i])):
      f = files[i][j]
      ivectors[i][j,:] = whitening_machine(ivectors[i][j,:])


  ####################
  #Training LDA
  #####################
  if(lda_file != ""):
    logging.info("Loading LDA")
    lda_machine = bob.machine.LinearMachine(bob.io.HDF5File(lda_file))
    dim_lda = lda_machine.shape[1]
  else:
    logging.info("LDA Training")
    lda_trainer = bob.trainer.FisherLDATrainer(strip_to_rank=False)
    lda_machine = bob.machine.LinearMachine(dim, dim)
    lda_trainer.train(lda_machine, ivectors)
    lda_machine.resize(dim, dim_lda)

    hdf5file = bob.io.HDF5File(os.path.join(output_dir, "LDA.hdf5"),"w")
    lda_machine.save(hdf5file)
    del hdf5file


  logging.info("LDA projected iVectors")
  reduced_ivectors = precompute_ivector_list(files,dim_lda)
  for i in range(len(files)):
    for j in range(len(files[i])):
      f = files[i][j]
      reduced_ivectors[i][j,:] = lda_machine(ivectors[i][j,:])


  ####################
  #Training WCCN
  #####################
  if(wccn_file != ""):
    logging.info("Loading WCCN")
    wccn_machine = bob.machine.LinearMachine(bob.io.HDF5File(wccn_file))
  else:
    logging.info("WCCN Training")
    wccn_trainer = bob.trainer.WCCNTrainer()
    wccn_machine = bob.machine.LinearMachine(dim_lda, dim_lda)
    wccn_trainer.train(wccn_machine, reduced_ivectors)

    hdf5file = bob.io.HDF5File(os.path.join(output_dir, "WCCN.hdf5"),"w")
    wccn_machine.save(hdf5file)
    del hdf5file




  logging.info("End!!!")
  

if __name__ == "__main__":
  main()

