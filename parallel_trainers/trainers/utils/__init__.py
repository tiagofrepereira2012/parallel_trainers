#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>

import resources
import file_loader
import numpy
import bob
from file_loader import FileLoader
import os


def split_files(files,rank,number_process):
  """This is the call function that you have to overwrite in the derived class.

    This method will split a list into different process
  """
  file_indexes = numpy.array(range(len(files)))
  files = numpy.array(files)

  #Selecting the indexes for each rank
  mod = file_indexes % number_process
  selected_indexes = list(numpy.where(mod==rank)[0])

  files = list(files[selected_indexes])
  if(type(files[0]) == str):
    for i in range(len(files)):
      files[i] = files[i].rstrip("\n")

  return files



def load_gmm_from_file(file_name, gaussians, dim):
  """
  Load a GMM from file
  """

  hdf5file  = bob.io.HDF5File(file_name,openmode_string='r')
  gmm = bob.machine.GMMMachine(gaussians,dim)
  bob.machine.GMMMachine.load(gmm,hdf5file)

  return gmm

def save_gmm(gmm, file_name):
  """
  Save a GMM in a file
  """

  hdf5file = bob.io.HDF5File(file_name,openmode_string='w')
  bob.machine.GMMMachine.save(gmm,hdf5file)


def save_isvbase(isv_base, file_name, ubm=None):
  """
  Save a isv base in a file
  """
  hdf5file = bob.io.HDF5File(file_name,openmode_string='w')

  if(ubm==None):
    isv_base.save(hdf5file)
  else:
    hdf5file.create_group('Projector')
    hdf5file.cd('Projector')
    ubm.save(hdf5file)

    hdf5file.cd('/')
    hdf5file.create_group('Enroller')
    hdf5file.cd('Enroller')
    isv_base.save(hdf5file)



def select_data(data, number_process, rank):
  """
  Split the data according a rank ID and the number of process
  """
  file_indexes = numpy.array(range(data.shape[0]))

  #Selecting the indexes for each rank
  mod = file_indexes % number_process
  selected_indexes = list(numpy.where(mod==rank)[0])

  return data[selected_indexes]



def load_list_from_resources(database_list, database_resource_name, file_loader, arrange_by_client=False):
  files = []
  for d in database_list:
    db = resources.load_resource(database_resource_name, d)
    files.append(file_loader.load_lists_from_database(db, arrange_by_client))
  files = sum(files,[])
  return files



def load_features_from_dir(directory, file_loader):
  files = os.listdir(directory)
  list_files = []
  for f in files:
    list_files.append(os.path.join(directory,f))
  whole_data = file_loader.load_features_from_list(list_files)

  return whole_data


def compute_likelihood(gmm_stats):
  return gmm_stats.log_likelihood / gmm_stats.t;

