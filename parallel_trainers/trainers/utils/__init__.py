#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>

import resources
import file_loader
import numpy
import bob
from file_loader import FileLoader



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


def select_data(data, number_process, rank):
  """
  Split the data according a rank ID and the number of process
  """
  file_indexes = numpy.array(range(data.shape[0]))

  #Selecting the indexes for each rank
  mod = file_indexes % number_process
  selected_indexes = list(numpy.where(mod==rank)[0])

  return data[selected_indexes]


def load_features_from_resources(database_list, database_resource_name, arrange_by_client=False):
  """
  Load a set of features from a database resource

  TODO: Concatenate numpy arrays
  """
  whole_data = None
  for d in database_list:
    db = resources.load_resource(database_resource_name, d)
    file_loader = FileLoader(db, arrange_by_client=arrange_by_client)
    whole_data = file_loader()
  
  return whole_data




def compute_likelihood(gmm_stats):
  return gmm_stats.log_likelihood / gmm_stats.t;

