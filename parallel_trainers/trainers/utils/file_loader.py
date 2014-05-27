#!/uisr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Mon Dec 05 12:08:00 CEST 2013

import numpy
import bob

import os
import array



class FileLoader:
  """This class load features files from different formats"""

  def __init__(self, dim=40):
    self.dim = dim

  def load_lists_from_database(self, database, arrange_by_client=False):

    from facereclib.toolchain import FileSelector
    fs = FileSelector(
        database,
        "",
        "",
        database.original_directory,
        "",
        "",
        "",
        "",
        "",
        zt_score_directories = None,
        default_extension = '.hdf5'
    )

    directory_type = "features"
    file_list      = fs.training_list(directory_type,"train_extractor", arrange_by_client = arrange_by_client)
    return file_list

    #if(arrange_by_client):
      #features = []
      #for f in file_list:
        #features.append(self.load_features_from_list_per_user(f))
      #return features
    #else:
      #return self._load_features_from_list(file_list)



  def load_features_from_list(self, list_files):
    #Counting for pre-allocation
    dim     = 0
    counter = 0
    for o in list_files:
      s,self.dim = self.get_shape(o)
      counter = counter + s

    #pre-allocating
    features = numpy.zeros(shape=(counter,self.dim), dtype='float')

    #Loading the feaiures
    i = 0
    for o in list_files:
      f = self.load_features_from_file(o)
      s = f.shape[0]
      features[i:i+s,:] = f
      i = i + s
    return features


  def load_features_from_list_per_user(self, list_files):
    #Counting for pre-allocation
 
    if(len(list_files) > 0):
      size,self.dim = self.get_shape(list_files[0])
    else:
      raise ValueError("Empty list!!")

    #pre-allocating
    features = numpy.zeros(shape=(len(list_files),size,self.dim), dtype='float')

    #Loading the feaiures
    i = 0
    for o in list_files:
      f = self.load_features_from_file(o)
      features[i,:,:] = f
      i = i + 1

    return features




  def get_shape(self,o):
    """

    Keyword Parameters:
      o
        File object
    """
    f = self.load_features_from_file(o)
    return f.shape



  def load_features_from_file(self, file_name):
    """
    Load a feature file

    Keyword Parameters:
      file_name
        File name
    """    
    if(file_name[len(file_name)-4:len(file_name)] == "hdf5"):
      return bob.io.load(file_name)
    else:
      return self.__paramread(file_name)



  def __paramread(self,arquivo):
    """
    Converts a sequence of floats (binary format) in a numpy array
    """ 
 
    numberOfFloats = os.path.getsize(arquivo)/4  # each feature is a float
    file = open(arquivo,mode = 'rb')  # opens feature input file
    parameters = array.array('f')
    parameters.fromfile(file,numberOfFloats)
    parameters = numpy.array(parameters, dtype=numpy.float64)
    file.close()

    number_of_vectors = parameters.shape[0] / float(self.dim)
    parameters = numpy.reshape(parameters,(int(number_of_vectors),int(self.dim)))
    
    return parameters
