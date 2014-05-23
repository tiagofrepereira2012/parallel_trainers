#!/uisr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Mon Dec 05 12:08:00 CEST 2013

import numpy
import bob

import os
import array



class FileLoader:
  """This class load features files from different formats"""

  def __init__(self,database, from_database=True, file_format="", dim=0, arrange_by_client = False):


    if(from_database):
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

      self.list = fs.training_list(directory_type,"train_extractor", arrange_by_client = arrange_by_client)   
      self.file_format = file_format
      self.arrange_by_client = arrange_by_client

    else:

      if(type(database) == list):
        self.list = database
        self.file_format = file_format
        self.arrange_by_client = arrange_by_client

      else:
 
        f = open(database, "r")
        lines = f.readlines()
        f.close()

        i = 0
        for l in lines:
          l=l.rstrip("\n")
          lines[i] =l
          i = i + 1
 
        self.list              = list(lines)
        self.arrange_by_client = False

    self.dim = dim


  def __call__(self):
    """
    Keyword Parameters:
    """ 
    if(self.arrange_by_client):
      features = []
      for f in self.list:
        features.append(self._load_features_from_list_per_user(f))
      return features
    else:
      return self._load_features_from_list(self.list)



  def _load_features_from_list(self, list_files):
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
      f = self.load_features_from_object(o)
      s = f.shape[0]
      features[i:i+s,:] = f
      i = i + s
    return features


  def _load_features_from_list_per_user(self, list_files):
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
      f = self.load_features_from_object(o)
      features[i,:,:] = f
      i = i + 1

    return features




  def get_shape(self,o):
    """

    Keyword Parameters:
      o
        File object
    """
    f = self.load_features_from_object(o)
    return f.shape



  def load_features_from_object(self,o):
    """
    Load a feature file

    Keyword Parameters:i
      o
        File object
   
    """    

    full_filename = o

    if(full_filename[len(full_filename)-4:len(full_filename)] == "hdf5"):
      return bob.io.load(full_filename)
    else:
      return self.__paramread(full_filename)


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
