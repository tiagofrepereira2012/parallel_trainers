#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Mon Dec 05 12:08:00 CEST 2013

from facereclib.toolchain import FileSelector
import numpy


class FileLoader:
  """This class load features files from different formats"""

  def __init__(self,database, file_format=""):

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

    self.list = fs.training_list("features","train_extractor")
    self.file_format = file_format


  def __call__(self):
    """
    Keyword Parameters:
    """

    #Counting for pre-allocation
    dim     = 0
    counter = 0
    for o in self.list:
      s,dim = get_shape(o)
      counter = counter + s

    #pre-allocating
    features = numpy.zeros(shape=(counter,dim), dtype='float')
    
    #Loading the feaiures
    i = 0
    for o in list_:
      f = load_features_from_object(o)
      s = f.shape[0]
      features[i:i+s,:] = f
      i = i + s


  def get_shape(self,o):
    """

    Keyword Parameters:
      o
        File object
    """

    f = load_features_from_object(o)
    return f.shape



  def load_features_from_object(self,o):
    """
    Load a feature file

    Keyword Parameters:
      o
        File object
   
    """

    full_filename = os.path.join(original_directory,o.path+extension)
    return bob.io.load(full_filename)
