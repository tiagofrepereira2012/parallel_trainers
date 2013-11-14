#!/uisr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Mon Dec 05 12:08:00 CEST 2013

from facereclib.toolchain import FileSelector
import numpy
import bob


class FileLoader:
  """This class load features files from different formats"""

  def __init__(self,database, file_format="", arrange_by_client = False):

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


  def __call__(self):
    """
    Keyword Parameters:
    """ 
    if(self.arrange_by_client):
      features = []
      for f in self.list:
        features.append(self._load_features_from_list(f))
      return features
    else:
      return self._load_features_from_list(self.list)



  def _load_features_from_list(self, list_files):
    #Counting for pre-allocation
    dim     = 0
    counter = 0
    for o in list_files:
      s,dim = self.get_shape(o)
      counter = counter + s

    #pre-allocating
    features = numpy.zeros(shape=(counter,dim), dtype='float')

    #Loading the feaiures
    i = 0
    for o in list_files:
      f = self.load_features_from_object(o)
      s = f.shape[0]
      features[i:i+s,:] = f
      i = i + s

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

    Keyword Parameters:
      o
        File object
   
    """

    full_filename = o
    return bob.io.load(full_filename)
