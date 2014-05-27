#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Mon Oct 18 08:47 BRST 2013


class MPIEMTrainer():


  def __init__(self,communicator):
    """
    Constructor

    Keyword parameters:

    communicator
      MPI Intercomunicator
    """

    self.communicator = communicator
    self.rank = communicator.rank  #RANK i.e. Who am I?
    self.size = communicator.size #Number of allocated procces


  def train(self, data):
    """
    Train a machine using a small part of the whole data

    Keyword parameters:

    data
      A small part of the whole data
    """
    raise NotImplementedError()


  def get_machine(self):
    """
    Get the trained machine (M Machine)
    """

    raise NotImplementedError()
