#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Mon Oct 18 08:47 BRST 2013

import bob
import numpy
from mpi_emtrainer import *
from mpi4py import MPI
from .. import utils

import logging


class MPITVTrainer(MPIEMTrainer):
  """
  MPI Implementation of the Total Variability Matrix
  """


  def __init__(self, communicator, base_ubm, update_sigma=True, T_dimension=100, iterations=10):
    """
    Constructor

    Keyword parameters:

    communicator
      MPI Intracomunicator
    """
    
    MPIEMTrainer.__init__(self,communicator)

    self.base_ubm              = base_ubm
    self.T_dimension           = T_dimension
    self.iterations            = iterations
    self.update_sigma          = update_sigma

    #Creating the proper trainers and machines
    #Machines and trainers for E step   
    self.e_machine  = bob.machine.IVectorMachine(self.base_ubm, rt=T_dimension)
    self.e_trainer  = bob.trainer.IVectorTrainer(update_sigma = self.update_sigma)
    if(self.rank==0):
      #Machines and trainers for the M step
      self.m_machine  = bob.machine.IVectorMachine(self.base_ubm, rt=T_dimension)
      self.m_trainer  = bob.trainer.IVectorTrainer(update_sigma = self.update_sigma)



  def train(self, data):
    """
    Train a machine using a small part of the whole data

    Keyword parameters:

    data
      A small part of the whole data (statistics)
    """
    #initializing the machines
    self.e_trainer.initialize(self.e_machine,data)
    if (self.rank==0):
      self.m_trainer.initialize(self.m_machine, data)

    run = True #flag that controls the iterations of the algorithm along different process
    i = 0
    fresh_T = None
    fresh_sigma = None
    while(run):
      i = i + 1

      if(i>=self.iterations):
        run = False

      if(self.rank==0):
        logging.info("T - Iteration " + str(i))
        logging.info("  E Step")


      ####
      #E Step (computes in each process)
      ####
      self.e_trainer.e_step(self.e_machine,data)

      #Preparing the GMMStatistics for reduce
      if(self.rank==0):
        reduce_acc_fnormij_wij = numpy.zeros(shape=self.e_trainer.acc_fnormij_wij.shape,dtype="float")
        reduce_acc_nij_wij2    = numpy.zeros(shape=self.e_trainer.acc_nij_wij2.shape,dtype="float")
        reduce_acc_nij         = numpy.zeros(shape=self.e_trainer.acc_nij.shape,dtype="float")
        reduce_acc_snormij     = numpy.zeros(shape=self.e_trainer.acc_snormij.shape,dtype="float")

      else:
        reduce_acc_fnormij_wij = None
        reduce_acc_nij_wij2    = None
        reduce_acc_nij         = None
        reduce_acc_snormij     = None

      #Summing up the Statistics
      self.communicator.Reduce(self.e_trainer.acc_fnormij_wij , reduce_acc_fnormij_wij , op=MPI.SUM, root=0)
      self.communicator.Reduce(self.e_trainer.acc_nij_wij2    , reduce_acc_nij_wij2    , op=MPI.SUM, root=0)
      self.communicator.Reduce(self.e_trainer.acc_nij         , reduce_acc_nij         , op=MPI.SUM, root=0)
      self.communicator.Reduce(self.e_trainer.acc_snormij     , reduce_acc_snormij     , op=MPI.SUM, root=0)


      ########
      #M-Step (Only in the root node)
      ########
      if(self.rank==0): #m-step only in the rank 0   
        logging.info("  M Step")

        #new statistiscs
        self.m_trainer.acc_fnormij_wij = reduce_acc_fnormij_wij
        self.m_trainer.acc_nij_wij2    = reduce_acc_nij_wij2   
        self.m_trainer.acc_nij         = reduce_acc_nij        
        self.m_trainer.acc_snormij     = reduce_acc_snormij    


        #trainer for the m-step
        self.m_trainer.m_step(self.m_machine,data)

        #TODO: Testing the convergence
        fresh_T     = self.m_machine.t #preparing for send the new T matriz and sigma
        fresh_sigma = self.m_machine.sigma


      #Broadcasting the new means and the stop condition
      run                  = self.communicator.bcast(run, root = 0)
      self.e_machine.t     = self.communicator.bcast(fresh_T,root=0)
      self.e_machine.sigma = self.communicator.bcast(fresh_sigma,root=0)

  def get_machine(self):
    return self.m_machine
    

  def save(self, file_name):
    hdf5file = bob.io.HDF5File(file_name,openmode_string='w')
    bob.machine.IVectorMachine.save(self.get_machine(), hdf5file)

