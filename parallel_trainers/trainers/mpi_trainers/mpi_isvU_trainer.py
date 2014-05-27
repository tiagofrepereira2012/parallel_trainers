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


class MPIISVUTrainer(MPIEMTrainer):


  def __init__(self, communicator, base_ubm, U_dimension=100, relevance_factor = 4, iterations=10):
    """
    Constructor

    Keyword parameters:

    communicator
      MPI Intracomunicator
    """
    
    MPIEMTrainer.__init__(self,communicator)

    self.base_ubm              = base_ubm
    self.U_dimension           = U_dimension
    self.iterations            = iterations
    #self.convergence_threshold = convergence_threshold
    self.relevance_factor      = relevance_factor

    #Creating the proper trainers and machines
    #Machines and trainers for E step   
    self.e_machine  = bob.machine.ISVBase(self.base_ubm, self.U_dimension)
    self.e_trainer  = bob.trainer.ISVTrainer(relevance_factor = self.relevance_factor)
    if(self.rank==0):
      #Machines and trainers for the M step
      self.m_machine  = bob.machine.ISVBase(self.base_ubm, self.U_dimension)
      self.m_trainer  = bob.trainer.ISVTrainer(relevance_factor = self.relevance_factor)



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
    average_likelihood_previous = float("inf")
    fresh_U = None
    while(run):
      i = i + 1

      if(i>=self.iterations):
        run = False

      if(self.rank==0):
        logging.info("U - Iteration " + str(i))
        logging.info("  E Step")


      ####
      #E Step (computes in each process)
      ####
      self.e_trainer.e_step(self.e_machine,data)

      #Preparing the GMMStatistics for reduce
      if(self.rank==0):
        reduce_u_a1 = numpy.zeros(shape=self.e_trainer.acc_u_a1.shape, dtype=self.e_trainer.acc_u_a1.dtype)
        reduce_u_a2 = numpy.zeros(shape=self.e_trainer.acc_u_a2.shape, dtype=self.e_trainer.acc_u_a2.dtype)
      else:
        reduce_u_a1 = None
        reduce_u_a2 = None

      #Summing up the GMMStatistics
      self.communicator.Reduce(self.e_trainer.acc_u_a1           ,reduce_u_a1           , op=MPI.SUM, root=0)
      self.communicator.Reduce(self.e_trainer.acc_u_a2           ,reduce_u_a2           , op=MPI.SUM, root=0)


      ########
      #M-Step (Only in the root node)
      ########
      if(self.rank==0): #m-step only in the rank 0   
        logging.info("  M Step")

        #new statistiscs
        self.m_trainer.acc_u_a1 = reduce_u_a1
        self.m_trainer.acc_u_a2 = reduce_u_a2

        #trainer for the m-step
        self.m_trainer.m_step(self.m_machine,data)

        #TODO: Testing the convergence

        fresh_U = self.m_machine.u #preparing for send the new means


      #Broadcasting the new means and the stop condition
      run                  = self.communicator.bcast(run, root = 0)
      self.e_machine.u     = self.communicator.bcast(fresh_U,root=0)

  def get_machine(self):
    return self.m_machine
    


