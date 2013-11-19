#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Mon Oct 18 08:47 BRST 2013

import bob
import numpy
from mpi_emtrainer import *
from mpi4py import MPI
from .. import utils


class MPIUBMTrainer(MPIEMTrainer):


  def __init__(self, communicator, base_ubm, iterations=20, convergence_threshold = 0.0001):
    """
    Constructor

    Keyword parameters:

    communicator
      MPI Intracomunicator
    """
    
    MPIEMTrainer.__init__(self,communicator)

    self.base_ubm            = base_ubm
    self.iterations            = iterations
    self.convergence_threshold = convergence_threshold


    #Creating the proper trainers and machines

    #Machines and trainers for E step   
    self.e_machine = bob.machine.GMMMachine(self.base_ubm)
    self.e_trainer  = bob.trainer.ML_GMMTrainer(update_means=True, update_variances=False, update_weights=False)
    if(self.rank==0):
      #Machines and trainers for the M step
      self.m_machine = bob.machine.GMMMachine(self.base_ubm)
      self.m_trainer = bob.trainer.ML_GMMTrainer(update_means=True, update_variances=False, update_weights=False)



  def train(self, data):
    """
    Train a machine using a small part of the whole data

    Keyword parameters:

    data
      A small part of the whole data
    """
        
    #initializing the machines
    self.e_trainer.initialize(self.e_machine,data)
    if (self.rank==0):
      self.m_trainer.initialize(self.m_machine, data)

    run = True #flag that controls the iterations of the algorithm along different process
    i = 0
    average_likelihood_previous = float("inf")
    fresh_means = None
    while(run):
      i = i + 1

      if(self.rank==0):
        print("UBM - Iteration " + str(i))
        print("  E Step")


      ####
      #E Step (computes in each process)
      ####
      self.e_trainer.e_step(self.e_machine,data)
      gmm_stats = self.e_trainer.gmm_statistics 

      #Preparing the GMMStatistics for reduce
      if(self.rank==0):
        reduce_t              = numpy.zeros((1),dtype=int)
        reduce_n              = numpy.zeros(shape=gmm_stats.n.shape, dtype=gmm_stats.n.dtype)
        reduce_log_likelihood = numpy.zeros((1), dtype=float)
        reduce_sum_px         = numpy.zeros(shape=gmm_stats.sum_px.shape, dtype=gmm_stats.sum_px.dtype)
        reduce_sum_pxx        = numpy.zeros(shape=gmm_stats.sum_pxx.shape, dtype=gmm_stats.sum_pxx.dtype)
      else:
        reduce_t              = None
        reduce_n              = None
        reduce_log_likelihood = None
        reduce_sum_px         = None
        reduce_sum_pxx        = None

      #Summing up the GMMStatistics
      self.communicator.Reduce(numpy.array([gmm_stats.t])           ,reduce_t              , op=MPI.SUM, root=0)
      self.communicator.Reduce(gmm_stats.n                          ,reduce_n              , op=MPI.SUM, root=0)
      self.communicator.Reduce(numpy.array(gmm_stats.log_likelihood), reduce_log_likelihood, op=MPI.SUM, root=0)
      self.communicator.Reduce(gmm_stats.sum_px                     ,reduce_sum_px         , op=MPI.SUM, root=0)
      self.communicator.Reduce(gmm_stats.sum_pxx                    ,reduce_sum_pxx        , op=MPI.SUM, root=0)


      ########
      #M-Step (Only in the root node)
      ########
      if(self.rank==0): #m-step only in the rank 0   
        print("  M Step")

        #new statistiscs
        sum_stats = bob.machine.GMMStats(self.base_ubm.dim_c, self.base_ubm.dim_d)
        sum_stats.n                    = reduce_n
        sum_stats.t                    = int(reduce_t[0])
        sum_stats.log_likelihood       = reduce_log_likelihood[0]
        sum_stats.sum_px               = reduce_sum_px
        sum_stats.sum_pxx              = reduce_sum_pxx

        #trainer for the m-step
        m_trainer = bob.trainer.ML_GMMTrainer(update_means=True, update_variances=False, update_weights=False)
        m_trainer.initialize(self.m_machine,data)
        m_trainer.gmm_statistics = sum_stats
        m_trainer.m_step(self.m_machine,data)

        ###########
        #testing convergence
        ##########
        average_likelihood = utils.compute_likelihood(sum_stats)
        conv               = abs((average_likelihood_previous - average_likelihood)/average_likelihood_previous)
        if(conv < self.convergence_threshold):
          run = False
        average_likelihood_previous = average_likelihood

        fresh_means = self.m_machine.means #preparing for send the new means


      #Broadcasting the new means and the stop condition
      run                  = self.communicator.bcast(run, root = 0)
      self.e_machine.means = self.communicator.bcast(fresh_means,root=0)

  def get_machine(self):
    return self.m_machine
    


