#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Mon Oct 18 08:47 BRST 2013

import bob
import numpy
from mpi_emtrainer import *
from mpi4py import MPI
import logging


class MPIKmeansTrainer(MPIEMTrainer):


  def __init__(self, communicator, n_means, dim, iterations=500, convergence_threshold = 0.0001):
    """
    Constructor

    Keyword parameters:

    communicator
      MPI Intracomunicator
    """
    
    MPIEMTrainer.__init__(self,communicator)

    self.n_means               = n_means
    self.dim                 = dim
    self.iterations            = iterations
    self.convergence_threshold = convergence_threshold


    #Creating the proper trainers and machines

    #Machines and trainers for E step   
    self.e_kmeans_machine = bob.machine.KMeansMachine(self.n_means, self.dim)
    self.e_kmeans_trainer                       = bob.trainer.KMeansTrainer()
    if(self.rank==0):
      #Machines and trainers for the M step
      self.m_kmeans_machine = bob.machine.KMeansMachine(self.n_means,self.dim)
      self.m_kmeans_trainer = bob.trainer.KMeansTrainer()



  def train(self, data):
    """
    Train a machine using a small part of the whole data

    Keyword parameters:

    data
      A small part of the whole data
    """
    random_means = None
    if(self.rank == 0):
      random_means = numpy.random.rand(self.n_means, self.dim)

    random_means = self.communicator.bcast(random_means,root=0) #Broadcasting the random means

    #initializing the machines
    self.e_kmeans_trainer.initialize(self.e_kmeans_machine,data)
    self.e_kmeans_machine.means = random_means
    if (self.rank==0):
      #self.m_kmeans_trainer.rng = bob.core.random.mt19937(5489)
      self.m_kmeans_trainer.initialize(self.m_kmeans_machine, data)
      self.m_kmeans_machine.means = random_means

    run = True #flag that controls the iterations of the algorithm along different process
    i = 0
    average_likelihood_previous = float("inf")
    fresh_means = None
    while(run):
      i = i + 1

      if(self.rank==0):
        logging.info("KMeans - Iteration " + str(i))
        logging.info("  E Step")


      ####
      #E Step (computes in each process)
      ####
      self.e_kmeans_trainer.e_step(self.e_kmeans_machine,data)

      #Preparing the KMEANS Statistics for reduce
      if(self.rank==0):
        reduce_zeroeth_order_statistics = numpy.zeros(self.e_kmeans_trainer.zeroeth_order_statistics.shape, self.e_kmeans_trainer.zeroeth_order_statistics.dtype)
        reduce_first_order_statistics   = numpy.zeros(self.e_kmeans_trainer.first_order_statistics.shape, self.e_kmeans_trainer.first_order_statistics.dtype)
        reduce_average_min_distance    = numpy.zeros((1),dtype='float')
      else:
        reduce_zeroeth_order_statistics = None
        reduce_first_order_statistics   = None
        reduce_average_min_distance    = None


      #Summing up the K Means Statistics
      self.communicator.Reduce(numpy.array([self.e_kmeans_trainer.average_min_distance])     ,reduce_average_min_distance     , op=MPI.SUM, root=0)
      self.communicator.Reduce(self.e_kmeans_trainer.zeroeth_order_statistics                ,reduce_zeroeth_order_statistics , op=MPI.SUM, root=0)
      self.communicator.Reduce(self.e_kmeans_trainer.first_order_statistics                  ,reduce_first_order_statistics   , op=MPI.SUM, root=0)

      ########
      #M-Step (Only in the root node)
      ########
      if(self.rank==0): #m-step only in the rank 0   
        logging.info("  M Step")

        # Creates the KMeansTrainer
        self.m_kmeans_trainer.zeroeth_order_statistics = reduce_zeroeth_order_statistics
        self.m_kmeans_trainer.first_order_statistics   = reduce_first_order_statistics
        self.m_kmeans_trainer.average_min_distance     = reduce_average_min_distance[0]

        self.m_kmeans_trainer.m_step(self.m_kmeans_machine, data) #m-step

        ###########
        #testing convergence
        ##########
        average_likelihood = self.m_kmeans_trainer.average_min_distance
        conv               = abs((average_likelihood_previous - average_likelihood)/average_likelihood_previous)
        if((conv < self.convergence_threshold) or (i >= self.iterations)):
          run = False
        average_likelihood_previous = average_likelihood

        fresh_means = self.m_kmeans_machine.means

      #Broadcasting the new means and the stop condition
      run                    = self.communicator.bcast(run, root = 0)
      self.e_kmeans_machine.means = self.communicator.bcast(fresh_means,root=0)

    

  def get_machine(self):
    return self.m_kmeans_machine
    


