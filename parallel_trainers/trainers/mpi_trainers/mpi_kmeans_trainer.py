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
    self.dim                   = dim
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

    
  def compute_variances_weights(self, data):
    """
    Compute the variances and the weights for each cluster

    Keyword parameters:

    data
      A small part of the whole data
    
    """ 

    n_means, dim = self.e_kmeans_machine.means.shape
    
    cache_means  = numpy.zeros((n_means, dim))
    variances    = numpy.zeros((n_means, dim))
    weights      = numpy.zeros((1,n_means))
    
    #For each small set of day (remember, this code runs in all process), accumulate some statistics
    for d in data:
      i = self.e_kmeans_machine.get_closest_mean(d)[0]
      weights[0,i] += 1
      cache_means += d
      variances += numpy.power(d,2)


    #Preparing the for reduce the statistics
    if(self.rank==0):
      reduced_cache_means = numpy.zeros((n_means, dim))
      reduced_variances   = numpy.zeros((n_means, dim))
      reduced_weights     = numpy.zeros((1,n_means))
    else:
      reduced_cache_means = None
      reduced_variances   = None
      reduced_weights     = None
 
    #Reducing it
    self.communicator.Reduce(cache_means, reduced_cache_means, op=MPI.SUM, root=0)
    self.communicator.Reduce(variances, reduced_variances, op=MPI.SUM, root=0)
    self.communicator.Reduce(weights, reduced_weights, op=MPI.SUM, root=0)

    #Only the rank 0 knows the result of the sum above.
    if(self.rank == 0):

      #print(variances)
      #exit()

      cache_means /= weights.transpose()

      variances /= weights.transpose()
      variances -= numpy.power(cache_means,2)

      weights /= numpy.sum(weights)
      #print(variances)
      #exit()

      
    weights   = self.communicator.bcast(weights, root=0)
    variances = self.communicator.bcast(variances, root=0)   
   
    return weights[0,:], variances



  def get_machine(self):
    return self.e_kmeans_machine
    


