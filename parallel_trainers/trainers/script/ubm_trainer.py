#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Fri Dec 08 14:22 BRST 2013

import bob
import numpy
import argparse
from .. import utils
from mpi4py import MPI

from parallel_trainers.trainers.utils.file_loader import FileLoader


def train_kmeans(data, gaussians, dim):
  #Starting the kmeans (MUST BE SERIAL)
  kmeans = bob.machine.KMeansMachine(gaussians, dim)
  kmeansTrainer                       = bob.trainer.KMeansTrainer()
  kmeansTrainer.max_iterations        = 500
  kmeansTrainer.convergence_threshold = 0.0001
  kmeansTrainer.rng                   = bob.core.random.mt19937(5489)
  kmeansTrainer.train(kmeans, data)

  #GMM Trainer and machine
  [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(data)
  means = kmeans.means

  return means,variances,weights




def main():

  DATABASES_RESOURCE_NAME       = "databases"
  K_MEANS_MAX_ITERATIONS        = 50
  K_MEANS_CONVERGENCE_THRESHOLD = 0.0001


  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-i', '--input-dir', metavar='DIR', type=str, dest='input_dir', default='', help='Base directory for the features.')
  parser.add_argument('-o', '--output-file', metavar='DIR', type=str, dest='output_file', default='UBM.hdf5', help='Output file.')

  parser.add_argument('-g', '--gaussians', type=int, dest='gaussians', default=16, help='Number of Gaussians.')

  parser.add_argument('-t', '--convergence_threshold', type=float, dest='convergence_threshold', default=0.0001, help='Convergence threshold.')

  parser.add_argument('-r', '--iterations', type=int, dest='iterations', default=10, help='Number of iterations.')

  parser.add_argument('-d','--databases', nargs = '+', required = True,
        help = 'Database and the protocol; registered databases are: %s'%utils.resources.get_available_resources(DATABASES_RESOURCE_NAME))

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')

  parser.add_argument('-u', '--ubm-initialization-file', type=str, dest='ubm_initialization', default='', help='Use the paramenters of this UBM for initialization instead of use kmeans')


  args = parser.parse_args()

  input_dir             = args.input_dir
  output_file           = args.output_file
  gaussians             = args.gaussians
  convergence_threshold = args.convergence_threshold
  verbose               = args.verbose
  databases             = args.databases
  ubm_initialization    = args.ubm_initialization
  iterations            = args.iterations
  

  ###############
  #SETTING UP MPI
  ##############
  comm        = MPI.COMM_WORLD #Starting MPI
  rank        = comm.Get_rank() #RANK i.e. How am I?
  size        = comm.Get_size() #Number of parallel process for T-matrix calculation

  
  means        = None
  variances    = None
  weights      = None
  partial_data = None
  fresh_means  = None

  ####
  # Loading features
  ####
  if(rank==0):
    print("Loading features...")
    whole_data = utils.load_features_from_resources(databases, DATABASES_RESOURCE_NAME)
    #whole_data = numpy.random.rand(50000, 44)

    #sending the proper data for each node
    print("Transmitting proper data to each node...")
    for i in range(1,size):
      partial_data = utils.select_data(whole_data, size, i)
      comm.send(partial_data,dest=i,tag=11)
    partial_data = utils.select_data(whole_data, size, 0)#data for the rank 0

  else: #if rank!=0
    #receiving the data for each node
    partial_data = comm.recv(source=0,tag=11)
    

  dim = partial_data.shape[1] #Fetching the feature dimensionality
  ####
  # K-Means
  ###

  #Setting UP the initialization
  if(ubm_initialization != ""):
    #if it was from a previous UBM, load from there
    print("Load UBM from: " + ubm_initialization)
    previous_ubm = utils.load_gmm_from_file(ubm_initialization, gaussians, dim)
    means     = previous_ubm.means
    variances = previous_ubm.variances
    weights   = previous_ubm.weights
  else:
    #if it was not, RUN the kmeans

    e_kmeans_machine = bob.machine.KMeansMachine(gaussians,dim)
    if(rank==0):
      print("Parallel Kmeans")
      m_kmeans_machine = bob.machine.KMeansMachine(gaussians,dim)

    ####
    # RUN K-means
    ####
    run = True #flag that controls the iterations of the algorithm
    i = 0
    average_likelihood_previous = float("inf")
    while(run):
      i = i + 1
      if(i >= K_MEANS_MAX_ITERATIONS):
        run  = False

      if(rank==0):
        print("KMeans - Iteration " + str(i))
        print("  E Step")


      ####
      #K-means E Step
      ####
      e_kmeans_trainer                       = bob.trainer.KMeansTrainer()
      e_kmeans_trainer.initialize(e_kmeans_machine,partial_data)
      e_kmeans_trainer.e_step(e_kmeans_machine,partial_data)

      #Preparing the KMEANS Statistics for reduce
      if(rank==0):
        reduce_zeroeth_order_statistics = numpy.zeros(e_kmeans_trainer.zeroeth_order_statistics.shape, e_kmeans_trainer.zeroeth_order_statistics.dtype)
        reduce_first_order_statistics   = numpy.zeros(e_kmeans_trainer.first_order_statistics.shape, e_kmeans_trainer.first_order_statistics.dtype)
        reduce_average_min_distance    = numpy.zeros((1),dtype='float')
        #reduce_n_samples               = numpy.zeros((1),int)
      else:
        reduce_zeroeth_order_statistics = None
        reduce_first_order_statistics   = None
        reduce_average_min_distance    = None
        #reduce_n_samples               = None
 

      #Summing up the K Means Statistics
      comm.Reduce(numpy.array([e_kmeans_trainer.average_min_distance])     ,reduce_average_min_distance     , op=MPI.SUM, root=0)
      #comm.Reduce(numpy.array([partial_data.shape[0]])                     ,reduce_n_samples                , op=MPI.SUM, root=0)
      comm.Reduce(e_kmeans_trainer.zeroeth_order_statistics                ,reduce_zeroeth_order_statistics , op=MPI.SUM, root=0)
      comm.Reduce(e_kmeans_trainer.first_order_statistics                  ,reduce_first_order_statistics   , op=MPI.SUM, root=0)
 

      ########
      #K Means M-Step
      ########

      if(rank==0): #m-step only in the rank 0   
        print("  M Step")

        # Creates the KMeansTrainer
        m_kmeans_trainer = bob.trainer.KMeansTrainer()
        m_kmeans_trainer.initialize(m_kmeans_machine, partial_data)

        m_kmeans_trainer.zeroeth_order_statistics = reduce_zeroeth_order_statistics
        m_kmeans_trainer.first_order_statistics   = reduce_first_order_statistics
        m_kmeans_trainer.average_min_distance     = reduce_average_min_distance[0]

        m_kmeans_trainer.m_step(m_kmeans_machine, partial_data) #m-step

        ###########
        #testing convergence
        ##########
        average_likelihood = m_kmeans_trainer.average_min_distance
        conv               = abs((average_likelihood_previous - average_likelihood)/average_likelihood_previous)
        print(conv)
        if(conv < K_MEANS_CONVERGENCE_THRESHOLD):
          run = False
        average_likelihood_previous = average_likelihood



        fresh_means = m_kmeans_machine.means

      #Broadcasting the new means and the stop condition
      run                    = comm.bcast(run, root = 0)
      e_kmeans_machine.means = comm.bcast(fresh_means,root=0)


    ##Broadcasting the means, variances and weights    
    if(rank==0):

      if(ubm_initialization == ""):
        means = m_kmeans_machine.means
        [variances, weights] = m_kmeans_machine.get_variances_and_weights_for_each_cluster(whole_data)

    #Fetching the new means, variance and weight to start the UBM training
    means     = comm.bcast(means, root=0)
    variances = comm.bcast(variances, root=0)
    weights   = comm.bcast(weights, root=0)





  #####################
  # UBM Trainning
  #####################
  if(rank==0):
    print("############################")
    print("UBM - Training")
    #Creating the M machine (the Machine that will run the m-step)
    m_machine                 = bob.machine.GMMMachine(gaussians, dim)
    m_machine.means           = means
    m_machine.variances       = variances
    m_machine.weights         = weights

  #Creating the E machine (the Machine that will run the e step)
  e_machine                 = bob.machine.GMMMachine(gaussians,dim)
  e_machine.means           = means
  e_machine.variances       = variances
  e_machine.weights         = weights
  

  ###################
  ###### START
  ###################
  average_likelihood_previous = float("inf")

  fresh_means = None  
  run         = True #flaging the stop condition
  i = 0
  while(run):
    i = i + 1
    if(i >= iterations):
      run  = False
  
    if(rank==0):
      print("Iteration " + str(i))
      print("  E Step")

    ############
    # E-STEP
    ###########
    trainer = bob.trainer.ML_GMMTrainer(update_means=True, update_variances=False, update_weights=False)
    trainer.initialize(e_machine,partial_data)
    trainer.e_step(e_machine,partial_data) #e-step
    gmm_stats              = trainer.gmm_statistics #fetching each GMMStats

    #Preparing the GMMStatistics for reduce
    if(rank==0):

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
    comm.Reduce(numpy.array([gmm_stats.t])           ,reduce_t              , op=MPI.SUM, root=0)
    comm.Reduce(gmm_stats.n                          ,reduce_n              , op=MPI.SUM, root=0)
    comm.Reduce(numpy.array(gmm_stats.log_likelihood), reduce_log_likelihood, op=MPI.SUM, root=0)
    comm.Reduce(gmm_stats.sum_px                     ,reduce_sum_px         , op=MPI.SUM, root=0)
    comm.Reduce(gmm_stats.sum_pxx                    ,reduce_sum_pxx        , op=MPI.SUM, root=0)


    ########
    # M-Step
    ########
   
    if(rank==0): #m-step only in the rank 0   

      print("  M Step")

      #new statistiscs
      sum_stats = bob.machine.GMMStats(gaussians, dim)
      sum_stats.n                    = reduce_n
      sum_stats.t                    = int(reduce_t[0])
      sum_stats.log_likelihood       = reduce_log_likelihood[0]
      sum_stats.sum_px               = reduce_sum_px
      sum_stats.sum_pxx              = reduce_sum_pxx

      #trainer for the m-step
      m_trainer = bob.trainer.ML_GMMTrainer(update_means=True, update_variances=False, update_weights=False)
      m_trainer.initialize(m_machine,whole_data)
      m_trainer.gmm_statistics = sum_stats
      m_trainer.m_step(m_machine,whole_data)
 
      ###########
      #testing convergence
      ##########
      average_likelihood = utils.compute_likelihood(sum_stats)
      conv               = abs((average_likelihood_previous - average_likelihood)/average_likelihood_previous)
      #print(conv)
      if(conv < convergence_threshold):
        run = False
      average_likelihood_previous = average_likelihood
     
      fresh_means = m_machine.means #preparing for send the new means

    #Broadcasting the new means and the stop condition
    run           = comm.bcast(run, root = 0)
    e_machine.means = comm.bcast(fresh_means,root=0)


  if(rank==0):
    print("Saving GMM ...")
    utils.save_gmm(m_machine, output_file)
    print("End!")



if __name__ == "__main__":
  main()

