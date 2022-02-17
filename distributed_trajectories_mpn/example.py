import os
import numpy as np
import pandas as pd
from random import seed, random


import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import IntegerType


from distributed_trajectories_mpn.mpn import MPN
from distributed_trajectories_mpn.voronoi import Voronoi
from distributed_trajectories_mpn.ring import get_oneRingData, get_twoRingData, get_threeRingData
from distributed_trajectories_mpn.trajectories import DetermTraj, ProbTraj
from distributed_trajectories_mpn.TM import TM
from distributed_trajectories_mpn.OD import OD
from distributed_trajectories_mpn.time_inhomo import Time_inhomo
from distributed_trajectories_mpn.udfs import prepare_for_plot, plot_sparse, plot_dense, vectorize, stationary,\
                                              vectorConverge, observe_convergence, plot_trend, \
                                              simulate, plot_sim_result, rand_state
from distributed_trajectories_mpn.simulation import Vectorization, Simulation


spark = SparkSession.builder\
    .enableHiveSupport()\
    .appName('distributed_trajectories_mpn')\
    .master('local[*]')\
    .getOrCreate()


mpn_file = 'data/mpn/*'
voronoi_file = 'data/voronoi/*'

firstRing_file = 'data/ring/voronoiUlaStoOreb_bbox_queen1.txt'
secondRing_file = 'data/ring/voronoiUlaStoOreb_bbox_queen2.txt'
thirdRing_file = 'data/ring/voronoiUlaStoOreb_bbox_queen3.txt'


### 0 - PROCESSED DATASET:


## 0.1 - Mobile Phone Network:
mpn_data = MPN(spark, mpn_file).process()

## 0.2 - Vornnoi Polygon:
voronoi_data = Voronoi(spark, voronoi_file).process()

## 0.3 - Ring Distribution:
oneRing_data = get_oneRingData(spark, firstRing_file, secondRing_file, thirdRing_file)
twoRing_data = get_twoRingData(spark, firstRing_file, secondRing_file, thirdRing_file)
threeRing_data = get_threeRingData(spark, firstRing_file, secondRing_file, thirdRing_file)

## 0.4 - Deterministic Trajectories:
deterministic_trajectories = DetermTraj(mpn_data, voronoi_data).make_traj()

## 0.5 - Probabilistic Trajectories:
trajectories = DetermTraj(mpn_data, voronoi_data).join()

trajectories_oneRing = ProbTraj(trajectories, oneRing_data).make_traj()
trajectories_twoRing = ProbTraj(trajectories, twoRing_data).make_traj()
trajectories_threeRing = ProbTraj(trajectories, threeRing_data).make_traj()

probabilistic_trajectories = trajectories_oneRing



### 1 - TRANSITION MATRIX:

## 1.1 - Deterministic Trajectories:
tm_0 = TM(deterministic_trajectories, voronoi_data).make_tm()
plot_sparse(prepare_for_plot(tm_0, 'updates'), 'TM_0.png', 'Transition Matrix (Deterministic)', 'outputs/matrix')

## 2.2 - Probabilistic Trajectories:
tm_1 = TM(probabilistic_trajectories, voronoi_data).make_tm()
matrix = prepare_for_plot(tm_1, 'updates').toarray()
plot_sparse(prepare_for_plot(tm_1, 'updates'), 'TM_1.png', 'Transition Matrix (One Ring): 75% Confidence Level', 'outputs/matrix')
plot_dense(prepare_for_plot(tm_1, 'updates'), 'TM_1_prob.png', 'Transition Matrix (Probabilistic)', 'outputs/matrix')
# tm_GIS = TM(probabilistic_trajectories, voronoi_data).make_tm_GIS()
# tm_GIS.toPandas().to_csv('TM_prepare_for_GIS.csv')

tm_2 = TM(trajectories_twoRing, voronoi_data).make_tm()
plot_sparse(prepare_for_plot(tm_2, 'updates'), 'TM_2.png', 'Transition Matrix (Two Rings): 90% Confidence Level', 'outputs/matrix')

# tm_3 = TM(trajectories_twoRing, voronoi_data).make_tm()
# plot_sparse(prepare_for_plot(tm_3, 'updates'), 'TM_3.png', 'Transition Matrix (Three Rings): 95% Confidence Level', 'outputs/matrix')



### 2 - ORIGIN-DESTINATION MATRIX:

od = OD(probabilistic_trajectories, voronoi_data).make_od()
plot_sparse(prepare_for_plot(od, 'updates'), 'OD.png', 'Origin-Destination Matrix: h = 2 hours', 'outputs/matrix')
# od_GIS = OD(probabilistic_trajectories, voronoi_data).make_od_GIS()
# od_GIS.toPandas().to_csv('OD_prepare_for_GIS.csv')



### 3 - TIME-INHOMOGENEOUS SIMULATION:

## 3.0 - Time-inhomogeneous Trajectories: Probabilistic
## Paremeters are subjected to change
time_inhomogeneous_prob_traj_0 = Time_inhomo(probabilistic_trajectories, 1, 5, 2, 3).make_ti_traj()
time_tm_0 = TM(time_inhomogeneous_prob_traj_0, voronoi_data).make_tm()
plot_sparse(prepare_for_plot(time_tm_0, 'updates'), 'TI_TM_EX0.png', 'Transition Matrix (2am to 3am)', 'outputs/simulation')

time_inhomogeneous_prob_traj_1 = Time_inhomo(probabilistic_trajectories, 1, 5, 8, 9).make_ti_traj()
time_tm_1 = TM(time_inhomogeneous_prob_traj_1, voronoi_data).make_tm()
plot_sparse(prepare_for_plot(time_tm_1, 'updates'), 'TI_TM_EX1.png', 'Transition Matrix (8am to 9pm)', 'outputs/simulation')

time_inhomogeneous_prob_traj_2 = Time_inhomo(probabilistic_trajectories, 1, 5, 9, 10).make_ti_traj()
time_tm_2 = TM(time_inhomogeneous_prob_traj_2, voronoi_data).make_tm()
plot_sparse(prepare_for_plot(time_tm_2, 'updates'), 'TI_TM_EX2.png', 'Transition Matrix (9am to 10am)', 'outputs/simulation')




## 3.1 - Stationary Distribution:

# Initial vector:
init_vector = Vectorization(time_inhomogeneous_prob_traj_1, voronoi_data)\
              .process()\
              .rdd.map(lambda x: vectorize(x))\
              .toDF(['ml_SparseVector', 'np_vector'])
matrix_1 = prepare_for_plot(time_tm_1, 'updates').toarray()

init_vector_dev = stationary(150, init_vector, matrix_1)

vector_1 = init_vector_dev.iloc[[0, -1]]


# Next vector:
next_vector = vectorConverge(spark, init_vector_dev)\
              .rdd.map(lambda x: vectorize(x))\
              .toDF(['ml_SparseVector', 'np_vector'])
matrix_2 = prepare_for_plot(time_tm_2, 'updates').toarray()

next_vector_dev = stationary(50, next_vector, matrix_2)

vector_2 = next_vector_dev.iloc[-1:]


# Testing for convergence:
observe_convergence(init_vector_dev.loc[:, (init_vector_dev != 0).any(axis = 0)], 
                    'SD1_dev.png', 'Convergence Testing (Phase 1)', 'outputs/simulation')
observe_convergence(next_vector_dev.loc[:, (next_vector_dev != 0).any(axis = 0)], 
                    'SD2_dev.png', 'Convergence Testing (Phase 2)', 'outputs/simulation')

# Comparison: Beginning versus End
res = vector_1.append(vector_2).set_index([pd.Index([0,1,2])])
plot_trend(res, 'SD.png', 'Mobility Trend', 'outputs/simulation')




## 3.2 - Simulate discrete markov chain:

# Get initial state for each user
# window = Window.partitionBy(['user_id']).orderBy(F.lit('A'))

# init_state = time_inhomogeneous_prob_traj_1\
#              .withColumn('i', F.row_number().over(window))\
#              .filter(F.col('i') == 1).select('user_id', 'voronoi_id')

# init_state = Simulation(spark, time_inhomogeneous_prob_traj_1, voronoi_data, oneRing_data, matrix, 150).process()
# init_state.toPandas().to_csv('init_state.csv')

# seed(0)
# sim_traj = spark.createDataFrame(simulate(init_state, matrix, 150))

# sim_traj = sim_traj.withColumn('simulated_traj', F.explode(F.split(F.col('simulated_traj'), ',')))\
#                    .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\[', ''))\
#                    .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\]', ''))\
#                    .withColumn('simulated_traj', F.col('simulated_traj').cast(IntegerType()))\
#                    .withColumn('i', F.row_number().over(window))
# sim_traj.toPandas().to_csv('sim_traj.csv')


# simulation = Simulation(spark, time_inhomogeneous_prob_traj_1, voronoi_data, oneRing_data, matrix, 150).process()
# prepare_for_sim_plot = pd.DataFrame(np.vstack(simulation.toPandas()['vector']))
# observe_convergence(prepare_for_sim_plot, 'Sim_dev.png', 'Convergence Testing on Simulation', 'outputs/simulation')
# plot_sim_result(prepare_for_sim_plot, 'Sim.png', 'Simulation Result', 'outputs/simulation')


# simulation_TM = Simulation(spark, time_inhomogeneous_prob_traj_1, voronoi_data, oneRing_data, matrix, 150).process()
# simulated_tm = TM(simulation_TM, voronoi_data).make_sim_tm()
# plot_sparse(prepare_for_plot(simulated_tm, 'updates'), 'SIM_TI_TM.png', 'Simulated Transition Matrix', 'outputs/simulation')



### 4 - SCALIBILITY TEST IN DATABRICKS

# Create Random states
# Paremeters are subjected to change (randUser: number of users; randState: number of frequency)
# randUser = spark.range(1, 1001)
# randState = rand_state(randUser, 288)

# Test Scalability with Transition Matrix
# test_TM = Simulation(spark, randState, voronoi_data, oneRing_data, matrix, 150).scalability_test()
# test_tm = TM(test_TM, voronoi_data).make_sim_tm()
# plot_dense(prepare_for_plot(test_tm, 'updates'), 'TEST_TM_prob.png', 'Scalability Test - Simulated Transition Matrix', 'outputs/scalability')