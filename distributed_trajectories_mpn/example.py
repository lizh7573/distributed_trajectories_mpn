



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
from distributed_trajectories_mpn.udfs import prepare_for_plot, plot_sparse, plot_dense


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
# plot_sparse(prepare_for_plot(tm_0, 'updates'), 'TM_0.png', 'Transition Matrix (Deterministic)', 'outputs/tm')

## 2.2 - Probabilistic Trajectories:
tm_1 = TM(probabilistic_trajectories, voronoi_data).make_tm()
# plot_sparse(prepare_for_plot(tm_1, 'updates'), 'TM_1.png', 'Transition Matrix (One Ring): 75% Confidence Level', 'outputs/tm')
# plot_dense(prepare_for_plot(tm_1, 'updates'), 'TM_1_prob.png', 'Transition Matrix (Probabilistic)', 'outputs/tm')
tm_GIS = TM(probabilistic_trajectories, voronoi_data).make_tm_GIS()
tm_GIS.toPandas().to_csv('TM_prepare_for_GIS.csv')

# tm_2 = TM(trajectories_twoRing, voronoi_data).make_tm()
# plot_sparse(prepare_for_plot(tm_2, 'updates'), 'TM_2.png', 'Transition Matrix (Two Rings): 90% Confidence Level', 'outputs/tm')



### 2 - ORIGIN-DESTINATION MATRIX:

od = OD(probabilistic_trajectories, voronoi_data).make_od()
# plot_sparse(prepare_for_plot(od, 'updates'), 'OD.png', 'Origin-Destination Matrix: h = 2 hours', 'outputs/od')
# plot_dense(prepare_for_plot(od, 'updates'), 'OD_prob.png', 'Origin-Destination Matrix: h = 2 hours', 'outputs/od')
od_GIS = OD(probabilistic_trajectories, voronoi_data).make_od_GIS()
od_GIS.toPandas().to_csv('OD_prepare_for_GIS.csv')