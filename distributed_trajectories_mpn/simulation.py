"""
Simulation
==========
"""

from random import seed, random

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import IntegerType
from distributed_trajectories_mpn.udfs import sim_vectorize, simulate





class Vectorization:

    def __init__(self, df, voronoi):
        self.df = df
        self.voronoi = voronoi

    def set_help_cols(self):
        window = Window.partitionBy(['user_id']).orderBy('timestamp')
        self.df = self.df.withColumn('i', F.row_number().over(window)).filter(F.col('i') == 1)\
                         .withColumn('states', F.explode('states'))\
                         .select('user_id', F.col('states.neighbors').alias('v_col'), 
                                                          F.col('states.props').alias('v_val'))
        return self.df

    def coords_simplify(self):
        self.df = self.df.join(F.broadcast(self.voronoi.select('voronoi_id', 'simplified_id')), 
                                           self.df.v_col == self.voronoi.voronoi_id, 'inner')\
                         .select('user_id', 'simplified_id', 'v_val')\
                         .withColumnRenamed('simplified_id', 'v_col')
        return self.df

    def weighted_average(self):
        count = self.df.agg(F.countDistinct('user_id')).collect()[0][0]
        self.df = self.df\
            .select('v_col', 'v_val')\
            .groupBy('v_col')\
            .agg(F.sum('v_val').alias('v_val'))\
            .withColumn('v_val', F.col('v_val')/F.lit(count))\
            .orderBy('v_col')
        return self.df

    def collect(self):
        self.df = self.df.agg(F.collect_list('v_col').alias('col'),
                              F.collect_list('v_val').alias('val'))
        return self.df
  
    def process(self):
        self.set_help_cols()
        self.coords_simplify()
        self.weighted_average()
        self.collect()
        return self.df




class Simulation:

    def __init__(self, spark, df, voronoi, noise, matrix1, N1):
        self.spark = spark
        self.df = df
        self.voronoi = voronoi
        self.noise = noise
        self.matrix1 = matrix1
        self.N1 = N1
        # self.matrix2 = matrix2
        # self.N2 = N2

    def intercept(self):
        window = Window.partitionBy(['user_id']).orderBy(F.lit('A'))
        self.df = self.df.withColumn('i', F.row_number().over(window))\
                         .filter(F.col('i') == 1).select('user_id', 'voronoi_id')\
                         .join(F.broadcast(self.voronoi.select('voronoi_id', 'simplified_id')), ['voronoi_id'], 'inner')\
                         .select('user_id', 'simplified_id')\
                         .withColumnRenamed('simplified_id', 'voronoi_id')
        return self.df

    def simulate(self):
        seed(0)
        self.df = self.spark.createDataFrame(simulate(self.df, self.matrix1, self.N1))
        return self.df

    def clean(self):
        window = Window.partitionBy(['user_id']).orderBy(F.lit('A'))
        self.df = self.df.withColumn('simulated_traj', F.explode(F.split(F.col('simulated_traj'), ',')))\
                         .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\[', ''))\
                         .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\]', ''))\
                         .withColumn('simulated_traj', F.col('simulated_traj').cast(IntegerType()))\
                         .withColumn('i', F.row_number().over(window))\
                         .drop('voronoi_id')
        return self.df

    def add_noise(self):
        self.df = self.df.join(F.broadcast(self.voronoi), 
                               self.df.simulated_traj == self.voronoi.select('voronoi_id', 'simplified_id').simplified_id, 
                               how = 'inner')\
                         .join(F.broadcast(self.noise), ['voronoi_id'], how = 'inner')\
                         .orderBy(['user_id', 'i']).select('user_id', 'simulated_traj', 'states', 'i')
        return self.df

    def reformulate_TM(self):
        self.df = self.df.select('user_id', 'states', 'i')
        return self.df

    def set_help_cols(self):
        self.df = self.df\
             .withColumn('states', F.array_sort(F.col('states')))\
             .withColumn('col', F.col('states').__getitem__('neighbors'))\
             .withColumn('val', F.col('states').__getitem__('props'))\
             .select('user_id', 'simulated_traj', 'col', 'val', 'i')
        return self.df

    def vectorization(self):
        self.df = self.df\
            .rdd.map(lambda x: sim_vectorize(x))\
            .toDF(['user_id', 'simulated_traj', 'sim_vector', 'i'])
        return self.df

    def simulate_vector(self):
        w = Window().orderBy('i') 
        self.df = self.df\
             .select(['i', 'sim_vector']).groupBy('i')\
             .agg(F.array(*[F.avg(F.col('sim_vector')[m]) for m in range(4069+1)]).alias('sim_vector')).orderBy('i')\
             .withColumn('avg_sim_vector', F.array(*[F.avg(F.col('sim_vector')[n]).over(w) for n in range(4069+1)]))\
             .drop('i', 'sim_vector').withColumnRenamed('avg_sim_vector', 'vector')
        return self.df

    def process(self):
        self.intercept()
        self.simulate()
        self.clean()
        self.add_noise()
        self.reformulate_TM()
        # self.set_help_cols()
        # self.vectorization()
        # self.simulate_vector()
        return self.df

    def scalability_test(self):
        self.add_noise()
        self.reformulate_TM()
        return self.df



    



