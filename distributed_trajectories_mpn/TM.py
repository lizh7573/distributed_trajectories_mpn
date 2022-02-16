"""
Transition Matrix
=================
"""

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import DecimalType, ArrayType, StringType
from distributed_trajectories_mpn.udfs import matrix_updates

class TM:

    def __init__(self, df, voronoi):
        self.df = df
        self.voronoi = voronoi

    def tm_states_vector(self):
        window = Window.partitionBy(['user_id', F.to_date('timestamp')]).orderBy('timestamp')
        self.df = self.df.withColumn('states_lag', F.lag('states').over(window)).dropna()
        return self.df

    def sim_tm_states_vector(self):
        window = Window.partitionBy(F.col('user_id')).orderBy('i')
        self.df = self.df.withColumn('states_lag', F.lag('states').over(window)).dropna()
        return self.df

    def tm_states_update(self):
        updates_udf = F.udf(matrix_updates, ArrayType(ArrayType(StringType())))
        self.df = self.df.withColumn('updates', updates_udf('states_lag', 'states'))
        return self.df

    def states_collect(self):
        self.df = self.df.select(['updates'])\
                    .withColumn('updates', F.explode('updates'))\
                    .withColumn('y', F.col('updates').getItem(0).cast(DecimalType(16, 0)))\
                    .withColumn('x', F.col('updates').getItem(1).cast(DecimalType(16, 0)))\
                    .withColumn('val', F.col('updates').getItem(2))\
                    .drop('updates')\
                    .groupBy(['y', 'x']).agg(F.sum('val').alias('updates'))
        return self.df

    def coords_simplify(self):
        self.df = self.df.join(F.broadcast(self.voronoi.select('voronoi_id', 'simplified_id')), 
                                           self.df.y == self.voronoi.voronoi_id, 'inner')\
                         .withColumnRenamed('simplified_id', 'simple_y')\
                         .withColumnRenamed('voronoi_id', 'voronoi_id_y')\
                         .join(F.broadcast(self.voronoi.select('voronoi_id', 'simplified_id')), 
                                           self.df.x == self.voronoi.voronoi_id, 'inner')\
                         .withColumnRenamed('simplified_id', 'simple_x')\
                         .withColumnRenamed('voronoi_id', 'voronoi_id_x')\
                         .drop('y', 'x', 'voronoi_id_y', 'voronoi_id_x')\
                         .withColumnRenamed('simple_y', 'y')\
                         .withColumnRenamed('simple_x', 'x')
        return self.df

    def states_normalize(self):
        window = Window.partitionBy(F.col('y'))
        self.df = self.df.withColumn('updates', F.col('updates')/F.sum(F.col('updates')).over(window))
        return self.df

    def prepare_for_GIS(self):
        self.df = self.df.join(F.broadcast(self.voronoi), self.df.y == self.voronoi.simplified_id, 'inner')\
                         .withColumnRenamed('voronoi_id', 'locid_Y')\
                         .withColumnRenamed('coord_x', 'xtr_10_Y')\
                         .withColumnRenamed('coord_y', 'ytr_10_Y')\
                         .drop('simplified_id')\
                         .join(F.broadcast(self.voronoi), self.df.x == self.voronoi.simplified_id, 'inner')\
                         .withColumnRenamed('voronoi_id', 'locid_X')\
                         .withColumnRenamed('coord_x', 'xtr_10_X')\
                         .withColumnRenamed('coord_y', 'ytr_10_X')\
                         .withColumnRenamed('updates', 'val')\
                         .drop('simplified_id', 'y', 'x')
        return self.df

    def make_tm(self):
        self.tm_states_vector()
        self.tm_states_update()
        self.states_collect()
        self.coords_simplify()
        self.states_normalize()
        return self.df

    def make_tm_GIS(self):
        self.tm_states_vector()
        self.tm_states_update()
        self.states_collect()
        self.coords_simplify()
        self.states_normalize()
        self.prepare_for_GIS()
        return self.df

    def make_sim_tm(self):
        self.sim_tm_states_vector()
        self.tm_states_update()
        self.states_collect()
        self.states_normalize()
        return self.df



    

    