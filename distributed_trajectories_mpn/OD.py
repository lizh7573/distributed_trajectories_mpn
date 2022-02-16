"""
Origin-Destination Matrix
=========================
"""

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import ArrayType, DecimalType, LongType, StringType
from distributed_trajectories_mpn.udfs import matrix_updates
from distributed_trajectories_mpn.constants import OD_time_frame

class OD:

    def __init__(self, df, voronoi):
        self.df = df
        self.voronoi = voronoi

    def od_states_vector(self):
        window = Window.partitionBy(['user_id', F.to_date('timestamp')])\
                       .orderBy('timestamp_long').rangeBetween(-OD_time_frame, OD_time_frame)
        self.df = self.df.withColumn('timestamp_long', F.col('timestamp').cast(LongType()))\
                         .withColumn('last_states', F.last('states').over(window))
        return self.df

    def od_states_update(self):
        updates_udf = F.udf(matrix_updates, ArrayType(ArrayType(StringType())))
        self.df = self.df.withColumn('updates', updates_udf('states', 'last_states'))
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

    def make_od(self):
        self.od_states_vector()
        self.od_states_update()
        self.states_collect()
        self.coords_simplify()
        self.states_normalize()
        return self.df

    def make_od_GIS(self):
        self.od_states_vector()
        self.od_states_update()
        self.states_collect()
        self.coords_simplify()
        self.states_normalize()
        self.prepare_for_GIS()
        return self.df