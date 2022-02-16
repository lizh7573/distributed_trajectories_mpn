"""
Processing MPN Dataset
======================
"""

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
# from cdr_trajectories.constants import Spark


class MPN:

    def __init__(self, spark, path):
        self.spark = spark
        self.path = path
        self.df = self.spark.read.format("csv")\
            .option("inferSchema", "true")\
            .option("header", "true")\
            .option("sep", ";")\
            .load(self.path)

    def process(self):
        self.df = self.df\
             .withColumnRenamed('uid', 'user_id') \
             .withColumnRenamed('usage_dttm', 'timestamp') \
             .withColumn('weekday', ((F.dayofweek('timestamp')+5)%7)+1)\
             .withColumnRenamed('tidenhnum', 'hour')\
             .withColumnRenamed('xtr10', 'coord_x')\
             .withColumnRenamed('ytr10', 'coord_y')\
             .select('user_id', 'timestamp', 'weekday', 'hour', 'coord_x', 'coord_y') \
             .dropDuplicates(['user_id', 'timestamp']) \
             .orderBy(['user_id', 'timestamp'])
        return self.df