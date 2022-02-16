"""
Processing Voronoi Dataset
==========================
"""

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType



class Voronoi:

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
            .withColumnRenamed('locid', 'voronoi_id')\
            .withColumnRenamed('xtr10', 'coord_x')\
            .withColumnRenamed('ytr10', 'coord_y')\
            .select(['voronoi_id', 'coord_x', 'coord_y'])
        return self.df
