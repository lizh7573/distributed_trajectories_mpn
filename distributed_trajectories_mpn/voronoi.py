"""
Processing Voronoi Dataset
==========================
"""

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import StringType



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
            .withColumn('voronoi_id', F.col('voronoi_id').cast(StringType()))\
            .withColumnRenamed('xtr10', 'coord_x')\
            .withColumnRenamed('ytr10', 'coord_y')\
            .withColumn('simplified_id', F.row_number().over(Window().orderBy(F.lit('A'))))\
            .select('voronoi_id', 'coord_x', 'coord_y', 'simplified_id')
        return self.df
