"""
Processing Ring Datasets
========================
"""


import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import StringType, FloatType, ArrayType
from distributed_trajectories_mpn.constants import ring_fraction



class Preprocess:

    def __init__(self, spark, path):
        self.spark = spark
        self.path = path
        self.df = self.spark.read.format("csv").load(self.path)

    def read(self):
        w = Window().orderBy(F.lit('A'))
        self.df = self.df.withColumn('row', F.row_number().over(w))\
                         .filter(F.col('row') != 1).drop('row')
        return self.df

    def seperate_adjNum(self):
        self.df = self.spark.createDataFrame(self.df.toPandas().iloc[::2])
        return self.df

    def seperate_adjMem(self):
        self.df = self.spark.createDataFrame(self.df.toPandas().iloc[1::2])
        return self.df

    def adjNum(self):
        self.read()
        self.seperate_adjNum()
        return self.df

    def adjMem(self):
        self.read()
        self.seperate_adjMem()
        return self.df





class Ring_adjNum:
    
    def __init__(self, first, second, third):
        self.first = first
        self.second = second
        self.third = third

    def join(self):
        self.df = self.first.withColumn('row', F.monotonically_increasing_id())\
                .join(self.second.withColumn('row', F.monotonically_increasing_id()), ['row'])\
                .join(self.third.withColumn('row', F.monotonically_increasing_id()), ['row'])\
                .toDF('row', '1st', '2nd', '3rd')
        return self.df

    def probability(self):    
        self.df = self.df.withColumn('voronoi_id', F.split(self.df['1st'], ' ').getItem(0))\
                    .withColumn('selfProp', F.lit(ring_fraction[0]))\
                    .withColumn('1st_adjNum', F.split(self.df['1st'], ' ').getItem(1).cast(FloatType()))\
                    .withColumn('1st_adjProp', ring_fraction[1]/F.col('1st_adjNum'))\
                    .withColumn('2nd_adjNum', F.split(self.df['2nd'], ' ').getItem(1).cast(FloatType()))\
                    .withColumn('2nd_adjProp', ring_fraction[2]/F.col('2nd_adjNum'))\
                    .withColumn('3rd_adjNum', F.split(self.df['3rd'], ' ').getItem(1).cast(FloatType()))\
                    .withColumn('3rd_adjProp', ring_fraction[3]/F.col('3rd_adjNum'))\
                    .drop('1st', '2nd', '3rd')
        return self.df

    def get_adjNum(self):
        self.join()
        self.probability()
        return self.df






class Ring_adjMem:

    def __init__(self, first, second, third):
        self.first = first
        self.second = second
        self.third = third

    def process(self):
        self.df = self.first.withColumn('row', F.monotonically_increasing_id())\
                 .join(self.second.withColumn('row', F.monotonically_increasing_id()), ['row'])\
                 .join(self.third.withColumn('row', F.monotonically_increasing_id()), ['row'])\
                 .toDF('row', '1st_adj', '2nd_adj', '3rd_adj')
        return self.df

    def get_adjMem(self):
        self.process()
        return self.df







class Ring:

    def __init__(self, num, mem):
        self.num = num
        self.mem = mem

    def join(self):
        self.df = self.num.join(self.mem, ['row'], how = 'inner')
        return self.df

    def concat(self):
        self.df = self.df\
           .withColumn('temp1', F.concat(F.lit(' '), '1st_adjProp'))\
           .withColumn('temp2', F.concat(F.lit(' '), '2nd_adjProp'))\
           .withColumn('temp3', F.concat(F.lit(' '), '3rd_adjProp'))\
           .drop('1st_adjProp', '2nd_adjProp', '3rd_adjProp')
        return self.df

    def repeat(self):
        self.df = self.df\
           .withColumn("1st_adjProp", F.expr("repeat(temp1, 1st_adjNum)"))\
           .withColumn("2nd_adjProp", F.expr("repeat(temp2, 2nd_adjNum)"))\
           .withColumn("3rd_adjProp", F.expr("repeat(temp3, 3rd_adjNum)"))\
           .drop('1st_adjNum', '2nd_adjNum', '3rd_adjNum', 'temp1', 'temp2', 'temp3')
        return self.df
    
    def select_1ring(self):
        self.df = self.df\
            .select('row', 'voronoi_id', 'selfProp', '1st_adj', '1st_adjProp')
        return self.df
    
    def select_2ring(self):
        self.df = self.df\
            .select('row', 'voronoi_id', 'selfProp', '1st_adj', '1st_adjProp', '2nd_adj', '2nd_adjProp')
        return self.df

    def neigh_prop_1ring(self):
        self.df = self.df\
           .withColumn('neighbors', F.concat('voronoi_id', F.lit(' '), '1st_adj'))\
           .withColumn('props', F.concat('selfProp', '1st_adjProp'))\
           .withColumn('neighbors', F.split(F.col('neighbors'), ' ').cast(ArrayType(StringType())))\
           .withColumn('props', F.split(F.col('props'), ' ').cast(ArrayType(FloatType())))\
           .withColumn('props', F.expr("transform(props, x -> x / 0.74)"))\
           .drop('row', 'selfProp', '1st_adjProp', '1st_adj')
        return self.df

    def neigh_prop_2ring(self):
        self.df = self.df\
           .withColumn('neighbors', F.concat('voronoi_id', F.lit(' '), '1st_adj', F.lit(' '), '2nd_adj'))\
           .withColumn('props', F.concat('selfProp', '1st_adjProp', '2nd_adjProp'))\
           .withColumn('neighbors', F.split(F.col('neighbors'), ' ').cast(ArrayType(StringType())))\
           .withColumn('props', F.split(F.col('props'), ' ').cast(ArrayType(FloatType())))\
           .withColumn('props', F.expr("transform(props, x -> x / 0.89)"))\
           .drop('row', 'selfProp', '1st_adjProp', '2nd_adjProp', '1st_adj', '2nd_adj')
        return self.df

    def neigh_prop_3ring(self):
        self.df = self.df\
           .withColumn('neighbors', F.concat('voronoi_id', F.lit(' '), '1st_adj', F.lit(' '), '2nd_adj', F.lit(' '), '3rd_adj'))\
           .withColumn('props', F.concat('selfProp', '1st_adjProp', '2nd_adjProp', '3rd_adjProp'))\
           .withColumn('neighbors', F.split(F.col('neighbors'), ' ').cast(ArrayType(StringType())))\
           .withColumn('props', F.split(F.col('props'), ' ').cast(ArrayType(FloatType())))\
           .withColumn('props', F.expr("transform(props, x -> x / 0.95)"))\
           .drop('row', 'selfProp', '1st_adjProp', '2nd_adjProp', '3rd_adjProp', '1st_adj', '2nd_adj', '3rd_adj')
        return self.df

    def states(self):
        self.df = self.df\
            .withColumn('states', F.arrays_zip('neighbors', 'props'))
        return self.df

    def get_1ring_data(self):
        self.join()
        self.concat()
        self.repeat()
        self.select_1ring()
        self.neigh_prop_1ring()
        self.states()
        return self.df

    def get_2ring_data(self):
        self.join()
        self.concat()
        self.repeat()
        self.select_2ring()
        self.neigh_prop_2ring()
        self.states()
        return self.df

    def get_3ring_data(self):
        self.join()
        self.concat()
        self.repeat()
        self.neigh_prop_3ring()
        self.states()
        return self.df






def get_oneRingData(spark, firstRing_file, secondRing_file, thirdRing_file):

    firstRing_adjNum = Preprocess(spark, firstRing_file).adjNum()
    secondRing_adjNum = Preprocess(spark, secondRing_file).adjNum()
    thirdRing_adjNum = Preprocess(spark, thirdRing_file).adjNum()

    firstRing_adjMem = Preprocess(spark, firstRing_file).adjMem()
    secondRing_adjMem = Preprocess(spark, secondRing_file).adjMem()
    thirdRing_adjMem = Preprocess(spark, thirdRing_file).adjMem()

    adjNum_data = Ring_adjNum(firstRing_adjNum, secondRing_adjNum, thirdRing_adjNum).get_adjNum()
    adjMem_data = Ring_adjMem(firstRing_adjMem, secondRing_adjMem, thirdRing_adjMem).get_adjMem()
     
    oneRing_data = Ring(adjNum_data, adjMem_data).get_1ring_data()

    return oneRing_data


def get_twoRingData(spark, firstRing_file, secondRing_file, thirdRing_file):

    firstRing_adjNum = Preprocess(spark, firstRing_file).adjNum()
    secondRing_adjNum = Preprocess(spark, secondRing_file).adjNum()
    thirdRing_adjNum = Preprocess(spark, thirdRing_file).adjNum()

    firstRing_adjMem = Preprocess(spark, firstRing_file).adjMem()
    secondRing_adjMem = Preprocess(spark, secondRing_file).adjMem()
    thirdRing_adjMem = Preprocess(spark, thirdRing_file).adjMem()

    adjNum_data = Ring_adjNum(firstRing_adjNum, secondRing_adjNum, thirdRing_adjNum).get_adjNum()
    adjMem_data = Ring_adjMem(firstRing_adjMem, secondRing_adjMem, thirdRing_adjMem).get_adjMem()
     
    twoRing_data = Ring(adjNum_data, adjMem_data).get_2ring_data()

    return twoRing_data


def get_threeRingData(spark, firstRing_file, secondRing_file, thirdRing_file):

    firstRing_adjNum = Preprocess(spark, firstRing_file).adjNum()
    secondRing_adjNum = Preprocess(spark, secondRing_file).adjNum()
    thirdRing_adjNum = Preprocess(spark, thirdRing_file).adjNum()

    firstRing_adjMem = Preprocess(spark, firstRing_file).adjMem()
    secondRing_adjMem = Preprocess(spark, secondRing_file).adjMem()
    thirdRing_adjMem = Preprocess(spark, thirdRing_file).adjMem()

    adjNum_data = Ring_adjNum(firstRing_adjNum, secondRing_adjNum, thirdRing_adjNum).get_adjNum()
    adjMem_data = Ring_adjMem(firstRing_adjMem, secondRing_adjMem, thirdRing_adjMem).get_adjMem()
     
    threeRing_data = Ring(adjNum_data, adjMem_data).get_3ring_data()

    return threeRing_data