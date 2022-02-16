"""
Generating Trajectories
============
"""

import pyspark.sql.functions as F


# Deterministic Trajectories
class DetermTraj:

    def __init__(self, mpn, voronoi):
        self.mpn = mpn
        self.voronoi = voronoi

    def join(self):
        self.df = self.mpn.join(F.broadcast(self.voronoi), ['coord_x', 'coord_y'], how = 'inner')\
                       .orderBy(['user_id', 'timestamp'])
        return self.df
            
    def process(self):
        self.df = self.df.withColumn('neighbors', F.array('voronoi_id'))\
                         .withColumn('props', F.array(F.lit(1.0)))\
                         .withColumn('states', F.arrays_zip('neighbors', 'props'))\
                         .orderBy(['user_id', 'timestamp'])\
                         .drop('coord_x', 'coord_y', 'neighbors', 'props')
        return self.df

    def make_traj(self):
        self.join()
        self.process()
        return self.df





# Probabilistic Trajectories
class ProbTraj:

    def __init__(self, df, ring):
        self.df = df
        self.ring = ring

    def make_traj(self):
        self.df = self.df.join(F.broadcast(self.ring), ['voronoi_id'], how = 'inner')\
                         .orderBy(['user_id', 'timestamp'])\
                         .drop('coord_x', 'coord_y', 'neighbors', 'props')
        return self.df