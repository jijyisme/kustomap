
# coding: utf-8

# In[ ]:

#1 - import module
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler,MaxAbsScaler


# In[ ]:

#2 - Create SparkContext
#sc = SparkContext()
print sc
print ""
print sc.getConf().toDebugString()


# In[ ]:

#3 - Setup SparkSession(SparkSQL)
spark = (SparkSession
         .builder
         .appName("Kmean_(crime_homicide_subset)")
         .config("spark.sql.warehouse.dir", "/opt/jupyter_workspace/spark-warehouse")
         .getOrCreate())
print spark


# In[ ]:

#4 - Read file to spark DataFrame
# workspace = '/opt/jupyter_workspace/data/' #use this line when run spark with out hdfs
# model_dir = '/opt/jupyter_workspace/model' #use this line when run spark with out hdfs
workspace = 'hdfs://cdh-cluster-02:8020/user/training/data/' #use this line when run spark with hdfs
model_dir = 'hdfs://cdh-cluster-02:8020/user/training/model/' #use this line when run spark with hdfs

data = (spark
        .read
        .option("header","true")
        .option("inferSchema", "true")
        .csv(workspace + "crime_homicide_subset.csv"))
# If the path don't have file:/// -> it will call hdfs instead of local file system
data.cache()
print "finish caching data"


# In[ ]:

#5 - Print sample 5 rows of all variables
data.show(5)

column_name = ['latitude','longtitude']


# In[ ]:

#6 - Create Vector
assem =  VectorAssembler(inputCols=column_name ,outputCol="temp_features")

print assem


# In[ ]:

#7 - Normalize
scaler = MaxAbsScaler(inputCol="temp_features", outputCol="features")

print scaler


# In[ ]:

#8 - Create model
kmeans = KMeans().setSeed(50)


# In[ ]:

#9 - Set ML pipeline
all_process_list = [assem,scaler,kmeans]
for process in all_process_list: print process

pipeline = Pipeline(stages=all_process_list)
print pipeline


# In[ ]:

#10 - Train model
model = pipeline.fit(data)


# In[ ]:

#11 - Make predictions
predictions = model.transform(data).select("features","prediction")
predictions.show(truncate=False)


# In[ ]:

#12 Evaluate clustering by computing Within Set Sum of Squared Errors.
kmean_model = model.stages[-1]
wssse = kmean_model.computeCost(predictions)
print("Within Set Sum of Squared Errors = " + str(wssse))


# In[ ]:

#13 Shows Cluster's Center
centers = kmean_model.clusterCenters()
scaler_model = model.stages[-2]
max = scaler_model.maxAbs
print("Cluster Centers: ")
for center in centers:
    print(center*max)


# In[ ]:



