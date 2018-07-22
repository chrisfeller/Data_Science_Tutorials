### Spark: The Definitive Guide
#### Bill Chambers and Matei Zaharia
#### June 2018

---
#### Preface
**Github Repo**
* textbook repo: https://github.com/databricks/Spark-The-Definitive-Guide

---
#### Chapter 1 | What is Apache Spark?
**Introduction**
* Apache Spark is a unified computing engine and a set of libraries for parallel data processing on computer clusters.
* Spark supports SQL, Python, Scala, R, and Java

![Spark Components](https://www.safaribooksonline.com/library/view/spark-the-definitive/9781491912201/assets/spdg_0101.png)

**Apache Spark's Philosophy**
1) Unified
    - Offer a unified platform for writing big data applications.
    - Consistent API for wide range of data analytics tasks.
2) Computer Engine
    - Spark handles loading data from storage systems and performing computation on it, not permanent storage as the end itself.
    - Spark does not store data long term
3) Libraries
    - Unified API for common data analysis tasks
        - SparkSQL: SQL and structured data
        - MLlib: Machine Learning
        - Spark Streaming: Streaming data
        - GraphX: Graph analysis

**History of Spark**
* Research project at UC Berkley in 2009
* Alternative to MapReduce
    - In-memory instead of on disk
* Spurred creation of Databricks

#### Chapter 2 | A Gentle Introduction to Spark
**Spark's Basic Architecture**
* A cluster is a group of computers that pools the resources of many machines together, giving the user the ability to use all of the cumulative resources as if they were a single computer.
* Spark is a framework to coordinate work across a cluster.
* The cluster of machines that Spark will use to execute tasks is managed by a cluster manager like Spark's standalone cluster manager, YARN.
    * We submit Spark applications to these cluster managers, which will grant resources to our application so that we can complete our work.

**Spark Applications**
* Spark applications consist of a driver process and a set of executor processes.
    * The driver process is responsible for three things:
        1. Maintaining information about the spark application
        2. Responding to a user's program or input
        3. Analyzing, distributing, and scheduling work across the executors
    * The executors are responsible for actually carrying out the work that the driver assigns them.

![Spark Architecture](https://izhangzhihao.github.io/assets/images/spark-01.png)

**Major Points**
* Spark employs a cluster manager that keeps track of the resources available.
* The driver process is responsible for executing the driver program's commands across the executors to complete a given task.

**Spark's Language APIs**
* Spark is primarily written in Scala, making it Spark's 'default' language.
* When using Spark from Python or R, you don't write explicit JVM instructions; instead, you write Python and R code that Spark translates into code that it then can run on the executor JVMs.

**The SparkSession**
* You control your Spark Application through a driver process called the SparkSession.
* The SparkSession instance is the way Spark executes user-defined manipulations across the cluster.

**DataFrames**
* A DataFrame is the most common Structured API and simply represents a table of data with rows and columns.
    - The list that defines the columns and the types within those columns is called the schema.

**Partitions**
* To allow every executor to perform work in parallel, Spark breaks up the data into chunks called partitions.
    - A partition is a collection of rows that sit on one physical machine in your cluster.

**Transformations**
* In Spark, the core data structures are immutable.
* To change a DataFrame you must perform a transformation.
* Transformations are the core of how you express your business logic using Spark.
* There are two types of transformations:
    1. Those that specify narrow dependencies
        - Those for which each input partition will contribute to only one output partition.
        - Performed in memory.
    2. Those that specify wide dependencies
        - Those that will have input partitions contributing to many output partitions.
        - AKA: shuffle (where Spark will exchange partitions across the cluster).
        - Not performed in memory (writes to disk).

**Lazy Evaluation**
* Spark will wait until the very last moment to execute the graph of computation instructions.

**Actions**
* An action instructs Spark to compute a result from a series of transformations.
* Examples of actions:
    - Actions to view data in the console.
    - Actions to collect data to native objects in their respective language.
    - Actions to write to output data sources

**Spark UI**
* If you are running in local mode, you can monitor the progress of a job via http://localhost:4040

**Data Input Example**
* To start spark locally:
```
$ spark-shell
```
* To start pyspark:
```
pyspark
```
* Data input example:
```
flightData2015 = spark.read.option("inferSchema", "true").option("header", "true").csv("/Users/chrisfeller/Desktop/Spark-The-Definitive-Guide/data/flight-data/csv/2015-summary.csv")
```
* To view the first few records:
```
flightData2015.take(3)
```
* To sort by the 'count' field:
```
flightData2015.sort('count').explain()
```
```
flightData2015.sort('count').take(3)
```

**DataFrames and SQL**
* With SparkSQL you can register any DataFrame as a table or view (a temporary table) and query it using pure SQL.
* There is no performance difference between writing SQL queries or writing DataFrame code, they both 'compile' to the same underlying plan that we specify in DataFrame code.
* To make any DataFrame into a table or view:
```
flightData2015.createOrReplaceTempView('flightData2015')
```
* You can then query that table view SQL syntax:
```
sqlWay = spark.sql("""
    SELECT DEST_COUNTRY_NAME, count(1)
    FROM flightData2015
    GROUP BY DEST_COUNTRY_NAME
    """)
```
* You could do the same view python:
```
dataFrameWay = flightData2015.groupBy('DEST_COUNTRY_NAME').count()
```
* Similar comparison between query in SQL and python:
```
spark.sql("SELECT max(count) FROM flightData2015").take(1)
```
```
flightData2015.select(max('count')).take(1)
```
* The execution plan is a directed acyclic graph (DAG) of transformations, each resulting in a new immutable DataFrame, on which we can call an action to generate a result.

#### Chapter 3 | A Tour of Spark's Toolset
**Datasets: Type-Safe Structured APIs**
* Not available in Python and R (only Java and Scala) because those languages are dynamically typed.

**Structured Streaming**
* Structured Streaming is a high-level API for stream processing that became production-ready in Spark 2.2.
* With Structured Streaming, you can take the same operations that you perform in batch mode using Spark's structured APIs and run them in a streaming fashion.

**Machine Learning and Advanced Analytics**
* Spark is able to perform large-scale machine learning with it's built-in library of machine learning algorithms called MLlib.
* MLlib allows for preprocessing, munging, training, and making predictions at scale on data.

**Lower-Level APIs**
* Virtually everything in Spark is built on top of Resilient Distributed Datasets (RDDs).
* There are basically no instances in modern Spark, for which you should be using RDDs instead of structured APIs beyond manipulating some very raw unprocessed and unstructured data.

#### Chapter 4: Structured API Overview
**DataFrames and Datasets**
* DataFrames and Datasets are (distributed) table-like collections with well-defined rows and columns.
* Tables and views are basically the same thing as DataFrames. We just execute SQL against them instead of DataFrame code.

**Schemas**
* A schema defines the column names and types of a DataFrame.

**DataFrames vs. Datasets**
* Datasets are only available to Java Virtual Machine (JVM)-based languages (Scala and Java).

**Spark Types**
* Import Types:
```
from pyspark.sql.types import *
```

| Data Type  | Value type in Python  | API to access or create a data type  |
|---|---|---|
| ByteType  | int or long  | `ByteType()`  |
| ShortType  | int or long  | `ShortType()`  |
| IntegerType  | int or long  | `IntegerType()`  |
| LongType  | long  | `LongType()`  |
| FloatType  | float  | `FloatType()`  |
| DoubleType  | float  | `DoubleType()`  |
| DecimalType  | decimal.Decimal  | `DecimalType()`  |
| StringType  | string  | `StringType()`  |
| BinaryType  | bytearray  | `BinaryType()`  |
| BooleanType  | bool  | `BooleanType()`  |
| TimestampType  | datetime.datetime  | `TimestampType()`  |
| DateType  | datetime.date  | `DateType()`  |
| ArrayType | list, tuple, or array | `ArrayType(elementType)` |
| MapType | dict | `MapType(keyType, valueType)` |
| StructType | list or tuple | `StructType(fields)` |
| StructField | The value type in Python of the data type of this field | `StructField(name, dataType)`|

**Overview of Structured API Execution**
1. Write DataFrame/Dataset/SQL Code
2. If valid code, Spark converts this to a Logical Plan
3. Spark transforms this Logical Plan to a Physical Plan, checking for optimizations along the way
4. Spark then executes this Physical Plan (RDD manipulations) on the cluster.

#### Chapter 5 | Basic Structured Operations
**Schemas**
* A schema defines the column names and types of a DataFrame.
* To print the schema of a DataFrame:
```
df.printSchema()
```
* When using Spark for production Extract, Transform, Load (ETL), it is often a good idea to define your schemas manually, especially when working with untyped data sources like CSV and JSON.
* An example of how to enforce a specific schema on a DataFrame:
```
from pyspark.sql.types import StructField, StructType, StringType, LongType

myManualSchema = StructType([
    StructField('DEST_COUNTRY_NAME', StringType(), True),
    StructField('ORIGIN_COUNTRY_NAME', StringType(), True),
    StructField('count', LongType(), False, metadata={'hello':'world'})
    ])

df = spark.read.format('json').schema(myManualSchema).load('data/flight-data/json/2015-summary.json')
```

**Columns**
* To refer to a specific DataFrame's column:
```
from pyspark.sql.functions import col
df.col('column_name')
```
* To access a DataFrame's columns:
```
df.columns
```

**select and selectExpr**
* `select` and `selectExpr` allow you to do the DataFrame equivalent of SQL queries on a table of data.
* In the simplest possible terms, you can use them to manipulate columns in your DataFrames.
* To select a column:
```
df.select('col_name').show()
```
* To select multiple columns:
```
df.select('col_name1', 'col_name2').show()
```
* To select a column and rename it:
```
df.select(expr('col_name AS col_rename')).show()

# OR

df.select('col_names').alias('col_rename').show()

# OR

df.selectExpr('col_name AS col_rename').show()
```
* We can treat `selectExpr` as a simple way to build up complex expressions that create new DataFrames similar to SQL:
```
df.selectExpr('avg(col_name)', 'count(distinct(col_name))'').show()
```

**Adding Columns**
* To add a column:
```
df.withColumn('new_column', 'old_column' * 2).show()
```
* `.withColumn()` takes two arguments: the column name and the expression that will create the value for that given row in the DataFrame.
* You can also rename a column this way.

**Renaming Columns**
* To rename a column:
```
df.withColumnRenamed('new_name', 'old_name')
```

**Case Sensitivity**
* By default, Spark is case insensitive

**Removing Columns**
* To remove a column:
```
df.drop('col_name')
```
* To drop multiple columns:
```
df.drop('col_name1', 'col_name2')
```

**Changing a Column's Type (Cast)**
* To change a column from string to int:
```
df.withColumn('col_name', col('col_name').cast('int'))
```

**Filtering Rows**
* `.filter()` and `.where()` each work for filtering rows
* Example:
```
df.filter(col('col_name') < 2)
df.where('col_name < 2')
```
* To perform multiple filters, chain them together:
```
df.where(col('col_name') < 2).where(col('col_name') > 5)
```

**Getting Unique Rows**
* To get the unique values in a row:
```
df.select('col_name').distinct()
```

**Random Samples**
* To randomly sample some records from your DataFrame:
```
withReplacement = True
fraction=0.5
seed=42
df.sample(withReplacement, fraction, seed).show()
```

**Concatenating and Appending Rows (Union)**
* To append to a DataFrame, you must union the original DataFrame along with the new DataFrame
    - This just concatenates the two DataFrames
* To union two DataFrames, you must be sure that they have the same schema and number of columns; otherwise, the union will fail.
    - Unions are currently performed based on location, not on schema. This means that columns will not automatically line up the way you think they might.
```
df.union(newdf)
```

**Sorting Rows**
* Can be done with either `.sort()` or `.orderBy()`
* They accept both column expressions and strings as well as multiple columns.
* The default is to sort in ascending order
```
df.sort('col_name')
df.orderBy('col_name1', 'col_name2')

# OR
df.sort(col('col_name'))
df.orderBy(col('col_name1'), col('col_name2'))
```
* To specify sort direction:
```
from pyspark.sql.functions import desc, asc
df.orderBy('col_name').desc()
```

**Limit**
* To restrict what you extract from a DataFrame use `.limit()`:
```
df.limit(10)
```
* Similar to `.head()` in pandas.

**Repartition and Coalesce**
* To get the number of partitions:
```
df.rdd.getNumPartitions()
```
* You should typically only repartition when the future number of partitions is greater than your current number of partitions or when you are looking to partition by a set of columns
* To repartition:
```
df.repartition(5)
```
* To repartition by a certain column:
```
df.repartition(col('col_name'))
```

**Collecting Rows to the Driver**
* `.collect()` gets all data from the entire DataFrame, `.take()` selects the first *N* rows, and `.show()` prints out a number of rows nicely.
* Any collection of data to the driver can be a very expensive operation! If you have a large dataset and call `.collect()` you can crash the driver.

#### Chapter 6 | Working with Different Types of Data
