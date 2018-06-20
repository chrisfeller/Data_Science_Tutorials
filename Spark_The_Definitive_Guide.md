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
