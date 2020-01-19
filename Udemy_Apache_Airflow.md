### The Complete Hands-On Course to Master Apache Airflow
#### Marc Lamberti
#### January 2020

---
#### Introduction
**Course Link**
* Main course homepage: https://www.udemy.com/course/the-complete-hands-on-course-to-master-apache-airflow/

**Environment**
1. Start VirtualBox LeanAirflow environment
2. ssh into virtualbox in terminal via terminal `ssh -p 2200 airflow@localhost`
    - password: airflow
3. Once inside the environment run: `sudo -E ./start.sh` to start hive, spark, etc.
    - password: airflow
4. To make sure everything is running, run `hive` and wait for `hive>` prompt

**Accounts**

| Role/App  | Login  | Password  |
|---|---|---|
| root  | root  | centos  |
| user  | airflow  | airflow  |
| mysql | root | mysql |
| mysql | hive | hive  |
|rabbitmq | admin | rabbitmq|

**Important URLs**
 | App | URL|
 | --- | --- |
 | Spark master | `http://localhost:8081` |
 | Kibana | `http://localhost:5601` |
 | Airflow | `http://localhost:8080` |
 | RabbitMQ | `http://localhost:15672` |


#### First Approach to Airflow
**What is Airflow?**
* Apache airflow is a way to programmatically author, schedule, and monitor data pipelines.
* Core components of Airflow:
    - Web Server
    - Scheduler
    - Metadata Database
    - Executor
    - Worker

**Key Concepts**
* DAG: A graph object representing your data pipeline
* Operator: Describe a single task in your data pipeline
* Task: An instance of an operator
* TaskInstance: Represents a specific run of a task = DAG + Task + Point in Time
* Workflow: Combination of all above

**What Airflow Provides**
* Pipelines developed in Python
* Graphical representation of your DAGs
* Airflow is scalable
* Catchup and Backfill
* Very extensible

**What Airflow is not**
* Airflow is not a data streaming solution
* Airflow is not in the scope of Apache Spark or Storm
* Primarily build to perform scheduled batch jobs

**How Airflow Works**
1. The Scheduler reads the DAG folder
2. Your DAG is parsed by a process to create a DagRun based on the scheduling parameters of your DAG.
3. A TaskInstance is instantiated for each Task that needs to be executed and flagged to 'Scheduled' in the metadata database
4. The Scheduler gets all TaskInstances flagged 'Scheduled' from the metadata database, changes the state to 'Queued' and sends them to the executors to be executed.
5. Executors pull our Tasks from the queue (depending on your execution setup), change the state from 'Queued' to 'Running' and Workers start executing the TaskInstances.
6. When a Task is finished, the Executor changes the state of that task to its final state (success, failed, etc.) in the database and DAGRun is updated by the Scheduler with the state 'Success' or 'Failed.' Of course, the web server periodically fetches data from the metadata database to update the UI.

**Installing Airflow**
1. Move into python virtual environment:
    `source .sandbox/bin/activate`
2. Install airflow and dependency libraries:
    `pip install "apache-airflow[celery, crypto, postgres, rabbitmq, redis]"==1.10.6`
3. Initiate airflow database:
    `airflow initdb`
4. Create folder name dags in `/airflow`
    `mkdir dags`
5. Open airflow configuration file:
    `vim airflow.cfg`
6. Change the setting `load_examples` from True to False:
    `:wq` to save and exit vim
7. Initiate airflow databse again:
    `airflow initdb`
8. Copy `airflow_files_hello_world.py` to dags folder:
    `cp ../airflow_files/hello_world.py dags/`
9. Open a new terminal and connect to virtual machine again (see Environment section).
10. Activate python environment in second terminal session:
    `source .sandbox/bin/activate`
11. Run airflow scheduler in second terminal session:
    `airflow scheduler`
12. Run airflow webserver in first terminal session:
    `airflow webserver`
13. Go to your web browser to view airflow UI:
    `localhost:8080`

**Connecting to Airflow**
1. Open two terminal sessions.
2. Start the virtual environment in both and activate the python environment.
3. Start the airflow webserver from the first terminal session.
4. Start the airflow scheduler from the second terminal session
5. Open airflow UI in web browser

**Airflow UI**
* Keep everything in UTC instead of local timezone for dags.
* To stop the airflow webserver an scheduler hit `ctrl+c` in each respective terminal session.

**Airflow CLI**
* To access sqlite tables:
    `sqlite3 airflow.db`
* To view tables in `airflow.db`:
    `.tables`
* Exit sqlite by hitting: `ctrl+d`
* To reset airflow databse:
    `airflow resetdb`
* To list dags:
    `airflow list_dags`
* For command help:
    `airflow -h`

**Quiz**
1. What is Airflow?
    A: An orchestrator to deal with batch data pipelines programmatically by authoring, scheduling and monitoring them through a web UI.
2. What are the core components of Apache Airflow?
    A: A web server, a scheduler, a worker, an executor, and a metadata database.
3. What is a DAG?
    A: A DAG is a graph object representing your data pipeline composed of different tasks with their dependencies.
4. What language is used to code your DAGs with Apache Airflow?
    A: Python
5. What is the first command to do once you've installed Apache Airflow to initialize the metadatabase?
    A: `airflow initdb`
6. Which component is responsible for creating DagRuns?
    A: The Scheduler

**Recap**
* Apache Airflow is a platform to programmatically author, schedule, and monitor workflows.
* 5 core components:   
    - Web server
    - Scheduler
    - Metadata database
    - Worker
    - Executor
* Technically, there is a sixth when in distributed mode:
    - Queue
* 4 Key Concepts:
    - DAG: graph of operators and dependencies
    - Operator: define your tasks, what they do
    - Task: instance of an operator
    - Task Instance: Specific run of task: DAG + TASK + POINT IN TIME
    - Workflow: Mix of all of above
* Your dag files should be in `AIRFLOW_HOME/dags/folder`
* Turn on the toggle next to the DAG name you want to schedule
* Graph View, Tree View, and Gantt view are really useful for exploring your DAG's behavior
* Airflow stores all its metadata in UTC format. The UI displays dates in UTC as well. This is useful as you want to schedule your DAGs independent of timezone.
* In distributed mode, it is recommended to have the scheduler and the webserver running on the same node.

#### Coding A Data Pipeline with AirFlow
**What is a DAG**
* Definition: In mathematics and computer science, a Directed Acyclic Graph (DAG), is a finite directed graph with no directed cycles. That is, it consists of finitely many vertices and edges, with each edge directed from one vertex to another, such that there is no way to start at any vertex *v* and follow a consistently-directed sequence of edges that eventually loops back to *v* again.
    - Equivalently, a DAG is a directed graph that has a topological ordering, a sequence of the vertices such that every edge is directed from earlier to later in the sequence.

**DAG Concise Definition**
* A DAG (Directed Acyclic Graph) is a finite directed graph that doesn't have any cycles (loops).
    - A cycle os a series of vertices that connect back to each other making a loop.
* In Apache Airflow, a DAG represents a collection of tasks to run, organized in a way that represent their dependencies and relationships.
    - Its job is to make sure that tasks happen at the right time and in the right order with the right handling of any unexpected issues.
    - It ultimately defines our Workflow.
* Each node is a task.
* Each edge is a dependency.

**Important Properties**
* DAGs are defined in Python files placed into Airflow's `DAG_FOLDER` (usually `~/airflow/dags`)
* `dag_id` serve as a unique identifier for your DAG
* `description` is the description of your DAG
* `start_date` describes when your DAG should start
* `schedule_interval` defines how oftern your DAG runs
* `depend_on_past` run the next DAGRun if the previous one completed successfully
* `default_args` a dictionary of variables to be used as constructor keyword parameter when initializing operators.

**Code First DAG**
* The DAG:
    - The purpose of this DAG is to show you a typical ETL process which can be automated using Apache Airflow and Twitter. For simplicity, the data are not being fetched from the twitter API but instead an existing database.
    - It is composed of four tasks:
        1. Fetching tweets
        2. Cleaning tweets
        3. Uploading tweets to HDFS
        4. Loading data into HIVE
1. Open up three terminals. The first for the scheduler, the second for the webserver, and the third to code up the dag.
2. Start the webserver and scheduler in the firs two terminal sessions.
    `airflow webserver` and `airflow scheduler`
3. In the third terminal session copy the existing `twitter_dag_v_1.py` into `airflow_dags`:
    `cp airflow_files/twitter_dag_v_1.py airflow/dags`
4. Open the `twitter_dag_v_1.py` in an editor to view the dag

**What is an Operator?**
* While DAGs describe how to run a workflow, Operators determine what actually gets done.
* Definition: An operator describes a single task in a workflow. Operators are usually (but not always) atomic, meaning they can stand on their own and don't need to share resources with any other operators. The DAG will make sure that operators run in the correct order; other than those dependencies, operators run independently. In fact they may run on two completely different machines (for scalability).

**Key Points of Operators**
* An operator is a definition of a single task.
* Should be idempotent
    - Meaning your operator should produce the same result regardless of how many times it is run.
* Retries automatically in case of failure
* A Task is created by instantiating an Operator class
* An Operator defines the nature of this Task and how should it be executed
* When an Operator is instantiated, this task becomes a node in your DAG.

**Airflow Provides Many Operators**
* `BashOperator`
    - Executes a bash command
* `PythonOperator`
    - Calls an arbitrary Python function
* `EmailOperator`
    - Sends an email
* `MySqlOperator`, `SqliteOperator`, `PostgreOperator`
    - Executes a SQL command
* You can also make your own Operators

**Types of Operators**
* All Operators inherit from `BaseOperator`
* There are actually 3 types of operators:
    1. Action operators that perform an action (`BashOperator`, `PythonOperator`, `EmailOperator`...)
    2. Transfer operators that move data from one system to another (`PrestoToMysqlOperator`, `SftpOperator`...)
    3. Sensor operators waiting for data to arrive at a defined location

**Transfer Operators**
* Operators that move data from one system to another
* Data will be pulled out from the source, staged on the machine where the executor is running, and then transferred to the target system.
* Don't use these operators if you are dealing with a large amount of data.

**Sensor Operators**
* Sensor operators inherit of `BaseSensorOperator` (`BaseOperator` being the superclass of `BaseSensorOperator`)
* They are useful for monitoring external processes like waiting for files to be uploaded in HDFS or a partition appearing in Hive.
* They are basically long running tasks.
* The Sensor Operator has a poke method called repeatedly until it returns True (it is the method used for monitoring the external process).

**Operator Relationships and Bitshift Composition**
* There are two ways of describing dependencies between operators in Apache Airflow:
    - By using the traditional operator relationships with:
        - `set_upstream()`
        - `set_downstream()`
    - From Apache Airflow 1.8 you can use Python bitshift operators
        - `<<` = `set_upstream()`
        - `>>` = `set_downstream()`
* Example:
    ```
    t1.set_downstream(t2); t2.set_downstream(t3); t3.set_downstream(t4)

    # Can also be written as
    t4.set_upstream(t3); t3.set_upstream(t2); t2.set_upstream(t1)

    # Can also be written as
    t1 >> t2 >> t3 >> t4

    # Can also be writtten as

    t4 << t3 << t2 << t1
    ```

**Scheduler**
* The scheduler's role is to monitor all tasks and DAGs to ensure that everything is executed based on the `start_date` and the `schedule_interval` parameters. There is also an `execution_date` which is the latest time you rDAG has been executed (`last(date) + schedule_interval`)
* the scheduler periodically scans the DAG folder (`airflow/dags`) to inspect tasks and verifies if they can be triggered or not.

**DagRun**
* A DAG consists of Tasks and needs those tasks to run
* When the Scheduler parses a DAG, it automatically creates a DagRun which is an instantiation of a DAG in time according to the `start_date` and the `schedukle_interval`
* When a DagRun is running all tasks inside it will be executed.

**Key Parameters for Instantiating DAGs**
* `start_date`: the first date for which you want to have data produced by the DAG in your database (can be set in the past)
* `end_date`: the date at which your DAG should stop running (usually set to None)
* `retries`: the maximum number of retries before the task fails
* `retry_delay`: the delay between retries
* `schedule_interval`: the interval at which the Scheduler will trigger your DAG

**Schedule Interval**
* The `schedule_interval` parameter is set to indicate at which interval the Scheduler should run your DAG. It preferably receives a CRON expression as a string or a datetime.timedelta object.
* Alternatively, you can also use a cron 'preset' as shown into the following table.

| Preset  | Meaning  | Cron  |
|---|---|---|
| None  | Don't schedule. Manually triggered  |   |
| @once | Schedule once and only once  |   |
| @hourly  | Run once an hour at the beginning of the hour  | `0 * * * *`  |
| @daily | Run once a day at midnight  | `0 0 * * * `  |
| @weekly | Run once a week at midnight on Sunday morning  | `0 0 * * 0`  |
| @monthly | Run once a month at midnight of the first day of the month  | `0 0 1 * *`  |
| @yearly | Run once a year at midnight of January 1  | `0 0 1 1 *`  |

**Important Scheduler Notes**
* If you run a DAG on a `schedule_interval` of one day, the run stamped 2016-01-01 will be triggered soon after 2016-01-01T23:59.
* The Scheduler runs your job one schEduler interval AFTER the `start_date`, at the END of the period.
* The Scheduler triggers tasks soon after the `start_date` + `scheduler_interval` is passed.

**Backfill and Catchup**
* An Airflow DAG with a `start_date` and a `schedule_interval` defines a series of intervals which the Scheduler turns into individual DagRuns to execute.
* Let's assume the `start_date` of your DAG is 2016-01-01T10L00 and you have started the DAG at 2016-01-01T10:30 with the `scheduled_interval` of `*/10****` (AFTER every 10 minutes).
* Apache Airflow will run past DAGs for any interval that has not been run. This concept is called Catchup/Backfill.
* This feature allows you to backfill your DB with data produced from your ETL as if it were run from the past.
* If you want to avoid this behavior, you can set the parameter `catchup=False` into the DAG arguments.

**Final Notes**
* The first DagRun is created based on the minimum `start_date` for the tasks in your DAG
* Subsequent DagRuns are created by the Scheduler based on your DAG's `schedule_interval` sequentially.

**Concept Review**
* DAG: a description of the order in which work should take place
* Operator: a class that acts as a template for carrying out some work
* Task: an instance of an operator
* Task Instance: a specific run of a task characterized as the combination of a dag, a taskm and a point in time.
* By combining DAGs and Operators to create TaskInstances, you can build complex workflows.

**Quiz**
1. What is a DAG?
    A: A collection of all the tasks you want to run, organized in a way that reflects their relationships and dependencies with no cycles.
2. What is the meaning of the `schedule_interval` property for a DAG?
    A: It defines how often a DAG should be run from the `start_date` + `schedule_time`
3. What is an Operator?
    A: An Operator describes a single task in a workflow.
4. What is a Sensor Operator?
    A: It is a long running task waiting for an event to happen. A poke function is called every n seconds to check if the criteria are met.
5. How can you represent a dependency into a DAG? >
    A: Using `set_upstream` and `set_downstream` functions and bitshift operators like `>>` and `<<`
6. Is `task1>>task2>>task3>>task4` equal to `task4<<task3<<task2<<task2`?
    A: Yes
7. Let's assume your DAG has a `start_date` with October 22, 2018 20:00:00 PM UTC and you have started the DAG at 10:30:00 PM UTC with the `schedule_interval` of `*/10****` (After every 10 miuntes). How many DagRuns are going to execute?
    A: 3

**Summary**
* Two ways of describing a dependency between operators in Apache Airflow:
    1. `set_upstream()` / `set_downstream()`
    2. `>>` / `<<`
* The scheduler's role is to monitor all tasks and DAGs to ensure that everything is executed based on the `start_date` and `scheduled_interval` parameters. There is also an `execution_date` which is the latest time your DAG has been executed (last(date) + schedule_interval). When the Scheduler parses a DAG, it automatically creates a DagRun which is an instantiation of a DAG in time according to the `start_date` and `schedule_interval`. The `schedule_interval` parameter is set to indicate at which interval the Scheduler should run your DAG. It receives preferably a CRON expression as a str or a datetime.timedelta object.
* If you run a DAG on a `schedule_interval` of one day, the run stamped 2016-01-01 will be triggered soon after 2016-01-01T23:59.
* The Scheduler runs your job one `schedule_interval` AFTER `start_date`. at the END of the period.
* The Scheduler triggers tasks soon after the `start_date` + `schedule_interval` is passed.
* An Airflow DAG with a `start_date` and a `schedue_interval` defines a series of intervals which the Scheduler turns into individual DagRuns to execute.
* Let's assume the `start_date` of your DAG is 2016-01-01T10:00 and you have started the DAG at 2016-01-01T10:30 with the `schedule_interval` of `*/10****` (AFTER every 10 minutes).
* Apache Airflow will run past DAGs for any interval that has not been run. This concept is called Catchup / Backfill.
* This feature allows you to backfill your DB with data produced from your ETL as if it were run from the past.
* IF you want to avoid this behavior, you can set the parameter `catchup=False` into the DAG arguments.
* Workflows are basically the combination of DAGs, Operators, Tasks, and TaskInstances.

#### Databases and Executors
**Sequential Executor with SQLite**
* Apache Airflow can work in a distributed environment. Tasks are scheduled according to their dependencies defined into a DAG and the Workers pick up and run jobs with their load balanced for performance optimization. All task information is stored into the Metadatabase which is regularly updated.

**SQLite**
* SQLite is the default database used by airflow
* SQLite is a relational database
* SQLite is ACID-compliant (Atomicity, Consistency, Isolation, Durability)
* It implements most of the SQL standard.
* It requires almost no configuration to run.
* It supports an unlimited number of simultaneous readers, but only one writer at an instant in time.
* Limited in size up to 140 TB and the entire database is stored into a single disk file.

**Executors**
* An Executor is fundamentally a message queue process which determines the Worker processes that execute each scheduled task.
* A Worker is a process where the task is executed.

**What is a Sequential Executor**
* It is the most basic executor to use.
* This executor only runs one task at a time (Sequential), useful for debugging.
* It is the only executor hat can be used with SQLite since SQLite doesn't support multiple writers.
* It is the default executor you get when you run Apache Airflow for the first time.

**The Configuration File**
* The configuration file is where you would update the type of executor you would like to use in airflow.
* `executor=SequentialExecutor`
    - The executor class that Airflow should use
    - Options: SequentialExecutor, LocalExecutor, CeleryExecutor, DaskExecutor
* `sql_alchemy_conn=sqlite:////home/airflow/airflow/airflow.db`
    - The SqlAlchemy connection string used to connect airflow to the metadatabase. SqlAlchemy supports many different database engines.
* `sql_alchemy_pool_enabled=True`
    - If SqlAlchemy should pool database connections. A connection pool is a standard technique used to maintain long running connections in memory for efficient re-use as well as to provide management for the total number of connections an application might use simultaneously.

**Concurrency vs. Parallelism**
* A very important concept to understand before going forward is the difference between concurrency and parallelism in programming:
    - A system is said to be concurrent if it can support two or more actions in progress at the same time. A system is said to be parallel if it can support two or more actions executing simultaneously. The key concept and difference between these definitions is the phrase 'in progress.'
    - In concurrent systems, multiple actions can be in progress (may not be executed) at the same time, meanwhile, multiple actions are simultaneously executed in parallel systems.

**The Configuration File**
* There are three parameters in the configuration file that allow you to tune whether your DAGs are run in concurrency or parallelism.
* `parallelism=32`
    - The number of physical python processes (worker) the scheduler can run.
* `max_active_runs_per_dag=16`
    - The number of DAGRuns (per-DAG) to allow running at once.
* `dag_concurrency=16`
    - The number of task instances allowed to run per DAGRun at once.

**SQLite and Sequential Executor**
* It's basically the default configuration you get when you install Apache Airflow.
* Really suitable for debugging and testing.
* Do not use this configuration in Production since it doesn't scale.
* There is no parallelism and/or concurrency.

**PostgreSQL**
* PostgreSQL is an object-relational database and ACID compliant.
* PostgreSQL is a client-server database, so there is a server process (managing database files and connections, performing actions, etc.) and a client which is used to perform database operations.
* PostgreSQL can handle multiple concurrent connections form client in writing as well as in reading mode.
* It implements SQL standard as well as advanced SQL stuff like Window functions.
* Scalable.

**What is a Local Executor**
* Local Executor executes tasks locally in parallel. It uses the multiprocessing Python library and queues to parallelize the execution of tasks.
* It can run multiple tasks at a time.
* It run tasks by spawning processes in a controlled fashion in different modes on the same machine.
* You can tune the number of processes to spawn by using the parallelism parameter.

**Local Executor Strategies**
* There are two strategies depending on the parallelism value:
    1.  `parallelism == 0` which means unlimited parallelism. Every task submitted to the Local Executor will be executed in its own process. Once the task is executed and the result stored in the `result_queue`, the process terminates.
    2. `parallelism > 0` which means the Local Executor spawn the number of processes equal to the value of `parallelism` at start time using `task_queue` to coordinate the ingestion of tasks and the work distribution among the workers. During the lifecycle of the Local Executor in this mode, the worker processes are running waiting for tasks, once the Local Executor receives the call to shutdown the executor, a poison token is sent to the workers to terminate them.
* It is advised you set `parallelism` greater than zero.

**Local Executor with PostgreSQL**
* SQLite does not accept more than once writer which makes multiple tasks to run in parallel impossible.
* PostgreSQL is a perfect fit for Local Executor since it does accept multiple connections in both ways, writing and reading allowing for task parallelism.
* Technically a Sequential Executor could be thought of as a Local Executor with limited parallelism sets to 1.

**RabbitMQ**
* RabbitMQ is an open source message queuing software.
* Allows for the creation of queues where applications can be connected to in order to consume messages from these queues.
* Messages (data) placed onto the queue are stored until the consumer (a tierce application) retrieves them.
* A basic architecture of a message would be:
    - Client applications called producers, create messages and deliver them to the message queue (broker)
    - Other applications called consumers, connect to the queue and subscribe to the messages to process them.

**Celery Executor**
* Celery Executor is recommended for production use of Airflow.
* It allows distributing the execution of task instances to multiple worker nodes.
* Celery is a Python Task-Queue system that handle distribution of tasks on workers across threads or network nodes.
* The tasks need to be pushed into a broker like RabbitMQ, and Celery workers will pop them and schedule task executions.

**Celery Executor + RabbitMQ + PostreSQL**
* PostgreSQL is a database allowing multiple concurrent clients to connect in both read and write modes.
* Celery Executors allow to interact with Celery backend in order to distribute and execute task and instances on multiple worker nodes giving a way to high availability and horizontal scaling.
* Celery needs to use a broker in order to pull out from the worker nodes that task instances to execute and that's why we need to use RabbitMQ.

**Quiz**
1. What is a sequential executor?
    A: The executor is used to run one task at a time.
2. What is a local executor?
    A: A local executor can run multiple tasks in parallel
3. What is the main limitation of SQLite?
    A: It can accept only one writer at a time.
4. What does the parameter 'parallelism' do in the configuration file `Airflow.cfg`?
    A: It gives the number of allowed processes the executors can run to execute the tasks in parallel.
5. What is a Celery Executor?
    A: It allows distributing the execution of task instances to multiple worker nodes.

**Summary**
* The most basic configuration which is by default is the Sequential Executor with SQLite. This configuration is perfect for debugging but it does not allow you to run multiple concurrent tasks limiting Apache Airflow's performance. SQLite supports an unlimited number of readers but only one writer at a time.
* Only Sequential Executors can be used with SQLite.
* The difference between concurrency and parallelism is that a concurrent system can support two or more actions in progress at the same time whereas a parallel system supports two or more actions running simultaneously at the same time (they are at the same instruction to execute).
* Do not use Sequential Executors with SQLite in production since this configuration doesn't scale.
* The second configuration is Local Executor with PostgreSQL (or MySQL). PostgreSQL is a scalable client-server database allowing concurrent connections in reading and writing.
* Local Executors execute tasks locally in parallel by using the multiprocessing Python library and queues to parallelize the execution of tasks.
* They run tasks by spawning processes in a controlled fashion in different modes on the same machine.
* In the Airflow configuration file you can change `parallelism` with:
    - `0`: Unlimited parallelism, every task submitted to the Local Executor will be executed in its own process as they arrive.
    - `> 0`: Limited parallelism, spawn the number of processes equal to the value of `parallelism` at start time using a `task_queue` to coordinate the ingestion of tasks.
* The last configuration is Celery Executor with PostgreSQL (or MySQL) and Rabbit MQ.
* RabbitMQ is an open source message queuing software where multiple producers send messages to a queue where those messages are pulled out by the consumers.
* Celery is a Python Task-Queue system that handles distribution of tasks on workers across threads or network nodes.
* Celery Executors allow you to interact with Celery backend in order to distribute and execute task instances on multiple worker nodes giving a way to high availability and horizontal scaling.
* Celery needs to use a broker in order to pull out from the worker nodes and task instances to execute justifying the need of Rabbit MQ.
* This configuration is greatly recommended in production because it scales very well and allows you to achieve better performances.

#### Implementing Advanced Concepts in Airflow
**How to Create a SubDAGs**
* In order to create a subDAG you have to use a factory function that returns a DAG Object (the subDAG in our case) and the SubDagOperator to attach the subDAG to the main DAG.
* The factory function returns an instantiated DAG with the associated tasks. This function should be in a different file from where your main DAG is defined.

**Important Notes**
* The main DAG manages all the subDAGs as normal tasks. They are going to follow the same dependencies you had before. Nothing changes basically.
* Airflow UI only shows the main DAG. In order to see your subDAGs you will have to clock on the related main DAG and then 'zoom in' into the subDAGs from the 'Graph View'
* SubDags must be scheduled the same as their parent DAG.

**Interacting with External Sources Using Hooks**
* A Hook is simply an interface (a set of functions) to interact with external systems such as HIVE, PostgreSQL, Spark, SFTP, and so on. Apache Airflow provides you many different hooks in order to make your life easier.
* For instance, PostgreOperator uses PostgresHook to interact with your PostgreSQL database and execute your request. It handles the connection and allows you to execute SQL like if you were logged into your PostgreSQL command interface.

**Important Notes**
* The parameters 'schema' corresponds to the database name you want to connect to in PostgreSQL.
* `postgres_conn_id='postgre_sql'` Here `postgre_sql` is a connection created from the Airflow UI into the Connection view
* There are many official hooks such as PrestoHook, SqliteHook, SlackHook and so on that you can use.
* You can find also many very interesting unofficial hooks created by the community such as SparkSubmitHook (to kick off a spark submit job), FtpHook, JenkisHook, and so on.

**XCOMs**
* Apache Airflow introduced XCOMs to share key-value information between tasks.
* XCOM stands for 'cross-communication' and allows multiple tasks to exchange messages (data) between them.
* XCOMs are principally defined by a key, value and a timestamp.
* They are stored into the Airflow's metadatabase with an associated `execution_date`, `task_id` and `dag_id`.
* XCOMs (data) can be 'pushed' (sent) or 'pulled' (received).
* When we push a message from a task using `xcom_push()` this message because available to other tasks.
    - If a task returns a value (either from its Operator's `execute()` method, or from a PythonOperator's `python_callable()` function), a XCOM containing that value is automatically pushed.
* When we pull a message from a task using `xcom_pull()`, the task gets the message based on parameters such as `key`, `task_ids` (the 's' is not a mistyping) and `dag_id`.
    - By default, `xcom_pull()` for the keys that are automatically given to XCOMs when they are pushed by being returned from execute functions (as opposed to XCOMs that are pushed manually).

**Important Notes**
* XCOMs can be used to share any object that can be serialized (pickled) but be careful about the size of this object. Airflow is not a data streaming solution!
* Some operators such as BashOperator or SimpleHttpOperator have a parameter called `xcom_push=False` by default. If you set `xcom_push=True` the last output will be pushed to an XCOM for the BashOperator or if you use Simple HttpOperator, the response of the HTT request will also be pushed to an XCOM.
* Be careful, `execution_date` does not have the same meaning in the context of a DagRun and an XCOM. `execution_date` in XCOM is used to hide a XCOM until this date. For example, if we have two XCOMs with the same key value, dag id, and task id, the XCOM having the most recent `execution_date` will be pulled out by default. IF you didn't set an `execution_date`, this date will be equal to the `execution_date` of the DagRun.

**Branching**
* Branching is the mechanism allowing your DAG to choose between different paths according to the result of a specific task.
* To do this use the BranchPythonOperator
* The BranchPythonOperator is like the PythonOperator except that it expects a `python_callable` function that returns a `task_id`. In other words, the function passed to the parameter `python_callable` must return the `task_id` corresponding to the task which will be executed next.
* All other paths are skipped and only the path leading to the task with the corresponding `task_id` will be followed.
* The `task_id` returned by the Python function has to be referencing a task directly downstream from the BranchPythonOperator task.

**Depends_on_pas and Branching**
* You can use the property `depends_on_past` at the task level. It means this task will run only if the same task instance succeed in the previous DagRun. If there is no previous DagRun, the task will be triggered. Now you know that, there is no point to use `depends_on_past=True` on downstream tasks from the BranchPythonOperator, as skipped status will invariably lead to block tasks that depend on their past successes.

**SLA**
* A Service Level Agreement (SLA) is a contract between a service provider and the end user that defines the level of service expected from the service provider.
* SLAs do not define how the service itself is provided or delivered but rather define what the end user will receive. The level of service definitions should be specific and measurable.
* In Apache Airflow, Service Level Agreements are defined as the time by which a task or DAG should have succeeded.

**Adding SLA to a Task**
* SLAs are set at a task level (Operator) as a timedelta. For example, if we want to add an SLA to a task using the BashOperator we will do the following:
    `BashOperator(task_id='t1', sla=timedelta(seconds=5), bash_command="echo 'test'")`
* In this example, we are basically saying that the task t1 should not exceed 5 seconds to succeed. If it does, a SLA is recorded in the database and an email is automatically sent to the given email address of the DAG.

**Adding SLA Callback to the DAG**
* When you instantiate a DAG, there is one parameter that you can use to call a function when an SLA is missed:
    `sla_miss_callback=func(dag, t1, bt1, slas, btis)`
* This parameter expects a function with the following arguments:
    - `dag`: the dag object where the missed SLA(s) happened
    - `task_list`: the task list having missed their SLA with their associated execution_time
    - `blocking_task_list`: the task list being blocked by the tasks having missed their SLA with their associated execution_time
    - `slas`: same as the task_list but in a lost of object (not in string format)
    - `blocking_tis`: same as the blocking_task_list but in a lost object (not in string format)

**Important Notes**
* A SLA is relative to the execution_date of the task not the start time. You can only be alerted if the task runs 'more than 30 minutes FROM the execution_date'. You won't receive an alert if the task runs 'more than 30 minutes.'
* You may think of using the execution_timeout parameter to express 'more than 30 minutes' but it doesn't serve the same purpose as a SLA. The execution_timeout parameter sets the maximum allowed time before a task is stopped and marked as being failed. With a SLA, your task will still run and you will be alerted that tis processing time is longer than expected.
* Using SLAs with a DAG having a schedule_interval sets to None or `@once` has no effect. Because to check if a missed SLA must be triggered or not, Airflow looks at the next schedule_interval which in this case, does not exist.
* If one or multiple tasks haven't succeeded in time, an email alert is sent with the list of tasks that missed their SLA. This email alert is not sent from the callback and can't be turned off. The only way to avoid receiving email alerts is by setting the email parameter of your task to None.
* Be very careful when you use backfilling (process of replaying your DAG from the past) with tasks having a SLA. Indeed, you may end up with thousands of missed SLA in a very short period of time as it means the Apache Airflow that the tasks from the past didn't succeed in time.
* All tasks sharing the same missed SLA time are going to be merged together in order to be sent by email. Also, each SLA is saved into the database.
* SLAs are not well documented in Apache Airflow documentation

**Quiz**
1. What is the name of the special operator in order to use SubDAGs?
    A: SubDagOperator
2. What is a Hook?
    A: A Hook is used as an interface to interact with external systems
3. What happens when you return a value from a python_callable function?
    A: The value will be automatically pushed into a XCOM
4. How can I know which XCOM will be pulled out in first if they both have the same key?
    A: The XCOM having the most recent execution_date will be pulled out in first by default.
5. What is going to happen if we set depends_on_past=True when we use Branching?
    A: This completely lock your DAG.

**Summary**
* SubDAGs allow you to make your DAG clearer by encapsulating different logic group of tasks together. A SubDAG is created by sing a factory method that returns a DAG Object (subDAG) and the SubDagOperator to attach the subDAG to the main DAG.
* SubDags must be scheduled the same as their parent DAGs.
* SubDagOperator uses its own executor which by default is the Sequential Executor. You should stick to it in order to avoid possible bugs and performance degradations.
* SubDags must share the same start_date and schedule_interval as their parent.
* A Hook is simply an interface to interact with external systems such as HIVE, PostgreSQL, Spark, SFTP and so on.
* By using a Hook you can act like you are logged into your external system.
* Some operators actually use Hooks internally such as PostgreOperator or MySqlOperator.
* Hooks use connections stored into the metadatabase created from the Connection View.
* XCOM stands for 'cross-communication' and allows messages stored in the database to share data between multiple tasks.
* Those messages are defined by a key, a value, a timestamp, an execution date, a task id and a dag id.
* Data are pushed by `xcom_push()` and pulled by `xcom_pull()`
* If a task returns a value (either from its Operator's execute() method or from a PythonOperator's python_callable function), a XCOM containing that value is automatically pushed.
* Be default, `xcom_pull()` for the keys that are automatically given to XCOMs when they are pushed by being returned from execute functions (as opposed to XCOMs that are pushed manually).
* XCOMs are suitable from small values to share not for large sets of data.
* If we have two XCOMs with the same key value, dag id, and task id, the XCOM having the most recent `execution_date` will be pulled out by default. IF you didn't set an `execution_date`, this date will be equal to the `execution_date` of the DagRun.
* In both operators in which we use the functions xcom_push() and xcom_pull(), the parameter `provide_context` has been set to True. When `provide_context` is set to True, Airflow will pass a set of keyword arguments that can be used in your function. Those keyword arguments are passed through `**kwargs` variable. By using the 'ti' key from `**kwargs`, we get the TaskInstance object representing the task running the python_callable function, needed to pull or push a XCOM.
* Branching is the mechanism allowing your DAG to choose between different paths according to the result of a specific task.
    - To do this we use the BranchPythonOperator
* The BranchPythonOperator is like the PythonOperator except that it expects a python_callable that returns a task_id. In other words, the function passed to the parameter python_callable must return the task_id corresponding to the task which will be executed next.
* The task_id returned by the Python function has to be referencing a task directly downstream from the BranchPythonOperator
* There is no point to use `depends_on_past=True` on downstream tasks from the BranchPythonOperator as skipped status will invariably lead to block tasks that depend on their past successes.

#### Creating Airflow Plugins with Elasticsearch and PostgreSQL
