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
