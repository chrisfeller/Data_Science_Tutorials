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
