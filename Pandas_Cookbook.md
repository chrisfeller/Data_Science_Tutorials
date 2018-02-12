### Pandas Cookbook
#### October 2017

---
#### A Quick Tour of IPython Notebook
* IPython Notebook has been renamed to Jupyter Notebook.
* To start Jupyter Notebook, navigate to the .ipynb file in the console, and then:
~~~
jupyter notebook
~~~
* To run a cell, hit `Shift+Enter`
* Jupyter Notebook comes with robust tab-completion. So anytime you need help with which method to use or which arguments a method has, utilize tab completion.
* Jupyter Notebook autosaves.
* Useful magic functions:
    * To time how long a command runs: `%time print('hello world!')`
    * To render plots inline: `%matplotlib inline`
    * To view all magic commands: `%quickref`

#### Chapter 1: Reading from a CSV:
* The file for this chapter can be found at '/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/bikes.csv'
* First import numpy, pandas, and matplotlib.pyplot:
~~~
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
~~~

**Reading date from a csv file:**
* To read a .csv file:
~~~
broken_df = pd.read_csv('broken_df.csv')
~~~
* By default, `pd.read_csv` assumes that the fields of a .csv file are comma-separated.
* To look at the first three rows:
~~~
broken_df[:3]
 #OR
broken_df.head(3)
~~~
* If the .csv file is not separated by commas (in this case ';'):
~~~
broken_df = pd.read_csv('broken_df.csv', sep=';')
~~~
* If the encoding is latin1 instead of the default utf8:
~~~
broken_df = pd.read_csv('broken_df.csv', encoding='latin1')
~~~
* If there is a column that should be in datetime format (in this case the column name is 'Date'). We'll also include the `dayfirst` argument because the datetime in the column is European and thus the day is included before the month in 31/1/2012 format.
~~~
pd.read_csv('broken_df.csv', parse=['Date'], dayfirst=True)
~~~
* To specify one of the columns to be the index of the DataFrame (in this case the 'Date' column):
~~~
pd.read_csv('broken_df.csv', index_col='Date')
~~~
* The total import with all of the above arguments specified:
~~~
fixed_df = pd.read_csv('broken_df.csv', sep=';', encoding='latin1', parse_dates=True, dayfirst=True, index_col='Date')
~~~

**Selecting a columns:**
* You select columns out of a DataFrame the same way you get elements out of a dictionary.
* To select the 'Berri 1' column from the fixed_df DataFrame:
~~~
fixed_df['Berri 1']
~~~

**Plotting:**
* To plot a column:
~~~
fixed_df['Berri 1'].plot()
~~~
* To plot all of the columns:
~~~
fixed_df.plot()
~~~

#### Chapter 2: Selecting Data & Finding the Most Common Complaint Type:
* The file for this chapter can be found at: '/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/311-service-requests.csv'
* First import numpy, pandas, and matplotlib.pyplot:
~~~
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
~~~
* Next, load the data:
~~~
complaints = pd.read_csv('311-service-requests.csv')
~~~
* To print the entire dataframe:
~~~
complaints
~~~
* To look at the first five rows:
~~~
complaints[:5]
 #OR
complaints.head()
~~~
**Selecting columns and rows:**
* To select a column, index with the name of the column:
~~~
complaints['Complaint Type']
~~~
* To look at the first five rows of a column:
~~~
complaints['Complaint Type'][:5]
 #OR
complaints['Complaint Type'].head()
~~~
* To select multiple columns:
~~~
complaints[['Complaint Type', 'Borough']]
~~~
* To select the first five rows of multiple columns:
~~~
complaints[['Complaint Type', 'Borough']][:5]
 #OR
complaints[['Complaint Type', 'Borough']].head()
~~~
* To get the count of each value in a column:
~~~
complaints['Complaint Type'].value_counts()
~~~
* To plot the top ten complaint types:
~~~
complaints['Complaint Type'].value_counts()[:10].plot(kind='bar')
~~~

#### Chapter 3: Which borough has the most noise complaints (or, more selecting data)
* The file for this chapter can be found at: '/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/311-service-requests.csv'
* First import numpy, pandas, and matplotlib.pyplot:
~~~
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
~~~
* Next, load the data:
~~~
complaints = pd.read_csv('311-service-requests.csv')
~~~
* To select all of the noise complaints:
~~~
complaints[complaints['Complaint Type']  == 'Noise']
~~~
* To select all of the noise complaints that occur in Brooklyn:
~~~
complaints[(complaints['Complaint Type']  == 'Noise') & (complaints['Borough'] == 'BROOKLYN')]
~~~
* Note: All DataFrame columns by themselves are just Series. To get all of the values of a column:
~~~
complaints['Borough'].values
~~~
* Which borough has the most noise complaints:
~~~
complaints[complaints['Complaint Type']  == 'Noise']['Borough'].value_counts()
~~~
* Which borough has the highest percentage of noise complaints?
~~~
complaints[complaints['Complaint Type']  == 'Noise']['Borough'].value_counts() / complaints['Borough'].value_counts()
~~~
* Plot the answer to the following question:
~~~
complaints[complaints['Complaint Type']  == 'Noise']['Borough'].value_counts() / complaints['Borough'].value_counts().plot(kind='bar')
~~~

#### Chapter 4: Find out on which weekday people bike the most with groupby and aggregate
* The file for this chapter can be found at '/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/bikes.csv'
* First import numpy, pandas, and matplotlib.pyplot:
~~~
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
~~~
* Next, load the data:
~~~
bikes = pd.read_csv('/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/bikes.csv', sep=';', encoding='latin1', parse_dates=True, dayfirst=True, index_col='Date')
~~~
* Create a weekday column:
~~~
bikes.loc[:, 'weekday'] = berri_bikes.index.weekday
~~~
* Add up the cyclists by weekday:
~~~
bikes.groupby('weekday').sum()['Berri 1']
~~~
* Plot the above question, but first change the index from numbers to days:
~~~
grouped = bikes.groupby('weekday').sum()['Berri 1']
grouped.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
grouped.plot(kind='bar')
~~~

#### Chapter 5: Combining dataframes and scraping Canadian weather
* The file for this chapter can be found at '/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/weather_2012.csv'
* First import numpy, pandas, and matplotlib.pyplot:
~~~
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
~~~
* Next, load the data:
~~~
weather_2012_final = pd.read_csv('/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/weather_2012.csv', index_col='Date/Time')
~~~

**Subsetting Individual Months:**
* To subset weather data for March 2012 :
~~~
weather_mar2012 = weather_2012_final['2012-03-01':'2012-03-21']
~~~
* To plot the March data:
~~~
weather_mar2012['Temp (C)'].plot()
~~~
* To drop columns with missing data:
~~~
weather_mar2012 = weather_mar2012.dropna(axis=1, how='any')
~~~
* To drop an individual columns, in this case the 'Visibility (km)' column:
~~~
weather_mar2012 = weather_mar2012.drop(['Visibility (km)'], axis=1)
~~~

**Plotting the temperature by hour of day:**
* First change the index to datetime:
~~~
weather_mar2012.index = pd.to_datetime(weather_mar2012.index)
~~~
* Next, grouby and plot:
~~~
weather_mar2012['Temp (C)'].groupby(weather_mar2012.index.hour).agg(np.median).plot()
~~~

**Scraping month-by-month data:**
* Example from Cookbook:
* First, create function to scrape individual months:
~~~
def download_weather_month(year, month):
    if month == 1:
        year += 1
    url = url_template.format(year=year, month=month)
    weather_data = pd.read_csv(url, skiprows=15, index_col='Date/Time', parse_dates=True, header=True)
    weather_data = weather_data.dropna(axis=1)
    weather_Data.columns = [col.replace('\xbo', '') for col in weather_data.columns]
    weather_data = weather_data.drop(['Year', 'Day', 'Month', 'Time', 'Data Quality'], axis=1)
    return weather_data
~~~
* Next, run the function for all months:
~~~
data_by_month = download_weather_month(2012, i) for i in range(1, 13)]
~~~
* Combine months:
~~~
weather_2012 = pd.concat(data_by_month)
~~~

**Saving DataFrame to file:**
* To save the weather_2012 dataframe to a .csv file:
~~~
weather_2012.to_csv('weather_2012.csv')
~~~

#### Chapter 6: String Operations - Which month was the snowiest?
* The file for this chapter can be found at '/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/weather_2012.csv'
* First import numpy, pandas, and matplotlib.pyplot:
~~~
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
~~~
* Next, load the data:
~~~
weather_2012_final = pd.read_csv('/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/weather_2012.csv', index_col='Date/Time')
~~~
* To create a new column to indicate if it was snowing or not:
~~~
weather_2012_final['is_snowing'] = np.where(weather_2012_final['Weather'].str.contains('Snow'), 1, 0)
~~~
* The above could also be done via:
~~~
weather_2012_final['is_snowing'] = weather_2012_final['Weather'].str.contains('Snow').astype(int)
~~~

**Use resampling to find the snowiest month:**
* To find the median temperature of each month, first change the index to datetime:
~~~
weather_2012_final.index = pd.to_datetime(weather_2012_final.index)
~~~
* Next, use resample to get the individual months:
~~~
weather_2012_final['Temp (C)'].resample('M').apply(np.median)
~~~
* To plot the above:
~~~
weather_2012_final['Temp (C)'].resample('M').apply(np.median).plot(kind='bar')
~~~
* To find the percentage of time it was snowing each month:
~~~
weather_2012_final['is_snowing'].resample('M').apply(np.mean)
~~~
* To plot the above:
~~~
weather_2012_final['is_snowing'].resample('M').apply(np.mean).plot(kind='bar')
~~~

**Plotting temperature and snowiness stats together:**
* To combine these two statistics (temperature and snowiness) into one dataframe:
~~~
temperature = weather_2012_final['Temp (C)'].resample('M').apply(np.median)
snowiness = weather_2012_final['is_snowing'].resample('M').apply(np.mean)
stats = pd.concat([temperature, snowiness], axis=1)
stats.index = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
~~~
* To plot the two columns in separate subplots:
~~~
stats.plot(kind='bar', subplots=True)
~~~

#### Chapter 7: Cleaning up messy data:
* The file for this chapter can be found at: '/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/311-service-requests.csv'
* First import numpy, pandas, and matplotlib.pyplot:
~~~
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
~~~
* Next, load the data:
~~~
complaints = pd.read_csv('311-service-requests.csv')
~~~

**How do we know if it's messy?:**
* To get a sense for whether a column has problems:
    1) If the column is categorical:
        2) use `unique()` to look at the underlying values.
            * Are all of the data points of the same type?
            * Are there `Nan` values?
            * Are there obscure values?
            * Are there 'N/A' or 'NO CLUE' values that pandas doesn't understand?
    1) If the column is numeric:
        2) Plot a histogram to get a sense of the distribution.
            * Are there outliers?

**Cleaning Column Names:**
* To remove spaces in header names to snake case:
~~~
df.columns = [x.replace(" ", "_") for x in df.columns]
~~~

**Fixing the nan values and string/float confusion:**
* To turn incorrect `Nan` values, such as 'NO CLUE', 'N/A', OR '0', to correct `Nan`:
~~~
na_values = ['NO CLUE', 'N/A', '0']
complaints = pd.read_csv('311-service-requests.csv', na_values=na_values)
~~~
* To specify the data type of a column so that all of its values are correct:
~~~
complaints = pd.read_csv('311-service-requests.csv', dtype={'Incident Zip':str})
~~~

**Standardizing Column Values:**
* Our 'Incident_Zip' still has both five-digit and nine-digit zip codes. To truncate all of the nine-digit zip-codes:
~~~
complaints['Incident_Zip'] = complaints['Incident_Zip'].str.slice(0,5)
~~~
* There still are two rows with zip code '00000', to replace those with `Nan` values:
~~~
complaints[complaints.Incident_Zip == '00000'] = np.nan
~~~

#### Chapter 8: How to Deal with timestamps:
* The file for this chapter can be found at: '/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/popularity-contest'
* First import numpy, pandas, and matplotlib.pyplot:
~~~
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
~~~
* Next, load the data:
~~~
popcon = pd.read_csv('popularity-contest', sep=' ',)[:-1]
popcon.columns = ['atime', 'ctime', 'package-name', 'mru-program', 'tag']
~~~
* The 'atime' and 'ctime' are both in unix time. Fortunately, the magical part about parsing timestamps in pandas is that numpy datetimes are already stored as Unix timestamps. So all we need to do is tell pandas that these integers are actually datetimes - it doesn't need to do any conversion at all.
* First, change the unix type to an integer:
~~~
popcon['atime'] = popcon['atime'].astype(int)
popcon['ctime'] = popcon['ctime'].astype(int)
~~~
* Second, change the int type to datetime:
~~~
popcon['atime'] = pd.to_datetime(popcon['atime'], unit='s')
popcon['ctime'] = pd.to_datetime(popcon['ctime'], unit='s')
~~~
* To filter all rows based on a timestamp:
~~~
popcon[popcon['atime'] > '2011-01-01']
~~~

#### Chapter 9: Loading data from SQL databases:
* The file for this chapter can be found at: '/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/weather_2012.sqlite'
* First import numpy, pandas, and matplotlib.pyplot:
~~~
import numpy as np
import pandas as pd
import sqlite3
~~~
* You read from a sql database via the `pd.read_sql` function, which takes two arguments:
    1) a SELECT statement
    2) a database connection object
* `pd.read_sql` can read from MySQL, SQLite, PostgreSQL and others.
    * It automatically converts SQL column names to DataFrame column names.
* To read in a SQL database:
~~~
con = sqlite3.connect('/Users/chrisfeller/Desktop/Data_Science/Python/Example_Code/Tutorials/Pandas_Cookbook/data/weather_2012.sqlite')
df = pd.read_sql("SELECT * FROM weather_2012 LIMIT 3", con)
df
~~~
* `read_sql` doesn't automatically set the primary key (id) to the index of the dataframe. To do so:
~~~
df = pd.read_sql("SELECT * from weather_2012 LIMIT 3", con, index_col='id')
~~~
* If you want your dataframe to be indexed by more than one column, you can give a list of columns to the `index_col` argument:
~~~
df = pd.read_sql("SELECT * from weather_2012 LIMIT 3", con,
                 index_col=['id', 'date_time'])
~~~

**Writing to a SQLite database:**
* To write a dataframe to a database:
~~~
weather_df = pd.read_csv('weather_2012.csv')
con = sqlite3.connect("test_db.sqlite")
con.execute("DROP TABLE IF EXISTS weather_2012")
weather_df.to_sql("weather_2012", con)
~~~

**Reading from SQL Databases:**
* You read from an SQL database similar to how you would query a SQL database in SQL:
~~~
df = pd.read_sql("SELECT * from weather_2012 ORDER BY Weather LIMIT 3", con)
~~~

**Other SQL databases:**
* To connect to a MySQL database:
~~~
import MySQLdb
con = MySQLdb.connect(host="localhost", db="test")
~~~
* To connect to a PostgreSQL database:
~~~
import psycopg2
con = psycopg2.connect(host="localhost")
~~~
