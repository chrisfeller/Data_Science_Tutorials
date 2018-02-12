### Data School Pandas Tutorial
#### October 2017

#### How do I read a tabular data file into pandas?

~~~
import pandas as pd
orders = pd.read_table('http://bit.ly/chiporders')
~~~

~~~
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('http://bit.ly/movieusers', sep='|', header=None, names=user_cols)
~~~

#### How do I select a pandas Series from a DataFrame?
* There are two data types in pandas
    1) DataFrame (similar to excel table)
    2) Series (similar to a column in an excel table)
~~~
import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports')
~~~
* We will use 'bracket notation' to select a Series:
~~~
ufo['City']
~~~
* Bracket notation is case sensitive
* A shortcut to bracket notation is called 'dot notation':
~~~
ufo.City
~~~
* How would I select the Series 'Colors Reported'?
    * If there is a space in the column name, dot notation does not work and you must use bracket notation.
    * This also goes for columns that are named similar to core pandas functions:
        * (i.e., shape, head, describe)
* To create a new column, you must use bracket notation:
~~~
ufo['Location'] = ufo.City + ' ' + ufo.State
~~~
**Takeaway: Bracket notation will always work, dot notation will not. However dot notation is easier to write.**

#### Why do some pandas commands end with parentheses (and others don't)?
~~~
import pandas as pd
movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head() #Look at the first five rows
movies.describe() #Shows descriptive statistics of all numerical columns
movies.shape #Tuples that displays the number of rows and columns
movies.dtypes #Shows us the data types of each column
~~~
* Since movies is a DataFrame it has certain methods and attributes.
    * Methods are actions
        * Methods end in parentheses `()`
    * Attributes are descriptions
        * Attributes do not end in parentheses
* To see all of the methods and attributes of an object type the name followed by a period and then hit tab. `movies.<>tab`
* In Ipython, anytime you have a method or function and you want to know what the arguments are you put a ? behind the call:
~~~
movies.describe?
~~~

#### How do I rename columns in a pandas DataFrame?
~~~
import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports')
~~~
* To see all of the column names:
~~~
ufo.columns
~~~
* To rename the 'Colors Reported' and 'Shape Reported':
~~~
ufo.rename(columns={'Colors Reported': 'Colors_Reported', 'Shape Reported': 'Shape_Reported'}, inplace=True)
~~~
* In place means that you want it to work on the underlying DataFrame not just return the results.
* To rename all of the columns without having to specify each:
~~~
ufo_cols = ['city', 'colors reported', 'shape reported', 'state', 'time']
ufo.columns = ufo_cols
~~~
* To rename the columns while reading in the file:
~~~
ufo = pd.read_csv('http://bit.ly/uforeports', names=ufo_cols, header=0)
~~~
* To replace all spaces with underscores:
~~~
ufo.columns = ufo.columns.str.replace(' ', '_')
~~~

#### How do I remove columns from a pandas DataFrame
~~~
import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports')
~~~
* To remove the 'Colors Reported' column:
~~~
ufo.drop('Colors Reported', axis=1, inplace=True)
~~~
* To remove multiple columns:
~~~
ufo.drop(['City', 'State'], axis=1, inplace=True)
~~~
* To remove rows:
~~~
ufo.drop([:2], axis=0, inplace=True)
~~~
* Axis changes to zero and instead of column names you use indexes, which in most case are numerical.
    * `axis=0` is the default so you don't need to include it but as a beginner it may be useful to use it.

#### How do I sort a pandas DataFrame or Series
~~~
import pandas as pd
movies = pd.read_csv('http://bit.ly/imdbratings')
~~~
* To sort a single Series in a DataFrame:
~~~
movies.title.sort_values() #dot notation
movies['title'].sort_values() #bracket notation
~~~
* `sort_values()` sorts in ascending by default.
* To sort in descending order:
~~~
movies['title'].sort_values(ascending=False)
~~~
* To sort a DataFrame by a Series:
~~~
movies.sort_values('title')
~~~
* To sort by multiple columns:
~~~
movies.sort_values(['content_rating', 'duration'])
~~~
* This sorts first by content_rating and then by duration.

#### How do I filter rows of a pandas DataFrame by column value?
~~~
import pandas as pd
movies = pd.read_csv('http://bit.ly/imdbratings')
~~~
* To filter the movies dataframe for movies that are over 200 minutes in duration (the long way):
~~~
booleans = []
for length in movies.duration:
    if length >= 200:
        booleans.append(True)
    else:
        booleans.append(False)
is_long = Series(booleans)
movies[is_long]
~~~
* To filter the movies dataframe for movies that are over 200 minutes in duration (the short way):
~~~
movies[movies.duration >= 200]
~~~
* To select the genre of movies with duration greater than 200 minutes:
~~~
movies.loc[movies.duration >=200, 'genre']
~~~
* The above is the best practice!

#### How do I apply multiple filter criteria to a pandas DataFrame?
~~~
import pandas as pd
movies = pd.read_csv('http://bit.ly/imdbratings')
~~~
* True or False = True
* True or True = True
* False or False = False
* True and False = False
* False and False = False
* True and True = True
* To select movies that are over 200 minutes in duration **AND** of the genre drama:
~~~
movies[(movies.duration >= 200) & (movies.genre == 'Drama')]
~~~
* To select movies that are over 200 minutes in duration **OR** of the genre drama:
~~~
movies[(movies.duration >= 200) | (movies.genre == 'Drama')]
~~~
* To select movies that are either crime or drama or action for their genre:
~~~
movies[(movies.genre.isin(['Crime', 'Drama', 'Action']))]
~~~

#### How do I import only certain columns or rows from a .csv file?
* To select only the City and State columns via their name:
~~~
import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports', usecols=['City', 'State'])
~~~
* To select only the City and State columns via their position:
~~~
ufo = pd.read_csv('http://bit.ly/uforeports', usecols=[0,3])
~~~
* To select only the first 10 rows:
~~~
ufo = pd.read_csv('http://bit.ly/uforeports', nrows=3)
~~~

#### How do I drop every non-numeric column from a dataframe?
~~~
import pandas as pd
import numpy as np
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.select_dtypes(include=[np.number])
~~~

#### How do I use the 'axis' parameter in pandas?
~~~
import pandas as pd
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
~~~
* Columns are Axis=1
* Rows are Axis=0
    * The default is Axis=0
* To delete the column 'continent':
~~~
drinks.drop('continent', axis=1, inplace=True)
~~~
* To delete the row 'Afghanistan' (which has index of 0):
~~~
drinks.drop(0, axis=0, inplace=True)
~~~
* To find the mean of each column:
~~~
drinks.mean()
~~~
* The above will return the mean of each column since the default is axis=0
* Instead to find the mean of each country and thus row:
~~~
drinks.mean(axis=1)
~~~
* Aliases for the axis numbers:
    * You can substitute 'index' for 0 and 'columns' for 1:
~~~
drinks.mean(axis='index')
 #OR
drinks.mean(axis='columns')
~~~

#### How do I use string methods in pandas?
~~~
import pandas as pd
orders = pd.read_table('http://bit.ly/chiporders')
~~~
* To uppercase the `item_name` column:
~~~
orders.item_name.str.upper()
~~~
* Note the `.str` in from of upper, which we need to include for all string methods in pandas.
* To select all items with the word chicken in it:
~~~
orders[orders.item_name.str.contains('Chicken')]
~~~
* To remove the word 'chicken' from all order types:
~~~
orders.item_name.str.replace('chicken', '')
~~~
* To remove two words from the order type:
~~~
orders.item_name.str.replace('chicken', '').str.replace('carnitas', '')   
~~~

#### How do I read data from the clipboard?
~~~
import pandas as pd
df = pd.read_clipboard()
~~~

#### How do I change the data type of a pandas Series?
~~~
import pandas as pd
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
~~~
* To change the `beer_servings` columns from integer to float:
~~~
drinks.beer_servings.astype(float)
~~~
* To define the data type of a column during import:
~~~
drinks = pd.read_csv('http://bit.ly/drinksbycountry', dtype={'beer_servings':float})
~~~
* To remove the $ from the item_price column in the chipotle dataset in order to do calculations:
~~~
import pandas as pd
orders = pd.read_table('http://bit.ly/chiporders')
orders.item_price.str.replace('$', '').astype(float)
~~~
* To change boolean filter values to binary zeroes and ones:
~~~
orders.item_name.str.contains('Chicken').astype(int)
~~~

#### When should I use 'groupby' in pandas?
~~~
import pandas as pd
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
~~~
* To find the average beer servings across all countries:
~~~
drinks.beer_servings.mean()
~~~
* To find the average beer servings by continent:
~~~
drinks.groupby('continent').beer_servings.mean()
~~~
* When to use groupby?
    * Anytime you want to analyze some pandas Series by a category.
* To find multiple descriptors at once:
~~~
drinks.groupby('continent').beer_servings.agg(['min', 'max', 'mean', 'count'])
~~~
* To find the mean of each column in the dataframe:
~~~
drinks.groupby('continent').mean()
~~~
* To display these groupby's visually:
~~~
import matplotlib.pyplot as plt
drinks.groupby('continent').mean().plot(kind='bar')
 # In Ipython you must start Ipython with the tag --pylab inline for this to work.
~~~

#### How do I explore a pandas Series
~~~
import pandas as pd
movies = pd.read_csv('http://bit.ly/imdbratings')
~~~
* To see descriptive statistics on the `genre` columns:
~~~
movies.genre.value_counts()
 #OR
movies.genre.describe()
~~~
* To turn descriptive statistics into percentages:
~~~
movies.genre.value_counts(normalize=True)
~~~
* To see all of the unique values of a column:
~~~
movies.genre.unique()
~~~
* To see how many unique values are in a column:
~~~
movies.genre.nunique()
~~~
* To get a cross tabulation (meaning how many records fit into two separate columns):
~~~
pd.crosstab(movies.genre, movies.content_rating)
~~~
* To visualize the distribution of movie duration:
~~~
import matplotlib.pylot as plt
movies.duration.plot(kind='hist')
~~~

#### How do I handle missing values in pandas?
~~~
import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports')
~~~
* To check which values are null:
~~~
ufo.isnull()
~~~
* To check which values are not null:
~~~
ufo.notnull()
~~~
* To see how many values are missing in each column:
~~~
ufo.isnull().sum()
~~~
* To select all of the rows in the columns City with null values:
~~~
ufo[ufo.City.isnull()]
~~~
* To drop all missing values:
~~~
ufo.dropna(how='any')
 #OR
ufo.dropna(how='all')
 #OR
ufo.dropna(subset=['City', 'Shape Reported'], how='any')
~~~
* The any argument means that it will drop all rows with missing values in any of the columns.
* The all argument means that it will drop all rows with missing values in all of the columns.
* The subset argument means that it will drop all rows with missing values if any or all of the values in the specified columns are missing.
* To replace all missing values in a column:
~~~
ufo['Shape Reported'].fillna(value='No Longer Missing')
~~~

#### What do I need to know about the pandas index?
~~~
import pandas as pd
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
~~~
* Index is synonymous with row labels
* Neither the index nor column headers are considered part of the dataframe contents.
    * Thus, they are not counted in the .shape information.
* Indexes and column headers default to integers (i.e. 1, 2, 3, etc.)
* Three reasons why indexes exist:
    1) Identification
    2) Selection
    3) Alignment
* To set index (in this case to the country names):
~~~
drinks.set_index('country', inplace=True)
~~~
* The index has a name. To remove that name:
~~~
drinks.index.name = None
~~~
* To move the index to it's own column:
~~~
drinks.index.name = 'country'
drinks.reset_index(inplace=True)
~~~
* To get the 25th percentile of a columns:
~~~
drinks.describe().loc['25%', 'beer_servings']
~~~
* To sort by the index:
~~~
drinks.continent.value_counts().sort_index()
~~~

#### How do I select multiple rows and columns from a pandas DataFrame?
~~~
import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports')
~~~
* `.loc` is for filtering rows and selecting columns *by label*
    * The format for `.loc` is **[what rows do I want, what columns do I want]**
        * Remember: [R, C] = RC Cars!
    * Example: `ufo.loc[0:10, :]` returns the first ten rows of all columns
        * Indexing in loc is inclusive!
    * Example: `ufo.loc[:, 'City']` returns all rows of the City column
    * Example: `ufo.loc[:,['City', 'State']]` return all rows of the City and State columns
    * Example: `ufo.loc[:,'City': 'State']` returns all rows between the columns City and State.
    * To use `.loc` for boolean indexing: `ufo.loc[ufo.City=='Oakland', :]`
* `iloc` is for filtering rows and selecting columns *by integer position*
    * The format for `.iloc` is also **[what rows do I want, what columns do I want]**
    * Example: `ufo.iloc[:, [0, 3]]` returns all rows from columns 0 and three.
    * Example: `ufo.iloc[:, 0:4]` returns all rows for the columns between 0 and 3.
        * indexing in iloc is exclusive!
    * Example: `ufo.iloc[0:3, :]`
* Don't ever use `.ix` as it is deprecated.

#### When should I use the 'inplace' parameter in pandas?
~~~
import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports')
~~~
* To drop the column `City` permanently:
~~~
ufo.drop('City', axis=1, inplace=True)
~~~
* To return a new DataFrame with changes use `inplace=False`, which is the default.
* To change the underlying DataFrame, use `inplace=True`

#### How do I make my pandas DataFrame smaller and faster?
~~~
import pandas as pd
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
~~~
* To get more info on the columns in the DataFrame (including the memory usage):
~~~
drinks.info()
~~~
* However, that memory usage really isn't the true size. To get the true size:
~~~
drinks.info(memory_usage='deep')
~~~
* To see the size (in bytes) of each column:    
~~~
drinks.memory_usage(deep=True)
~~~
* Columns with type object take up the most space.
* To reduce the size of object-type columns, change categories to integer values via lookup table:
~~~
drinks['continent'] = drinks.continent.astype('category')
~~~
* However, this really only works for columns with a few limited values (i.e. continents) and not for columns with lots of values (i.e. countries).

#### How do I create logical ordering of a category column?
* Create an example DataFrame:
~~~
df = pd.DataFrame({'ID':[100, 101, 102, 103], 'quality':['good', 'very good', 'good', 'excellent']})
~~~
* We know that there is an inherent logical ordering from good, to very good, to excellent. However, the DataFrame at the moment only sorts the column in alphabetical order (i.e. excellent, then good, then very good).
* To create a logical ordering of these categories:
~~~
df['quality'] = df.quality.astype('category', categories=['good', 'very good', 'excellent'], ordered=True)
~~~
* Now when we sort it will sort by logical order:
~~~
df.sort_values('quality')
~~~
* You are also now able to sort by boolean indexing. For example, to select all rows with a quality better than good.
~~~
df.loc[df.quality > 'good']
~~~

#### How do I use pandas with scikit-learn to create Kaggle submissions?
~~~
import pandas as pd
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('http://bit.ly/kaggletrain')
~~~
* To predict which passengers on the Titanic survived:
~~~
feature_cols = ['Pclass', 'Parch']
x = train.loc[:, feature_cols]
y = train.Survived

logreg = LogisticRegression()
logreg.fit(x, y)

test = pd.read_csv('http://bit.ly/kaggletest')
x_new = test.loc[:, feature_cols]
new_pred_class = logreg.predict(x_new)
test.PassengerID

pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId').to_csv('kaggle_submission.csv')
~~~

#### How do I take a random sample from a DataFrame?
~~~
import pandas as pd
uf = pd.read_csv('http://bit.ly/uforeports')
~~~
* To get a random sample of three rows from the ufo DataFrame:
~~~
ufo.sample(n=3)
~~~
* To make the random sample reproducible:
~~~
ufo.sample(n=3, random_state=42)
~~~
* To get a random sample made up of a fraction of the DataFrame:
~~~
ufo.sample(frac=0.75)
~~~
* This would be most useful when selecting a training (75%) and test set (25%):
~~~
train = ufo.sample(frac=0.75)
test = ufo.loc[~ufo.index.isin(train.index),:]
~~~

#### How do I create dummy variables in pandas?
* Dummy variables are also known as indicator variables.
~~~
import pandas as pd
train = pd.read_csv('http://bit.ly/kaggletrain')
~~~
* To create a dummy variable for the `Sex` column (only two values):
~~~
train['Sex_male'] = train.Sex.map({'female':0, 'male':1})
 #OR
pd.get_dummies(train.Sex, prefix='Sex').iloc[:,1:]
~~~
* To create a dummy variable for the `Embarked` column (three values):
~~~
pd.get_dummies(train.Embarked, prefix='Embarked').iloc[:,1:]
~~~
* To pass a DataFrame to get.dummies:
~~~
pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)
~~~

#### How do I work with dates and times in pandas?
~~~
import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports')
~~~
* The `Time` column is an object. To convert the `Time` column to datetime:
~~~
ufo['Time'] = pd.to_datetime(ufo.Time)
~~~
* To extract the hour:
~~~
ufo.Time.dt.hour
~~~
* To extract the weekday:
~~~
ufo.Time.dt.weekday_name
~~~
* To extract the day of the year:
~~~
ufo.Time.dt.dayofyear
~~~
* To get the difference in days between the first time and last time:
~~~
(ufo.Time.max() - ufo.Time.min()).days
~~~

#### How do I find and remove duplicate rows in pandas?
~~~
import pandas as pd
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('http://bit.ly/movieusers', sep='|', header=None, names=user_cols, index_col='user_id')
~~~
* To identify duplicate zip codes:
~~~
users.zip_code.duplicated()
~~~
* To identify duplicates in the entire DataFrame:
~~~
users.duplicated()
~~~
* To count the number of duplicates in the DataFrame:
~~~
users.duplicated().sum()
~~~
* To show the duplicate rows:
~~~
users.loc[users.duplicated(), :]
~~~
* To drop duplicates:
~~~
users.drop_duplicates(keep='first')
~~~
* The `keep='first'` argument is the default and means it will keep the first row of each duplciate.
* To remove the duplicates of specified columns:
~~~
users.drop_duplicates(subset=['age', 'zip_code'])
~~~

#### How do I change display options in pandas?
~~~
import pandas as pd
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
~~~
* To see all of the display options, visit: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_option.html (Pandas Version 0.20.3)
* To view the max rows in display view:
~~~
pd.get_option('display.max_rows')
~~~
* To change the max rows in display views:
~~~
pd.set_option('display.max_rows', 200)
~~~
* To reset the max rows in display views:
~~~
pd.reset_option('display.max_rows')
~~~
* To change the max columns:
~~~
pd.set_option('display.max_columns', 10)
~~~
* To change the max column width (meaning it won't cut off any text within a column):
~~~
pd.set_option('display.max_colwidth', 1000)
~~~
* To change the precision of decimal places:
~~~
pd.set_option('display.precision', 6)
~~~
* How to change columns to display commas within a number:
~~~
pd.set_option('display.float_format', '{:,}'.format)
~~~
* The above only affects float columns. Thus, to change an integer column to display commas, first change them to float type first.
* To display options without searching on google:
~~~
pd.describe_option()
 #OR
pd.describe_option('search')
~~~
* To reset all options:
~~~
pd.reset_option('all')
~~~

#### How do I create a pandas DataFrame from another object?
* To create a DataFrame from dictionaries:
~~~
pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']})
~~~
* To specify the order of the columns, which the above does not do:
~~~
pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']}, columns=['id', 'color'])
~~~
* To specify the index:
~~~
pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']}, index=['a', 'b', 'c'])
~~~
* To create a DataFrame from a list of lists:
~~~
pd.DataFrame([[100, 'red'], [101, 'blue'], [102, 'red']])
~~~
* To specify the column names:
~~~
pd.DataFrame([[100, 'red'], [101, 'blue'], [102, 'red']], columns=['id', 'color'])
~~~
* To create a DataFrame from a numpy array:
~~~
import numpy as np
arr = np.random.rand(4, 2)
pd.DataFrame(arr, columns=['one', 'two'])
~~~
* To create a large DataFrame:
~~~
pd.DataFrame({'student':np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)})
~~~

#### How do I apply a function to a pandas Series or DataFrame?
~~~
import pandas as pd
train = pd.read_csv('http://bit.ly/kaggletrain')
~~~
* `map` is a Series method, which allows you to map an existing value of a Series to a different set of values.
* To add 2 to the age of every passenger in the Titanic DataFrame:
~~~
map(lambda x: x + 2,train.Age)
~~~
* `apply` is both a Series and DataFrame method.
* As a Series method, `apply` applies a function to each element in a Series.
* To calculate the length of each name in the Titanic DataFrame:
~~~
train.Name.apply(len)
~~~
* As a DataFrame method, `apply` applies a function across a column or row:
* To get the max of the beer_servings, spirit_servings, and wine_servings columns:
~~~
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=0)
~~~
* `applymap` is a DataFrame method, which applies a function to every element of the DataFrame.
* To change every numerical value to float:
~~~
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.loc[:, 'beer_servings':'wine_servings'].applymap(float)
~~~

#### 4 new time-saving tricks in pandas?
* New pandas features for pandas 0.22

1) Create a datetime column from a DataFrame
    * Create an example DataFrame:
    ~~~
    import pandas as pd
    df = pd.DataFrame([[12, 25, 2017, 10], [1, 15, 2018, 11]], columns=['month', 'day', 'year', 'hour'])
    ~~~
    * To create a datetime series (column) from the entire DataFrame:
    ~~~
    pd.to_datetime(df)
    ~~~
    * To create a datetime series (column) from a subset of columns:
    ~~~
    pd.to_datetime(df[['month', 'day', 'year']])
    ~~~
    * To overwrite the index into a datetime series from a subset of columns:
    ~~~
    df.index = pd.to_datetime(df[['month', 'day', 'year']])
    ~~~

2) Create a category column during file reading
    * Old way to create a category (after file reading):
    ~~~
    drinks = pd.read_csv('http://bit.ly/drinksbycountry')
    drinks['continent'] = drinks.continent.astype('category')
    ~~~
    * New way to create a category (during file reading):
    ~~~
    drinks = pd.read_csv('http://bit.ly/drinksbycountry', dtype={'continent':'category'})
    ~~~

3) Convert the data type of multiple columns at once
    * Old way to convert data types (one at a time):
    ~~~
    drinks = pd.read_csv('http://bit.ly/drinksbycountry')
    drinks['beer_servings'] = drinks.beer_servings.astype('float')
    drinks['spirit_servings'] = drinks.spirit_servings.astype('float')
    ~~~
    * New way to convert data types (all at once):
    ~~~
    drinks = pd.read_csv('http://bit.ly/drinksbycountry')
    drinks = drinks.astype({'beer_servings': 'float', 'spirit_servings': 'float'})
    ~~~  

4) Apply multiple aggregations on a Series or DataFrame
    * Example of a single aggregation function after a groupby:
    ~~~
    drinks = pd.read_csv('http://bit.ly/drinksbycountry')
    drinks.groupby('continent').beer_servings.mean()
    ~~~
    * Example of multiple aggregation functions applied simultaneously:
    ~~~
    drinks.groupby('continent').beer_servings.agg(['mean', 'min', 'max'])
    ~~~
    * New: apply the same aggregations to a Series:
    ~~~
    drinks.beer_servings.agg(['mean', 'min', 'max'])
    ~~~
    * New: apply the same aggregations to a DataFrame:
    ~~~
    drinks.agg(['mean', 'min', 'max'])
    ~~~
    * DataFrame describe method provides similar functionality but is less flexible:
    ~~~
    drinks.describe()
    ~~~

#### 5 new changes in pandas you need to know about
* New pandas features for pandas 0.22

1) `ix` has been deprecated
    * Two ways to slice a DataFrame:
    ~~~
    drinks = pd.read_csv('http://bit.ly/drinksbycountry', index_col='country')

    # loc accesses by label
    drinks.loc['Angola', 'spirit_servings']

    # iloc accesses by position
    drinks.iloc[4,1]
    ~~~

2) Aliases have been added for `isnull` and `notnull`
    * To check which values are missing:
    ~~~
    ufo = pd.read_csv('http://bit.ly.uforeports')
    ufo.isnull()
    ~~~
    * To check which values are not missing:
    ~~~
    ufo.notnull()
    ~~~
    * To drop rows with missing values:
    ~~~
    ufo.dropna()
    ~~~
    * To fill in missing values:
    ~~~
    ufo.fillna(value='UNKNOWN')
    ~~~
    * Instead of `isnull` use:
    ~~~
    ufo.isna()
    ~~~
    * Instead of `notna` use:
    ~~~
    ufo.notna()
    ~~~

3) `drop` now accepts 'index' and 'columns' keywords
    * Old way to drop rows; specify labels and axis:
    ~~~
    ufo = pd.read_csv('http://bit.ly.uforeports')
    ufo.drop([0, 1], axis=0)
    ~~~
    * New way to drop rows; specify index:
    ~~~
    ufo.drop(index=[0, 1])
    ~~~
    * Old way to drop columns:
    ~~~
    ufo.drop(['City', 'State', axis=1])
    ~~~
    * New way to drop columns:
    ~~~
    ufo.drop(columns=['City', 'State'])
    ~~~

4) `rename` and `reindex` now accept 'axis' keyword
    * Old way to rename columns: specify columns:
    ~~~
    ufo.rename(columns={'City': 'CITY', 'State': 'STATE'})
    ~~~
    * New way to rename columns: specify mapper and axis:
    ~~~
    ufo.rename({'City':'CITY', 'State':'STATE'}, axis='columns')
    ~~~
    * Note: mapper can be a function:
    ~~~
    ufo.rename(str.upper, axis='columns')
    ~~~

5) Ordered categories must be specified independent of the data
    * Create a small DataFrame:
    ~~~
    df = pd.DataFrame({'ID':[100, 101, 102, 103], 'quality':['good', 'very good', 'good', 'excellent']})
    ~~~
    * Old way to create an ordered category (deprecated):
    ~~~
    df.quality.astype('category', categories=['good', 'very good', 'excellent'], ordered=True)
    ~~~
    * New way to create an ordered category:
    ~~~
    from pandas.api.types import CategoricalDtype
    quality_cat = CategoricalDtype(['good', 'very good', 'excellent'], ordered=True)
    df['quality'] = df.quality.astype(quality_cat)
    ~~~


---

#### Questions:
1) What is the difference between the `sep=` and `delimiter=` parameters in `pd.read_table`? Which should I use?
