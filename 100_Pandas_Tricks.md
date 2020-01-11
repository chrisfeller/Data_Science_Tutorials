### Data School 100 Pandas Tricks
#### January 2020

**1) 5 useful `.read_csv()` parameters**
1. `names`: specify column names
2. `usecols`: which columns to use
3. `dtype`: specify data types
4. `nrows`: number of rows to read
5. `na_values`: strings to recognize as NaN

**2) Use these `.read_csv()` parameters if you have bad data (or empty rows) at teh top of your csv file**
* `header`: row number of header (zero indexed)
* `skiprows`: list of row numbers to skip
* Example: `pd.read_csv('data.csv', header=2, skiprows=[3, 4])`

**3) Two easy ways to reduce DataFrame memory usage**
1. Only read in columns you need
2. Use `category` data type with categorical data
* Example: `pd.read_csv('data.csv', usecols=['A', 'C', 'D'], dtype={'D':'category'})`

**4) You can read or write directly from/to a compressed file**
* Read: `pd.read_csv('https://h.co/3JAwA8h7FJ')`
* Write: `df.to_csv('https://h.co/3JAwA8h7FJ')`

**5) Avoid having a column named `Unnamed: 0` by setting the first column as the index (when reading) or don't save the index to the file (when writing)**
* Read: `pd.read_csv('data.csv', index_col=0)`
* Write: `df.to_csv('data.csv', index=False)`

**6) Combine dataset rows spread across multiple files into single DataFrame**
* Use `glob` to list your files and then use a generator expression to read files and `concat()` to combine them.
* Example:
    ```
    import pandas as pd
    from glob import glob

    # Step 1: Use glob function to list all files that match a pattern
    stock_files = sorted(glob('data/stocks*.csv'))

    # Step 2: Use a generator expression to read the files and use concat to combine them
    pd.concat((pd.read_csv(file) for file in stock_files), ignore_index=True)
    ```

**7) Combine dataset rows spread across multiple files into single DataFrame, while keeping track of which rows come from which files**
* Use `glob` to list your files and then read files with generator expression, creating a new column with `assign()`, and combine with `concat()`
* Example:
    ```
    import pandas as pd
    from glob import glob

    # Step 1: Use glob function to list all files that match a pattern
    stock_files = sorted(glob('data/stocks*.csv'))

    # Step 2: Use a generator expression to read the files, assign() to create a new column, and concat() to combine the DataFrames
    pd.concat((pd.read_csv(file).assign(filename=file) for file in stock_files), ignore_index=True)
    ```

**8) Read in a huge dataset into pandas by sampling rows during the file read**
* Randomly sample the dataset during the file reading by passing a function 'skiprows'.
* Example:
    ```
    df = pd.read_csv('big_data.csv', skiprows = lamba x: x > 0 and np.random.rand() > 0.01)
    ```
* How it works:
    - `skiprows` accepts a function that is evaluated against the integer index
    - `x > 0` ensures that the header row is **not** skipped
    - `np.random.rand() > 0.01` returns True 99% of the time, thus skipping 99% of the rows.

**9) Read in data from Excel or Google Sheets via the clipboard**
* Copy data to the clipboard and then `pd.read_clipboard()`

**10) Extract tables from a PDF into a DataFrame**
* Use `tabula-py` library
* Example:
    ```
    from tabula import read_pdf
    df = read_pdf('test.pdf', pages='all')
    ```

**11) Read a JSON from the web**
* Use `read_json()` to read it directly from a URL into a DataFrame
* Example:
    `df = pd.read_csv('https://api.github.com/users.justmarkham/repos?per_page=100')`

**12) Scrape a table from a web page**
* Try `read_html()` before trying `BeautifulSoup` and `Selenium`
* Example: `df = pd.read_html('https://finance.yahoo.com/quote/AAPL?p-APPL')`

**13) Use `read_html()` to scrape a webpage containing multiple tables**
* If you're using `read_html()` to scrape a webpage but it's returning too many tables, use the `match` parameter to find tables that contain a particular string.
* Example:
    ```
    import pandas as pd

    url = 'https://en.wikipedia.org/wiki/Twitter'

    tables = pd.read_html(url)
    # returns 17 tables
    len(tables)

    matching_tables = pd.read_html(url, match='Followers')
    # returns one table
    len(matching_tables)
    ```

**14) Create an example DataFrame**
* Three ways:
    1. `pd.DataFrame({'col_one': [10, 20], 'col_two': [30, 40]})`
    2. `pd.DataFrame(np.random.rand(2, 3), columns=list('abc'))`
    3. `pd.util.testing.makeMixedDataFrame()`

**15) Create a DataFrame for testing**
* Three ways:
    1. `pd.util.testing.makeDataFrame()` # random values
    2. `pd.util.testing.makeMissingDataframe()` # includes missing values
    3. `pd.util.testing.makeTimeDataFrame()` # has DateTime index
    4. `pd.util.testing.makeMixedDataFrame()` # mixed data types

**16) Create a time series dataset for testing**
* You can create a timeframe index in an existing dataframe:
    ```
    df = pd.DataFrame({'col_one': [10, 20], 'col_two': [30, 40]})

    df.index = pd.util.testing.makeDateIndex(len(df), freq='H')
    ```

**17) Create new columns or overwrite existing columns within a method chain**
* Use `assign()`
    ```
    (df.assign(col1 = df.col1.lower(),
               col2 = df.col1.upper()))
    ```

**18) Create a bunch of new columns based on existing columns**
* Use this pattern:
    ```
    for col in df.columns:
    df[f'{col}_new'] = df[col].apply(my_function)
    ```
* Example:
    ```
    for col in df.columns:
        df[f'{col}_fixed'] = df[col].str.upper()
    ```

**19) Remove a column from a DataFrame and store it was a seperate Series**
* Use `pop()`
* Example:
    ```
    label = iris.pop('species')
    ```

**20) Insert a new column into a DataFrame at a specific location**
* Use `insert()`
* Example:
    ```
    df.insert(3, 'C2', df['C'] * 2)
    ```

**21) Three ways to rename a column**
1. Most flexible option:
    `df = df.rename({'A': 'a', 'B': 'b'}, axis='columns')`
2. Overwrite all columns names:
    `df.columns = ['a', 'b']`
3. Apply string method:
    `df.columns = df.columns.str.lower()`

**22) Add a prefix or suffix to all of your column names**
* Prefix:
    `df.add_prefix('X_')`
* Suffix:
    `df.add_suffix('_Y')`

**23) Rename all columns in the same way**
* Replace spaces with an underscore:
    `df.columns = df.columns.str.replace('', '_')`
* Make lowercase and remove trailing whitespace:
    `df.columns = df.columns.str.lower().rstrip()`

**24) Use f-strings to select columns in a DataFrame**
* You can use f-strings (Python 3.6+) when selecting a Series from a DataFrame:
    `df[f'{}_servings]` # selects all columns with `serving` in name

**25) Select multiple rows/columns using `.loc`**
* Select a slice (inclusive):
    `df.loc[0:4, 'col_A':'col_D']`
* Select a list:
    `df.loc[[0, 3], ['col_A', 'col_C']]`
* Select a condition:
    `df.loc[df.col_A=='val', 'col_D']`

**26) Select by both label and position**
* 'loc' selects by label and 'iloc' selects by position. To select by label and position you can still use 'loc' and 'iloc'.
* Use `loc` with a mix of label and position:
    `df.['a', df.columns[0]]` or `df.loc[df.index[0], 'col_one']`
* Use `iloc` with a mix of label and position:
    `df.iloc[df.index.get_loc('a'), 0]`
* Don't use `ix` as it was deprecated.

**27) Chain 'loc' and 'iloc' together**
* To select from a DataFrame by label and position you can chain together `loc` and `iloc`:
    `drinks.iloc[15:20, :].loc[:, 'beer_servings':'wine_servings']`

**28) Reverse column and row order**
* Reverse column order in a DataFrame:
    `df.loc[:, ::-1]`
* Reverse row order:
    `df.loc[::-1]`
* Reverse row order and reset the index:
    `df.loc[::-1].reset_index(drop=True)`

**29) Select multiple slices of columns from a DataFrame**
1. Use `df.loc` to select and `pd.concat` to combine:
    `pd.concat([df.loc[:, 'A':'C'], df.loc[:, 'F'], df.loc[:, 'J': 'K']], axis='columns')`
2. Slice `df.columns` and select using brackets:
    `df[list(df.columns[0:3]) + list(df.columns[5]) + list(df.columns[9:11])]`
3. Use `np.r_` to combine slices and `df.iloc` to select:
    `df.iloc[:, np.r_[0:3, 5, 9:11]]`

**30) Filter DataFrame by multiple OR conditions**
* Traditional way:
    `df[(df.color=='red') | (df.color=='green') | (df.color=='blue')]`
* Shorter way:
    `df[df.color.isin(['red', 'green', 'blue'])]`
* Invert the filter above:
    `df[~df.color.isin(['red', 'green', 'blue'])]`

**31) Count or percentage of rows that match a condition**
* To find the count of rows that match a condition:
    `(movies.content_rating=='PG').sum()`
* To find the percent of rows that match a condition:
    `(movies.content_rating=='PG').mean()`

**32) Filter a DataFrame to only include the largest categories**
* to filter a DataFrame to only include the largest categories?
1. Save `value_counts()` output:
    `counts = movies.genre.value_counts()`
2. Get the index of its `head()`
    `largest_categories = counts.head(3).index`
3. Use that index with `isin()` to filer the DataFrame:
    `movies[movies.genre.isin(largest_categories)]`

**33) Combine the smaller categories in a Series into a single category called 'other'**
1. Save the index of the largest values of `value_counts()`
2. Use `where()` to replace all other values with 'Other'
* Example:
    ```
    top_four = genre.value_counts().nlargest(4).index
    genre_updated = genre.where(genre.isin(top_four), other='Other')
    ```

**35) Combine the small categories in a Series (<10% frequency) into a single category**
1. Save the normalized value counts
2. Filter by frequency and save the index
3. Replace small categories with 'Other'
* Example:
    ```
    frequencies = genre.value_counts(normalize=True)
    small_categories = frequencies[frequencies < 0.10].index
    genre_updated = genre.replace(small_categories, 'Other')
    ```

**36) Simplify filtering with multiple conditions**
* Instead of filtering a DataFrame using lots of criteria, save the criteria as objects and use them to filter.
    ```
    crit1 = df.continent =='Europe'
    crit2 = df.beer_servings > 200
    crit3 = df.wine_servings > 200
    crit4 = df.spirit_servings > 100

    df[crit1 & crit2 & crit3 & crit4]
    ```

* Alternatively, use `reduce()` to combine the criteria:
    ```
    from functools import reduce
    criteria = reduce(lambda x: x, y: x & y, [crit1, crit2, crit3, crit4])

    df[criteria]
    ```

**37) Filter a DataFrame that doesn't have a name**
* Instead of creating an intermediate variable such as:
    ```
    temp = stocks.groupby('Symbol').mean()
    temp[temp.Close < 100]
    ```
* Use `query()`:
    ```
    stocks.groupby('Symbol').mean().query('Close < 100')
    ```

**38) Refer to a local variable with a `query()` string?**
* Prefix it with an @ symbol:
    ```
    mean_volumne = stocks.Volume.mean()
    stocks.query('Volume > @mean_volume')
    ```

**39) Use `query()` on a column name containing a space**
* Surround it with backticks:
    ```
    df.query('`column name` > 40')
    ```

**40) Concatenate two string columns**
1. Use a string method:
    `ufo.City.str.cat(ufo.State, sep=', ')`
2. Use a plus sign:
    `ufo.city + ', ' + ufo.State`

**41) Split a string into multiple columns**
* Use `str.split()` method, `expand=True` to return a DataFrame, and assign it to the original DataFrame:
    ```
    df[['first', 'middle', 'last']] = df.name.str.split(' ', expand=True)
    ```
* To do the same but only save the first name:
    ```
    df['first'] = df.name.str.split(' ', expand=True)[0]
    ```

**42) Split names of variable length into first_name and last_name**
1. Use `str.split(n=1)` to split only once (returns a Series of lists)
2. Chain `str[0]` and `str[1]` on the end to select the list elements
    ```
    df['first_name'] = df.name.str.split(n=1).str[0]
    df['last_name'] = df.name.str.split(n=1).str[1]
    ```

**43) Count the number of words in a Series**
* Use a string method to count the spaces and add 1
    `df['word_count'] = df.messages.str.count(' ') + 1`

**44) Convert string columns to numeric**
* If you have numbers stored as strings:
    `df.astype({'col1': 'int', 'col2': 'float})`
* However, the above will fail if you have any invalid input.
* Instead use:
    `df.apply(pd.to_numeric, errors='coerce')`

**45) Select columns by data type**
* Examples:
    ```
    df.select_dtypes(include='number')
    df.select_dtypes(include=['number', 'category', 'object'])
    df.select_dtypes(exclude=['datetime', 'timedelta'])
    ```

**46) Save a massive amount of memory by fixing data types**
* Use `int8` for small integers
* Use `category` for strings with few unique values
* Use `Sparse` if most values are 0 or NaN
* Example:
    ```
    df = df.astype({'Pclass': 'int8',
                    'Sex': 'category',
                    'Parch': 'Sparse[int]',
                    'Cabin': 'Sparse[str]'})
    ```

**47) Check to see if object column contains mixed data types**
* Use `df.col.apply(type).value_counts()` to check.

**48) Clean an object column with mixed data types**
* Use `replace` (not str.replace) and regex:
    `df['sales'] = df.sales.replace('[$,]', '', regex=True).astype('float')`

**49) Use logic to order categories**
* Create ordered category:
    ```
    cat_type = CategoricalDtype(['good', 'very_good', 'excellent'], ordered=True)

    df['quality'] = df.quality.astype(cat_type)
    ```
* Now you can sort and compare categories logically:
    ```
    df.sort_values('quality')

    df[df.quality > 'good']
    ```

**50) Ensure correct data types for many columns**
1. Create CSV of column names & dtypes
2. Read it into a DataFrame
3. Convert it to a dict
4. Use dict to specify dtypes of dataset

**51) Convert a column from continuous to categorical**
1. Use `cut()` to specify bin edges:
    `pd.cut(titan.Age, bins=[0, 18, 25, 99])`
2. Use `qcut()` to specify number of bins (creates bins of approx. equal size)
    `pd.qcut(titanic.Age, q=3)`
* Each allow you to label bins:
    `pd.cut(titanic.Age, bins=[0, 18, 25, 99], labels=['child', 'young_adult', 'adult', 'elderly'])`

**52) Dummy Encode (One-Hot Encode) a DataFrame**
* Use `pd.get_dummies(df)` to encode all object and category columns.
* To drop the first level since it provides redundant information, set `drop_first=True`
* Example:
    `pd.get_dummies(df, drop_first=True)`

**53) Convert one set of values to another**
1. `map()` using a dictionary:
    `df['gender_letter'] = df.gender.map({'male': 'M', 'female': 'F'})`
2. `factorize()` to encode each value as an integer:
    `df['color_num'] = df.color.factorize()[0]`
3. comparison statement to return boolean values
    `df['can_vote'] = df.age >= 18`

**54) Apply the same mapping to multiple columns at one**
* Use `applymap()`, which is a DataFrame method, with `get`, a dictionary method.
    ```
    mapping = {'male': 0, 'female': 1}
    cols = ['B', 'C']
    df[cols] = df[cols].applymap(mapping.get)
    ```

**55) Expand a Series of lists into a DataFrame**
* When your data is 'trapped' in a Series of lists:
    ```
    import pandas as pd
    df = pd.DataFrame({'col_one': [1, 2, 3], 'col_two': [[10, 40], [20, 50], [30, 60]]})
    ```
* Expand the Series in a DataFrame by using `apply()` and passing it to the Series constructor:
    `df.col_two.apply(pd.Series)`

**56) Use Explode**
* When you have a Series containing a list of items you can create one row for each item using the `explode()` method.
    ```
    import pandas as pd
    df = pd.DataFrame({'sandwich': ['PB&J', 'BLT', 'cheese'],
                       'ingredients': [['peanut butter', 'jelly'],
                                       ['bacon', 'lettuce', 'tomato'],
                                       ['swiss cheese']]},
                        index=['a', 'b', 'c'])

    df.explode('ingredients')
    ```

**57) Deal with Series containing comma-separated items**
* When you have a Series containing comma-separated items  you can create one row for each item via:
1. `str.split()` creates a list of strings:
2. `assign()` overwrites the existing column
3. `explode()` creates the rows
    ```
    import pandas as pd
    df = pd.DataFrame({'sandwich': ['PB&J', 'BLT', 'cheese'],
                       'ingredients': [['peanut butter', 'jelly'],
                                       ['bacon', 'lettuce', 'tomato'],
                                       ['swiss cheese']]},
                        index=['a', 'b', 'c'])

    df.assign(ingredients = df.ingredients.str.split(',')).explode('ingredients')
    ```

**58) Reverse Explode**
* Reverse `explode()` with `groupby()` and `agg()`:
    ```
    df['imploded'] = df_exploded.groupby(df_exploded.index).ingredients.agg(list)
    ```

**59) Create a single datetime columns from multiple columns**
* You must include: month, day, year
* You can also include: hour, minute, second
* Make a single datetime with `to_datetime()`
    `df['date'] = pd.to_datetime(df[['month', 'day', 'year']])`

**60) Convert 'year' and 'day of year' into a single datetime column**
1. Combine them into one number
2. Convert to datetime and specify its format
    ```
    # step 1
    df['combined'] = df['year'] * 1000 + df['day_of_year']

    # step 2
    df['date'] = pd.to_datetime(df['combined'], format('%Y%j'))
    ```

**61) Access date attributes via datetime**
* To access helpful datetime attributes of a date use:
    `df.columns.dt.year`
* Can access:
    - `year`
    - `month`
    - `day`
    - `hour`
    - `minute`
    - `second`
    - `timetz` (timezone)
    - `dayofyear`
    - `weekofyear`
    - `week`
    - `dayofweek`
    - `weekday`
    - `weekday_name`
    - `quarter`
    - `days_in_month`
    - etc...

**62) Perform an aggregation with a given frequency (monthly, yearly, etc.)**
* To perform an aggregation (sum, mean, etc.) with a given frequency (monthly, yearly, etc.) use `resample()`.
* It's like a groupby for time series.
* Example:
    ```
    # for each year, show the sum of 'sales'
    df.resample('Y').sales.sum()
    ```

**63) Aggregate time series by weekend day**
* Problem: You have time series data that you want to aggregate by day, but you're only interested in weekend.
* Solution:
    1. resample by day ('D')
    2. filter by day of week (5=Saturday, 6=Sunday)
* Example:
    ```
    daily_sales = df.resample('D').hourly_sales.sum()
    weekend_sles = daily_sales[daily_sales.index.dayofweek.isin([5, 6])]
    ```

**64) Calculate the difference between each row and the previous row**
* Use `df.col_name.diff()`
* Instead to calculate the percent change use `df.col_name.pct_change()`
* Example:
    ```
    # Calculate change from previous day
    df['Change'] = df.Close.diff()
    df['Percent_Change'] = df.Close.pct_change()*100
    ```

**65) Convert a datetime Series from UTC to another time zone**
1. Set current time zone via `tz_location('UTC')`
2. Convert `tz_convert('America/Chicago')`
* Example:
    ```
    s = s.dt.tz_localize('UTC')
    s.dt.tz_convert('America/Chicago')
    ```

**66) Calculate % of missing values in each column**
* To calculate the percent of missing values in each column:
    `df.isna().mean()`
* To drop columns with any missing values:
    `df.dropna(axis='columns')`
* To drop columns in which more than 10% of values are missing:
    `df.dropna(thresh=len(df)*0.9, axis='columns')`

**67) Fill missing values in time series data**
* To fill missing values in a time series:
    `df.interpolate()`
    - Defaults to linear interpolation, but other methods are supported.

**68) Store missing values ('NaN') in an integer Series**
* Use `Int64`
* The default data type for integers is `int64` but `int64` doesn't support missing values. As a solution use `Int64`, which supports missing values.
* Example:
    `pd.Series([10, 20, np.nan], dtype='Int64)`

**69) Aggregate by multiple functions**
* Instead of aggregating by a single function (such as 'mean'), you can aggregate by multiple functions by using `agg` (and passing it a list of functions) or by using `describe()` (for summary statistics).
* Example:
    ```
    # aggregate by a single function
    df.groupby('continent'.beer_servings.mean())

    # aggregate by multiple functions
    df.groupby('continent').beer_servings.agg(['mean', 'count'])
    ```

**70) Extract the last value in each group**
* `last()` is an aggregation function just like `sum()` and `mean()`, which means it can be used with groupby to extract the last value in each group.
* Example:
    ```
    # when was each patient's last visit?
    df.groupby('patient').visit.last()
    ```
* You can also use `first()` and `nth()`

**71) Name the output columns of multiple aggregations**
* Named aggregation allows you to name the output columns and avoids a column MultiIndex
* Example:
    ```
    # leads to uninformative and MultiIndex columns
    titanic.groupby('Pclass').Age.agg(['mean', 'max'])

    # solution
    titanic.groupby('Pclass').Age.agg(mean_age='mean', max_age='max')
    ```

**72) Combine the output of an aggregation with original DataFrame**
* Instead of `df.groupby('col').col2.func()` use `df.groupby('col1').col2.transform(func)`

**73) Calculate a running total of a Series**
* Use `cumsum()`
* Example:
    ```
    df['running_total'] = df.sales.cumsum()
    ```

**74) Calculate running count within groups**
* Use: `df.groupby('col').cumcount() + 1`
* Example:
    ```
    df['count_by_person'] = df.groupby('salesperson').cumcount() + 1
    ```

**75) Randomly sample rows from a DdataFrame**
* To get a random number of rows: `df.sample(n=10)`
* To get a random proportion of rows: `df.sample(frac=0.25)`
* Useful parameters:
    - `random_state`: use any integer for reproducibility
    - `replace`: sample with replacement
    - `weights`: weight based on values in a column

**76) Shuffle your DataFrame rows**
* Use `df.sample(frac=1, random_state=0)`
* To reset the index after shuffling:
    `df.sample(frac=1, random_state=0).reset_index(drop=True)`

**77) Split a DataFrame into two random subsets**
* To split a DataFrame into two random subsets:
    ```
    df1 = df.sample(frac=0.75, random_state=42)
    df2 = df.drop(df_1.index)
    ```
* Only works if df's index values are unique

**78) Identify the source of each row in DataFrame merge**
* To identify the source of each row (left, right, both) in a DataFrame merge use the setting `indicator=True`
* Example:
    ```
    pd.merge(df1, df2, how='left', indicator=True)
    ```

**79) Check that merge keys are unique in two DataFrames**
* To check that merge keys are unique in BOTH datasets use `pd.merge(left, right, validate='one_to_one')`
* Use `one_to_many` to only check uniqueness in LEFT
* Use `many_to_one` to only check uniqueness in RIGHT

**80) Style a DataFrame**
1. `df.style.hide_index()`
2. `df.style.set_caption('My caption')`

**81) Add formatting to your DataFrame**
* To format dates and numbers:
    ```
    format_dict = {'Date': '{:%m/%d/%y}', 'Close': '${:.2f}'}
    stocks.style.format(format_dict)
    ```
* To highlight min and max values:
    ```
    (stocks.style.highlight_min('Close', color='red')
                 .highlight_max('Close', color='green'))
    ```

**82) Explore a new dataset without too much work**
1. `conda install -c conda-forge pandas-profiling`
2. `import pandas_profiling`
3. `df.profile_report()`

**83) Check to see if two Series contain the same elements**
* Don't do this: `df.A == df.B`
* Do either:
    ```
    df.A.equals(df.B)

    # or

    df.equals(df2)
    ```
* `equals()` properly handles NaNs, whereas `==` doesn't

**84) To check if two Series are similar**
* Use `pd.testing.assert_series_equal(df.A, df.B)`
* Useful arguments:
    - `check_names=False`
    - `check_dtypes=False`
    - `check_exact=False`

**85) Change default of how many rows to display**
* If DataFrame has more than 60 rows, only show 10 rows (saves your screen space)
    `pd.set_options('min_rows', 4)`


**86) Examine the 'head' of a wide DataFrame**
1. Change the display options to show all columns:
    `pd.set_options('display.max_columns', None)`
2. Transpose the head (swaps rows and columns)
    `df.head().T`

**87) Plot a DataFrame**
* It's as easy as `df.plot(kind='...')`
* Plot options:
    - line
    - bar
    - barh
    - hist
    - box
    - kde
    - area
    - scatter
    - hexbin
    - pie

**88) Create interactive plots**
1. Either:
    1. `pip install hvplot`
    2. `conda install -c conda-forge hvplot`
2. `pd.options.plotting.backend = 'hvplot'`
3. `df.plot(...)`
* Example:
    ```
    import pandas as pd
    df = pd.read_csv('http://bit.ly/drinksbycountry')
    pd.options.plotting.backend = 'hvplot'
    df.plot(kind='scatter', x='spirit_servings', y='wine_servings', c='continent')
    ```

**89) Handle `SettingWithCopyWarning`**
* Rewrite your assignment using `loc`:
    ```
    import pandas as pd
    df = pd.DataFrame({'gender': ['Male', 'Female', 'Male', 'F', 'Female'})

    # Wrong and will lead to SettingWithCopyWarning
    df[df.gender == 'F'].gender = 'Female'

    # Correct solution using `loc`
    df.loc[df.gender == 'F'] = 'Female'
    ```

**90) Handle `SettingWithCopyWarning` when creating a new columns**
* You are probably assigning to a DataFrame that was created from another DataFrame.
* As a solution use `copy()` method when copying a DataFrame
    ```
    import pandas as pd
    df = pd.DataFrame({'gender': ['Male', 'Female', 'Male', 'Female']})

    # Wrong and will lead to SettingWithCopyWarning
    males = df[df.gender == 'Male']
    males['abbreviation'] = 'M'

    # Correct use `copy()`
    males = df[df.gender == 'Male'].copy()
    males['abbreviation'] = 'M'
    ```

**91) Rearrange the columns in your DataFrame**
1. Specify all column names in desired order
    ```
    cols = ['A', 'B', 'C', 'D']
    df[cols]
    ```
2. Specify columns to move, followed by remaining columns
    ```
    cols_to_move = ['A', 'C']
    cols = cols_to_move + [col for col in df.columns if col not in cols_to_move]
    df[cols]
    ```
3. Specify column positions in desired order
    ```
    cols = df.columns[[0, 2, 3, 1]]
    df[cols]
    ```

**92) Transform DataFrame from wide to long format**
* Use `melt()`
* Example:
    `df.melt(id_vars='zip_code', var_name='location_type', value_name='distance')`

**93) Access specific groups in groupby object**
* If you've created a groupby object, you can access any of the groups (as a DataFrame) using the `get_group()` method.
* Example:
    ```
    import pandas as pd
    df = pd.read_csv('http://bit.ly.imdbratings')
    gb = df.groupby('genre')
    gb.get_group('Western')
    ```

**94) Reshape a Series with a MultiIndex to a DataFrame**
* Use `unstack()`
* Example:
    ```
    import pandas as pd
    titanic = pd.read_csv('http://bit.ly/kaggletrain')
    titanic.groupby(['Sex', 'Pclass']).Survived.mean().unstack()
    ```

**95) Change display options**
* To change default options: `pd.set_option('display.max_rows', 80)`
* To revert to default options: `pd.reset_option('display.max_rows')`
* To view all options: `pd.describe_option()`
* Most common options:
    - `max_rows`
    - `max_columns`
    - `max_colwidth`
    - `precision`
    - `date_dayfirst`
    - `date_yearfirst`

**96) Show total memory usage of a DataFrame**
* To show total memory usage of a DataFrame: `df.info(memory_usage='deep')`
* To show memory used by each column: `df.memory_usage(deep=True)`
* To reduce the memory usage of a column either drop unused columns or con vert object columns to `category` type

**97) Find out what version of pandas you're using**
* To find out what version of pandas you're using: `pd.__version__`
* To determine the versions of its dependencies: `pd.show_versions()`

**98) Access numpy without importing it**
* You can use numpy without importing it via: `pd.np.random.rand(2, 3)`
