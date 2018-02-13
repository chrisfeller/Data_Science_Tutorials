### Python for Data Analysis
#### Wes Mckinney
#### September 2017

---
#### Python 2.7 | Python 3.6
* Python 2.7 is the default on my machine.
* To switch to Python 3.6 via the terminal:
~~~
source activate python3
~~~
* To switch back to Python 2.7 via the terminal:
~~~
source deactivate
~~~
* To view all environments (i.e. Python 2.7 and Python 3):
~~~
conda env list
~~~

#### Import Conventions
The Python community has adopted the following naming conventions for commonly-used modules.
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
~~~
* Thus, functions will be accessed via something similar to np.arange.
    * This is done instead of importing everything `from numpy import *`, which is considered bad practice.

#### Terminology
**Munge/Munging/Wrangling:** Describes the overall process of manipulating unstructured and/or messy data into a structured or clean form.

**Pseudocode:** A description of an algorithm or process that takes a code-like form while likely not being actual valid source code.
* Example:
~~~
If student's grade is greater than or equal to 60
    print 'passed'
else
    print 'failed'
~~~
**Syntactic sugar:** Programming syntax which does not add new features, but makes something more convenient or easier to type.

**DataFrame:** The main pandas data structure is the DataFrame, which you can think of as representing a table or spreadsheet of data.

**Vectorization:** Batch operations on data (arrays) without writing any for loops.
* Any arithmetic operations between equal-size arrays applies the operation *elementwise*.
* Operations between differently sized arrays is called *broadcasting*.

**Universal Functions:** A universal function, or `ufunc`, is a function that performs elementwise operations on data in ndarrays.

---
#### Chapter 3: IPython: An Interactive Computing and Development Environment
**Launching IPython on the command line:**
~~~
$ ipython
~~~
**Tab Completion:** A feature common to most interactive data analysis environments. While entering expressions in the shell, pressing `<Tab>` will search the namespace for any variable (objects, functions, etc.) matching the characters you have typed so far:
~~~
an_apple = 27
an_example = 42
an<Tab>
    an_apple and an_example
~~~
* You can also complete methods and attributes on any object after typing a period:
~~~
b = [1, 2, 3]
b.<Tab>
    b.append    b.extend    b.insert    b.remove    b.sort
    b.counts    b.index     b.pop       b.reverse
~~~
* The same goes for modules:
~~~
import datetime
datetime.<Tab>
    datetime.date   datetime.MAXYEAR    datetime.timedelta
    dateime.time    datetime.MINYEAR    datetime.tzinfo
~~~
* IPython by default hides methods and attributes starting with underscores, such as magic methods and internal 'private' methods and attributes. To access those via tab completion:
~~~
import datetime
datetime._<Tab>
    datetime.__doc__    datetime.__name__
    datetime.__file__   datetime.__package__
~~~
* Tab completion also completes anything that looks like a file path on your computer's file system matching what you've typed:
~~~
path = 'book_scripts/<Tab>
    book_scripts/cprof_example.py   book_scripts/ipython_script_test.py
    book_scripts/ipython_bug.py     book_scripts/prof_mod.py
~~~
**Object Introspection:** Typing a `?` before or after a variable will display come general information about the object:
~~~
a = [1, 2, 3]
a?
~~~
* Using `??` will also show the function's source code if possible.

**The %run command:** To run a python script within the IPython console:
~~~
run fivethirtyeight.py
~~~
* This works identically to running `$ python fivethirtyeight.py` on the command line.
* Once you've run a `.py` file, all of the variables (imports, functions, and globals) defined in the file will be accessible in the IPython shell.

**Interrupting Running Code:** Pressing `<Ctrl-C>` while any code is running will cause nearly all Python programs to stop.

**Executing Code From the Clipboard:** To execute code copied on the clipboard:
~~~
%paste
~~~
**Keyboard Shortcuts:**
| Command  | Description  |
|---|---|
| `Ctrl-p` or `up-arrow`  | Search backward in command history  |
| `Ctrl-n` or `down-arrow`  | Search forward in command history  |
| `Ctrl-r`  | Readline-style reverse history search  |
| `Command-v`  | Past text from clipboard  |
| `Ctrl-c`  | Interrupt currently-executing code  |
| `Ctrl-a`  | Move cursor to the beginning of the line  |
| `Ctrl-e`  | Move cursor to the end of the line  |
| `Ctrl-k`  | Delete text from cursor until end of line  |
| `Ctrl-u`  | Discard all text on current line  |
| `Ctrl-f`  | Move cursor forward one character  |
| `Ctrl-b`  | Move cursor back on character  |
| `Ctrl-l`  | Clear screen  |

**Magic Commands:** Designed to facilitate common tasks and enable you to easily control the behavior of the IPython system.
| Command  | Description  |
|---|---|
| `%quickref`  | Display the IPython Quick Reference Card  |
| `%magic`  | Display detailed documentation for all of the available magic commands  |
| `%debug`  | Enter the interactive debugger at the bottom of the last exception traceback  |
| `%hist`  | Print command input (and optionally output) history  |
| `pdb`  | Automatically enter debugger after any execution  |
| `%paste`  | Execute pre-formatted Python code from clipboard  |
| `%cpaste`  | Open a special prompt for manually pasting Python code to be executed  |
| `%reset`  | Delete all variables/names defined in interactive namespace  |
| `%page OBJECT`  | Pretty print the object and display it through a pager  |
| `%run script.py`  | Run a Python script inside IPython  |
| `%prun statement`  | Execute statement with cProfile and report the profiler output  |
| `%time statement`  | Report the execution time of single statement  |
| `%timeit statement`  | Run a statement multiple times to compute an emsemble average execution time.   |
| `%who, %who_ls, %whos`  | Display variables defined in interactive namespace, with varying levels of information/verbosity  |
| `%xdel variable`  | Delete a variable and attempt to clear any references to the object in the IPython internals  |

**Matplotlib Integration and Pylab Mode:** If you create a matplotlib plot window in the regular IPython shell, you'll be sad to find that the GUI event loop 'takes control' of the IPython session until the window is closed. To avoid this, launch IPython with matplotlib integration on the command line:
~~~
$ ipython --pylab
~~~
**Input and Output Variables:** IPython stores references to both the input (the text that you type) and output (the object that is returned) in special variables:
* The previous output is stored in the `_` (underscore).
* Input variables are scored in variables named `_iX`, where `X` is the input line number.
* Thus, you can access any input or output via their line number:
~~~
_i27 # Returns the input at line 27
_27  # Returns the output at line 27
~~~
**Logging the Input and Output:**: IPython is capable of logging the entire console session including input and output. Logging is turned in by typing:
~~~
%logstart:
~~~
* Companion functions are: `%loggoff, %logon, %logstate, and %logstop`

**Bookmarks:** IPython has a simple directory bookmarking system to enable you to bookmark common directories so you can jump to them very easily. For example, if you use Dropbox a lot, it would make sense to bookmark that file path:
~~~
%bookmark db /home/wesm/Dropbox/
~~~
* Then, whenever you need to quickly navigate to Dropbox you can:
~~~
cd db
~~~
* Bookmarks are automatically persisted between IPython sessions.
---

**Changing Working Directory**
* To change the working directory in iPython:
~~~
import os
os.chdir('/Users/chrisfeller/Desktop/Python_Code')
~~~

**Viewing file contents in Ipython:**
* To quickly view a file's contents in Ipython:
~~~
!cat file_name.csv
~~~

#### Chapter 4: Numpy Basics: Arrays and Vectorized Computation
**Numpy:** Short for Numerical Python, is the fundamental package required for high-performance scientific computing and data analysis. Here are some of the things it provides:
* `ndarray`: a fast and space-efficient multidimensional array providing vectorized arithmetic operations and sophisticated broadcasting capabilities
* Fast vectorized array operations for data munging and cleaning, subsetting and filtering, transformation, and any other kinds ofcomputations
* Common array algorithms like sorting, unique, and set operations
* Efficient descriptive statistics and aggregating/summarizing data
* Data alignment and relational data manipulations for merging and joining together heterogeneous data sets
* Expressing conditional logic as array expressions instead of loops with if-elif-else branches
* Group-wise data manipulations (aggregation, transformation, function application).
    * *While NumPy provides the computational foundation for these operations, you will likely want to use pandas as your basis for most kinds of data analysis (especially for structured or tabular data).*

**The NumPy ndarray: A Multidimensional Array Object**: a N-dimensional array object, which is a fast, flexible container for large data sets in Python.
* An ndarray is a generic multidimensional container for ***homogeneous*** data; that is, all of the elements must by the same type.
* Every ndarray has a `shape`, a tuple indicating the size of each dimension:
    `data.shape`
* And a `dtype`, which describes the data type of the array:
    `data.dtype`
* In most cases, 'array', 'NumPy array', and 'ndarray' are synonymous.

**Creating ndarrays:** The easiest way to create an array is to use the `array` function:
~~~
data = [6, 7.5, 8, 0, 1]
arr = np.array(data)
~~~
* Nested sequences, like a list of equal-length lists will be converted into a multidimensional array.
~~~
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
~~~
* `zeros` creates an array of 0's with a given length or shape:
~~~
np.zeros(10)
 # OR
np.zeroes((3, 6))
~~~
* `ones` creates an array of 1's with a given length or shape:
~~~
np.ones(10)
 # OR
np.ones((3, 6))
~~~
* `empty` creates an array of random placeholder values:
~~~
np.empty((2, 3))
~~~
* `arange` is an array-valued version of the build-in Python `range` function:
~~~
np.arange(15)
~~~
**Specifying Data Types for ndarrays:** To specify the data type or `dtype` for an array:
~~~
arr = np.array([1, 2, 3], dtype=np.float64)
~~~
**Change `dtype` of array:** To explicitly convert or cast an array from one dtype to another:
~~~
arr = np.array([1, 2, 3, 4, 5])
float_arr = arr.astype(np.float64)
~~~
**Operations Between Arrays:** Any arithmetic operations between equal-size arrays applies the operation elementwise:
~~~
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr * arr
arr - arr
~~~
* Arithmetic operations with scalars (a single number or non-vector item) are as you would expect, propagating the value to each element:
~~~
arr = np.array([[1, 2, 3], [4, 5, 6]])
1 / arr
arr ** 0.5
~~~
**Basic Indexing and Slicing:**
* One-dimensional arrays act similarly to Python lists:
~~~
arr = np.arange(10)
arr[5] # Selects item at index 5
arr[5:8] # Selects items between index 5 and 7
~~~
* If you assign a scalar value to a slice, the value is propagated (or broadcast) to the entire selection:
~~~
arr = np.arange(10)
arr[5:8] = 12 # returns array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])
~~~
* An important first distinction from lists is that array slices are views on the original array. This means that the data is not copied, and any modifications to the view will be reflected in the source array.
    * If you want to copy a slice of an ndarray instead of a view, you will need to explicitly copy the array:
~~~
arr[5:8].copy()
~~~
* In a two-dimensional array, the elements at each index are no longer scalars but rather one-dimensional arrays:
~~~
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2] # returns array([7, 8, 9])
~~~
* To acces an individual element in a X-d array:
~~~
arr2d[0][[2]
 # OR
arr2d[0, 2]
~~~
* Helpful diagram on Page 86

**Boolean Indexing:** To return a boolean for a certain condition:
~~~
names = np.array(['Bob', ' Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names == 'Bob' # returns array([ True, False, False,  True, False, False, False], dtype=bool)
~~~
* To select everything but `Bob`, use `!=`:
~~~
names != 'Bob'
~~~
* Selecting two of the three names to combine multiple boolean conditions, use boolean arithmetic operations like `&` (and) and `|` (or):
~~~
mask = (names == 'Bob') | (names == 'Will')
~~~
* The Python keywords `and` and `or` do not work with boolean arrays.
* To set values with boolean arrays:
~~~
names[names == 'Bob'] = 'Chris'
~~~
**Fancy Indexing:** Term adopted by NumPy to describe indexing using integer arrays:
~~~
arr = np.empty((8, 4)) # Create an 8x4 array of placeholder values.

for i in range(7):
    arr[i] = i         # Replace placeholder values with the numbers 0-7

arr[[4, 3, 0, 6]]      # Select the 4th, 3rd, 0th, and 6th rows from the array
~~~
**Reshaping Arrays:** To reshape an existing array:
~~~
arr = np.arange(10)
arr = arr.reshape((2, 5)) # Returns a 2x5 array
~~~
**Transposing Arrays and Swapping Axes:** Arrays have the `transpose` method and also the special `T` attribute.:
~~~
arr = np.arange(15).reshape((3, 5))
arr.T # Returns a 5x3 array
 # OR
arr.transpose() # Also returns a 5x3 array
~~~
* Simple transposing with `.T` is just a special case of swapping aces.

**Universal Functions: Fast Element-wise Array Functions:** A universal function, or `ufunc`, is a function that performs elementwise operations on data in ndarrays. Examples:
~~~
arr = np.arange(10)
np.sqrt(arr) # Returns an array with the square root of each element in arr
np.add(arr, 1) # Returns an array with each element in arr plus 1
~~~
| Function (unary)  | Description  |
|---|---|
| `abs, fabs`  | Compute the absolute value element-wise for integer, floating point, or complex vales. Use `fabs` as a faster alternative for non-complex valued data.   |
|  `sqrt` | Compute the square root of each element. Equivalent to arr ** 0.5  |
| `square`  | Compute the square of each element. Equivalent to arr ** 2  |
| `exp`  | Compute the exponent `e^x` of each element  |
| `log, log10, log2, log1p`  | Natural logarithm (base e), log base 10, log base 2, and log (1 + x), respectively.  |
| `sign`  | Compute the sign of each element: 1 (positive), 0 (zero), or -1 (negative)  |
| `ceil`  | Compute the ceiling of each element, i.e. the smallest integer greater than or equal to each element  |
| `floor`  | Compute the floor of each element, i.e. the largest integer less than or equal to each element  |
| `rint`  | Round elements to the nearest integer, preserving the dtype  |
| `modf`  | Return fractional and integral parts of array as separate array  |
| `isnan`  | Return a boolean array indicating whether each value is NaN (Not a number)  |
| `isfinite, isinf`  | Return boolean array indicating whether each element is finite or infinite, respectively.  |
| `cos, cosh, sin, sinh, tan ,tanh`  | Regular and hyperbolic trigonometric functions  |
| `arccos, arcosh, arcsin, arcsinh, arctan, arctanh`  | Inverse trigonometric functions  |
| `logical_not`  | Compute truth value of not x element-wise. Equivalent to -arr  |

| Function (binary)  | Description  |
|---|---|
| `add` | Add corresponding elements in arrays  |
| `subtract`  | Subtract elements in second array from first array  |
| `multiply`  | Multiply array elements  |
| `divide, floor_divide`  | Divide or floor divide (truncate the remainder)  |
| `power`  | Raise elements in first array to powers indicated in second array  |
| `maximum, fmax`  | Element-wise maximum. `fmax` ignores NaN  |
| `minimum, fmin`  | Element-wise minimum. 'fmin' ignores Nan |
| `mod`   | Element-wise modulus (remainder of division)  |
| `copysign`  | Copy sign of values in second argument to values in first argument  |
| `greater, greater_equal, less, less_equal, equal, not_equal`  | Perform element-wise comparison, yielding boolean array.  |
| `logical_and, logical_or, logical_xor`  | Compute element-wise truth value of logical operation.  |

**Expressing Conditional Logic as Array Operations with `np.where`:** The np.where function is a vectorized version of the ternary expression `x if condition else y`.
~~~
arr = np.random.randn(4, 4) # Creates 4x4 array w/ values from normal distribution
np.where(arr > 0, 2, -2) # If an element is greater than 0, then 2, else -2.
~~~
* Nested `where` expression (for more complicated logic):
~~~
np.where(cond1 & cond2, np.where(cond1, 1, np.where(cond2, 2, 3)))
~~~

**Mathematical and Statistical Methods:** A set of mathematical functions which compute statistics about an entire array or about the data along an axis are accessible as array methods.
* Aggregations (often called *reductions*) like `sum`, `mean`, and `std` can either be used by calling the array instance method or using the top-level NumPy function:
~~~
arr = np.random.randn(5, 4) # 5x4 array of normally-distributed data
arr.mean()
    # OR
np.mean(arr)
~~~
* Functions like `mean` and `sum` take an optional `axis` argument, which computes the statistic over the given axis.
~~~
arr.mean(axis=1) # Returns the mean of each row
arr.sum(axis=0)  # Returns the sum of each column
~~~

| Method  | Description  |
|---|---|
| `sum`  | Sum all elements in the array or along an axis.  |
| `mean`  | Arithmetic mean.  |
| `std, var`  | Standard deviation and variance, respectively, with optional degrees of freedom adjustment (default denominator n)  |
| `min, max`  | Minimum and maximum  |
| `argmin, argmax`  |  Indices of minimum and maximum elements, respectively. |
| `cumsum`  | Cumulative sum of elements starting from 0.  |
|  `cumprod` | Cumulative product of elements starting from 1.  |

**Methods for Boolean Arrays:** Boolean values are coerced to 1 (`True`) and 0 (`False`) in the above methods. Thus, `sum` is often used as a means of counting `True` values in a boolean array.
~~~
arr = np.random.randn(100)
(arr > 0).sum() # Returns the number of positive values
~~~
* There are two additional methods, `any` and `all`, useful especially for boolean arrays. `any` tests whether one or more values in an array is `True`, while `all` checks if every value is `True`.
~~~
bools = np.array([False, False, True, False])
bools.any() # Returns True because there is a True value in bools
bools.all() # Returns False because not all values in bools are True
~~~
* These methods also work with non-boolean arrays, where non-zero elements evaluate to True.

**Sorting:** Like Python's built-in list type, NumPy arrays can be sorted in-place using the `sort` method or not in-place with `sorted`:
~~~
arr = np.random.randn(8)
sorted(arr) # Returns a sorted version of arr
arr.sort()  # Sorts arr and does not return anything
~~~
* Multidimensional arrays can have each 1D section of values sorted in-place along an axis by passing the axis number to sort:
~~~
arr = np.random.randn(5, 3)
arr.sort(1) # Sorts arr by each row
~~~
* A quick-and-dirty way to compute the quantiles of an array is to sort it and select the value at a particular rank:
~~~
large_arr = np.random.randn(1000)
large_arr.sort()
larg_arr[int(0.05 * len(large_arr))] # 5th quantile
~~~

**Unique and Other Set Logic:** The most commonly used basic set operation in NumPy is `np.unique`, which returns the sorted unique values in an array:
~~~
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names) # Returns array(['Bob', 'Joe', 'Will'],
      dtype='|S4')
~~~

| Method  | Description  |
|---|---|
| `unique(x)`  | Compute the sorted, unique elements in x  |
| `intersect1d(x, y)`  | Compute the sorted, common elements in x and y  |
| `union1d(x, y)`  | Compute the sorted union of elements  |
| `in1d(x, y)`  | Compute a boolean array indicating whether each element of x is contained in y  |
| `setdiff1d(x, y)`  | Set difference, elements in x that are not in y  |
| `setxor1d(x, y)`  | Set symmetric differences; elements that are in either of the arrays, but not both  |

**Saving and Loading Text Files:** While the majority of loading and saving will be done through pandas, you can load files in NumPy via:
~~~
arr = np.loadtxt('array_ex.txt', delimiter=',')
~~~
* To save a numpy array:
~~~
np.savetxt('array_output.txt', arr)
~~~
**Linear Algebra:** To multiply two dimensional arrays:
~~~
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)
 # OR
np.dot(x, y)
~~~
* More linear algebra on Page 102

**Random Number Generators:** The `numpy.random` module supplements the built-in Python `random` with functions for efficiently generating whole arrays or sample values from many kinds of probability distributions.
* For example, you can get a 4 by 4 array of samples from the standard normal distribution using `normal`:
~~~
samples = np.random.normal(size=(4,4))
~~~
* Python's built-in `random`, by contrast, only samples one value at a time.

| Function  | Description  |
|---|---|
| `seed`  | Seed the random number generator  |
|  `permutation` | Return a random permutation of a sequence, or return a permuted range  |
| `shuffle`  | Randomly permute a sequence in place  |
|  `rand` | Draw samples from a uniform distribution  |
|  `randint` | Draw random integers from a given low-to-high range  |
| `randn`  | Draw samples from a normal distribution with mean 0 and standard deviation 1  |
| `binomial`  | Draw samples from a binomial distribution  |
| `normal`  | Draw samples from a a normal (Gaussian) distribution  |
| `beta`  | Draw samples from a beta distribution  |
|  `chisquare` | Draw samples form a chi-square distribution  |
|  `uniform` | Draw samples from a uniform [0, 1) distribution  |
|  `gamma` | Draw samples from a gamma distribution  |

---
#### Chapter 5: Getting Started with pandas
* pandas is built on top of numpy.
* Import conventions:
~~~
from pandas import Series, DataFrame
import pandas as pd
~~~
* The two main data structures in pandas:
    1) Series
    2) DataFrame

**Series:** One-dimensional array-like object containing an array of data and an associated array of data labels, called its *index*.
* The simplest Series is formed from only an array of data:
~~~
obj = Series([4, 7, -5, 3])
~~~
* To access the Series' values and index:
~~~
obj.values

obj.index
~~~
* To specify the index (it defaults to the integers 0 through N - 1):
~~~
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
~~~
* In contrast to Numpy arrays, you can use the values in the index when selecting or assigning to single values or a set of values:
~~~
obj2['a']
obj2[['c', 'a', 'd']]
obj2['a'] = 10
obj2[['c', 'a', 'd']] = 10
~~~
* Another way to think about a Series is a fixed-length, ordered dict, as it is a mapping of index to data values. It can be substituted into many functions that expect a dict
    * You can go from a Python dict to a Series by:
~~~
dictdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(dictdata)
~~~
* To update a Serie's index in place:
~~~
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
~~~
**DataFrame:** A DataFrame represents a tabular, spreadsheet-like data structure containing an ordered collection of columns, each of which can be a different value type (numeric, string, boolean, etc.).
* The DataFrame has both a row and a column index; it can be thought of as a dictionary of Series.
* DataFrame sorts columns alphabetically by default.
* Creating a DataFrame from a dict of equal-length lists of NumPy arrays:
~~~
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
~~~
* You can specify the sequence of the columns by:
~~~
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data, columns=['year', 'state', 'pop'])
~~~
* If you pass a column that isn't contained in the data, it will appear as `NaN`
* To retrieve a Series from a DataFrame:
~~~
frame['state']
 # OR
frame.state
~~~
* To assign to a Series in a DataFrame:
~~~
frame['state'] = 'Colorado'
~~~
* Assigning a column that doesn't exist will create a new column:
~~~
frame['capital'] = ['Columbus', 'Columbus', 'Columbus', 'Reno', 'Reno']
~~~
* To remove a column:
~~~
del frame['capital']
~~~
* To access the column names:
~~~
frame.columns
~~~
* To transpose a DataFrame:
~~~
frame.T
 # OR
frame.transpose
~~~
* To set a DataFrame's index or columns to a specific name:
~~~
frame.index.name = 'year'
frame.columns.name = 'state'
~~~
**Missing Data:** Pandas marks missing data or NA as `Nan` (Not a Number).
* The `isnull` `notnull` functions should be used to detect missing data:
~~~
dictdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(dictdata)
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(dictdata, index=states)
obj4

Out[45]:
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
dtype: float64

 # California does not have an accompanying value and is thus NaN.

pd.isnull(obj4)

Out[46]:
California     True
Ohio          False
Oregon        False
Texas         False
dtype: bool

pd.notnull(obj4)

Out[47]:
California    False
Ohio           True
Oregon         True
Texas          True
dtype: bool
~~~
**Reindexing:** Create a new object with the data conformed to a new index:
~~~
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
~~~
* Reindexing introduces missing values if any index values were not already present.
* To fill missing values when reindexing:
~~~
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')
~~~
* `ffill` forward fills (or carry) the values, `bfill` fills (or carry) the values backwards

**Dropping Entries from an Axis:**
* Example Data:

| |one | two | three | four           |
|-------|-----|-----|--------|--------|
| Ohio      |  0 |   1  |    2 |    3 |
| Colorado  |  4  |  5   |   6 |    7 |
| Utah       | 8 |   9 |    10 |   11 |
| New York  | 12 |  13 |    14 |   15 |

* Columns = Axis1, Rows = Axis0
    * drop defaults to `Axis=0`
* To drop a column:
~~~
data.drop('two', axis=1)
~~~
* To drop multiple columns:
~~~
data.drop(['one', 'two'], axis = 1)
~~~
* To drop a row:
~~~
data.drop('Colorado')
~~~
* To drop multiple rows:
~~~
data.drop(['Colorado', 'Utah'])
~~~

**Indexing, Selecting, and Filtering:**
* To select a row:
~~~
data['row_two', :]
~~~
* * To select multiple rows:
~~~
data[['row_two', 'row_three'], :]
~~~
* To select a column:
~~~
data[:, 'column_two']
~~~
* To select multiple columns:
~~~
data[:, ['column_two', 'column_three']]
~~~

**Sorting and Ranking:**
* To sort a DataFrame by the index:
~~~
df.sort_index() # In ascending order
df.sort_index(ascending=False) # In descending order
~~~
* to sort a DataFrame by the column names:
~~~
df.sort_index(axis=1) # In ascending order
df.sort_index(axis=1, ascending=False) # In descending order
~~~
* To sort a DataFrame by a column:
~~~
df.sort_values('column_name') # In ascending order
df.sort_values('column_name', ascending=False) # In descending order
~~~
* To sort a DataFrame by multiple columns:
~~~
df.sort_values(['column1, column2'])
~~~
* `Nan` will automatically be sorted to the end or the column or row.
* `.rank()` is similar to `.sort_values()` except it inserts the rank of the underlying value instead of the value itself.
    * By default `.rank()` breaks ties by assigning each group the mean rank (i.e. if two values are tied for 6th, they will each be ranked as 6.5)
* To rank a DataFrame by it's column values:
~~~
df.rank() # In ascending order
df.rank(ascending=False) # In descending order
~~~
* To rank a DataFrame by it's row values:
~~~
df.rank(axis=1) # In ascending order
df.rank(axis=1, ascending=False) # In descending order
~~~
* See page 132 for tie-breaking methods with `.rank()`

**Axis Indexes w/ Duplicate Values**
* To check if index values are unique:
~~~
df.index.is_unique
~~~

**Summarizing and Computing Descriptive Statistics:**
* Most pandas common mathematical and statistical methods call into two categories:
    1) reductions
    2) summary statistics
* To sum each column:
~~~
df.sum()
~~~
* To sum each row:
~~~
df.sum(axis=1)
~~~
* To get the cumulative sum of a column:
~~~
df.cumsum()
~~~
* To display descriptive statistics of a DataFrame:
~~~
df.describe()
~~~
* Table of Summary Statistics available in pandas:

|  Method | Description  |
|---|---|
| `count`  | Number of non-NA values  |
| `describe`  | Compute set of summary statistics for each DataFrame column  |
|  `min`, `max` | Compute minimum and maximum values  |
| `argmin`, `argmax`  | Compute index locations (integers) at which minimum or maximum value obtained  |
| `idxmin`, `idxmax`  | Compute index values at which minimum or maximum value obtained  |
| `quantile`  | Compute sample quantile ranging from 0 to 1  |
|  `sum` | Sum of values  |
| `mean`  | Mean of values  |
| `median`  | Arithmetic median (50% quantile) of values |
|  `mad` | Mean absolute deviation from mean value  |
| `var`  | Sample variance of values  |
| `std`  | Sample standard deviation of values  |
| `skew`  | Sample skewness of values  |
|  `kurt` | Sample kurtosis of values  |
| `cumsum`  | Cumulative sum of values  |
| `cummin`, `cummax`  | Cumulative minimum or maximum of values  |
| `cumprod`  | Cumulative product of values  |
| `diff`  | Compute arithmetic difference  |
| `pct_change` | Compute percent changes |

**Correlation and Covariance**
* `corr` computes the correlation of the overlapping aligned-by-index values in two Series:
~~~
returns.MSFT.corr(returns.IBM)
~~~
* `cov` computes the covariance of the overlapping aligned-by-index values in two Series:
~~~
returns.MSFT.cov(returns.IBM)
~~~
* When used with DataFrames, `corr` and `cov` return a matrix of correlations or covariances between all columns.
~~~
df.corr()
df.cov()
~~~
* To calculate the correlation or covariance between one column and the rest of the DataFrame:
~~~
df.corrwith(df.columns)
~~~

**Unique Values, Value Counts, and Membership**
* To get the unique values of a column:
~~~
df.column.unique()
~~~
* To get the count of each unique value in a column:
~~~
df.column.value_counts()
~~~
* To test membership of an item in a column:
~~~
df.column.isin(['item'])
~~~

**Handling Missing Data:**
* Pandas uses the floating point value `Nan` (not a number) to represent missing data in both floating as well as in non-floating point arrays.
* The built-in Python `None` value is also treated as NA in object arrays.
* To see missing data in a DataFrame:
~~~
df.isnull()
~~~
* To count the number of missing data:
~~~
df.isnull().sum()
~~~
* To see non-missing data:
~~~
df.notnull()
~~~
* To count the number of non-missing data:
~~~
df.notnull().sum()
~~~

**Filtering Out Missing Data:**
* To drop missing data in a Series:
~~~
series.dropna()
~~~
* With DataFrames, `dropna()` defaults to dropping any row containing a missing value.
* To drop all rows with any missing data:
~~~
df.dropna()
~~~
* To drop all rows with missing data in all columns:
~~~
df.dropna(how='all')
~~~
* To drop all columns with any missing data:
~~~
df.dropna(axis=1)
~~~
* To drop all columns with missing data in all rows:
~~~
df.dropna(how='all', axis=1)
~~~

**Filling in Missing Data:**
* To replace all missing values with a specific value:
~~~
df.fillna(value)
~~~
* To replace all missing values with a separate value for each column:
~~~
df.fillna({row1: 0, row2:1})
~~~
* Add the `inplace=True` argument to modify the underlying DataFrame.
* To forward fill missing values within a row:
~~~
df.fillna(method='ffill')
~~~
* To replace all missing values with the mean of each row:
~~~
df.fillna(df.mean())
~~~

**Hierarchical Indexing:**
* Hierarchical indexing enables you to have multiple index levels on an axis, which lets you work with higher dimensional data in a lower dimensional form.
* Hierarchical indexes also allow for partial indexing.
* With DataFrames, either axis can have a hierarchical index.
* Examples on pages 144-145

**Pandas Frequent Functions:**
| Pandas Function  | What It Does  |
|---|---|
| `import pandas as pd`  | imports pandas as pd  |
| `df=pd.read_csv('path-to-file.csv')`  | load data into pandas  |
| `df.head(5)` | prints the first n lines; in this case 5 lines |
| `df.index` | prints the index of your dataframe |
|`df.columns` | prints the columns of your dataframe |
| `df.set_index('col')` | make the index (aka row names) the values of col |
| `df.reset_index()` | reset index |
| `df.columns = ['new name1', 'new name2']` | rename cols |
| `df = df.rename(columns={'old name 1':'new name 1'})` | rename specific col |
| `df['col']` | selects one column |
| `df[['col1', 'col2']]` | select more than one col |
| `df['col'] = 1` | set the entire col to equal 1 |
| `df['empty col'] = np.nan` | make an empty column |
| `df['col3'] = df['col1'] + df['col2']` | create a new col, equal to the sum of other cols |
| `df.loc['row 0']` or `df.iloc[0]` | select row 0 |
| `df.loc['row 5': 'row 100']` or `df.iloc[5:100]`  | select rows 5 through 100 |
| `df.loc[[2,4,6,8]]` | select rows 2, 4, 6, 8 |
| `df.loc[0]['col']` | select row and column, retrieve cell value |
| `del df['col']` | delete or drop or remove a column |
| `df.drop('col', axis=`1`)` | delete or drop or remove a column |
| `df.drop('row')` | delete or drop or remove a row |
| `df = df.sort_values(by='col')` | sort dataframe on this column |
| `df.sort_values(by=['col', 'col2'])` | sort data by col, then col2 |
| `solo_col=df['col']` | make a variable that is equal the col n|
| `just_values = df['col'].values` | returns an array with just the values, NO INDEX |
| `df[(df['col']=='condition')]` | return df when col is equal to condition |
| `df['col'][(df['col1'] == 'this') & (df['col2'...])]` | select col1 to new value when col1 == this, and c... |
| `df.groupby('col').sum()` | group by groupby a column and SUM all other |
| `df.plot(kind='bar')**king='bar' or 'line'` | make a bar plot |
| `alist = df['cols'].values` | extract just the values of a column into a list |
| `a_matrix=df.as_matrix()` | extract just the values of a whole dataframe |
| `df.sort(axis=1)` | sort by column names |
| `df.sort('col', axis=0)` | will sort by the 'col' column in ascending order |
| `df.sort('col', axis=0, ascending=True)` | will sort by 'col' column in descending order |
| `df.sort(['col-1', 'col-b'], axis=0)` | sort by more than one column |
| `df.sort_index()` | this will sort the index |
| `df.rank()` | it keeps your df in order, but ranks them |
| `df = pd.DataFrame({'col-a: a list', 'col-b':...})` | how to put or how to insert a list into a dataframe |
| `df.dtypes` | will print out the type of value in each column |
| `df['float-col'].astype(np.int)` | will change column data type |
| `joined = dfone.join(dftwo)` | join two dataframes if the keys are in the index |
| `merged = pd.merge(dfone, dftwo, one='key col')` | merge two dataframes on a similar column |
| `pd. concat([dfone, dftwo, series3])` | append data to the end of the dataframe |

**Navigating the Pandas DataFrame:**
`df['colname']`: selects single column or sequence of columns from the DataFrame.
`df.loc[val]` or `df.iloc[val]`: selects single row of subset of rows from the DataFrame.
`df.loc[:, val]` or `df.iloc[:, val]`: selects single column of subset of rows.
`df.loc[val1, val2]` or `df.iloc[val1, val2]`: select both rows and columns.
`df.reset_index()`: conform one or more axes to new indexes
`df.xs()`: select single row or column as a Series by label

**Counting with Pandas versus Counting with Pandas:**
* Counting wihtout Pandas:
~~~
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            count[x] += 1
        else:
            counts[x] = 1
    return counts
~~~
OR
~~~
from collections import defaultdict

def get_counts2(sequence):
    counts = defaultdict(int) # values will initialize to 0
    for x in sequence:
        counts[x] += 1
    return counts
~~~
OR
~~~
from collections import Counter
Counter(sequence)
~~~
* Counting with Pandas:
~~~
import pandas as pd
data = pd.read_csv('2017_Season.csv')
counts = data['Tm'].value_counts()   
~~~

---
#### Chapter 6: Data Loading, Storage, and File Formats
* Input and output typically falls into three main categories:
1) Reading text files and other more efficient on-disk formats
2) Loading data from databases
3) Interacting with network sources like web APIs

**Reading and Writing Data in Text Format:**
* The two most used function for reading tabular data as a DataFrame object are:
1) `pd.read_csv()`
2) `pd.read_table()`
* Functions for reading in tabular data:

| Function  | Description  |
|---|---|
| `pd.read_csv()`  | Load delimited data from a file, URL, or file-like object. Use comma as default delimiter.  |
| `pd.read_table()`  | Load delimited data from a file, URL, or file-like object. Use tab ('\t') as default delimiter. |
| `pd.read_fwf()`  | Read data in fixed-width column format (that is, no delimiter)  |
| `pd.read_clipboard()`  | Version of `read_table` that reads data from the clipboard. Useful for converting tables from web pages.   |

* To import from a .csv file with header names:
~~~
df = pd.read_csv('filename.csv')
 #OR
df = pd.read_table('filename.csv', sep=',')
~~~
* To import a .csv without header names:
~~~
df = pd.read_csv('filename.csv', header=None) # Pandas will assign default column names
 #OR
df = pd.read_csv('filename.csv', names=['column1', 'column2', 'column3'])
~~~
* To specify a column as the index:
~~~
df = pd.read_csv('filename.csv', names=['column1', 'column2', 'column3'], index_col='column1')
~~~
* To read a fixed-delimiter file (like .txt):
~~~
df = pd.read_table('filename.txt', sep='value')
~~~
* To skip certain rows during import:
~~~
df = pd.read_csv('filename.csv', skiprow=[0, 2, 3])
~~~
* Create a dataframe from your own clipboard:
~~~
df = pd.read_clipboard()
~~~
* Import arguments:

| Argument  | Description  |
|---|---|
| `path`  | String indicating filesystem location, URL, or file-like object.  |
| `sep` or `delimiter`  | Character sequence or regular expression to use to split fields in each row.  |
| `header`  | Row number to use as column names. Defaults to 0 (first row), but should be None if there is no header row.  |
| `index_col`  | Column numbers or names to use as the row index in the result. Can be a single name/number or a list of them for a hierarchical index.  |
| `names`  | List of column names for result, combine with header=None   |
| `skiprows`  | Number of rows at beginning of file to ignore or list of row numbers (starting from 0) to skip  |
| `na_values`  | Sequence of values to replace with NA  |
| `comment`  | Character or characters to split comments off the end of lines.  |
| `parse_dates`  | Attempt to parse data to datetime; False by default.   |
| `keep_date_col`  | If joining columns to parse date, keep the joined columns. Default False.  |
| `converters`  | Dict containing column number of name mapping to function. For example: {'foo': f} would apply the function f to all values int he 'foo' column.  |
| `dayfirst`  | When parsing potentially ambiguous dates, treat as international format (e.g. 7/6/2012 -> June 7, 2012). Default False.  |
| `date_parser`  | Function to use to parse dates.  |
| `nrows`  | Number of rows to read from beginning of file. |
| `iterator`  | Return a TextParser object for reading file piecemeal  |
| `chunksize`  | For iteration, size of file chunk.  |
| `skip_footer`  | Number of lines to ignore at each of file.  |
| `verbose`  | Print various parser output information, like the number of missing values placed in non-numeric columns.  |
| `encoding`  | Text encoding for unicode. For example, 'utf-8' for UTF-8 encoded text.  |
| `squeeze`  | If the parsed data only contains one column return a Series  |
| `thousands`  | Separator for thousands, e.g. ',' or '.'  |

**Reading Text Files in Pieces:**
* If you want to only read out a small number of rows (avoiding reading the entire file), specify that with `nrows`:
~~~
pd.read_csv('file_name.csv', nrows=100)
~~~
* To split the file into iterable chunks:
~~~
chunks = pd.read_csv('ex6.csv', chunksize=1000)
~~~

**Writing Data Out to Text Format:**
* To write to a .csv file:
~~~
df.to_csv('file_name.csv')
~~~
* To write to a .csv with other seperators:
~~~
df.to_csv('file_name.csv', sep='|')
~~~
* To write a subset of the DataFrame to a .csv file:
~~~
df.to_csv('file_name.csv', cols=['column1', 'column2', 'column3'])
~~~

**Manually Working with Delimited Formats:**
* Pages 161-163

**JSON Data:**
* JSON is short for JavaSCript Object Notation and has become one of the standard formats for sending data by HTTP request between web browsers and other applications.
    * It is a much more flexible data format than a tabular text for like .csv.
* To convert a JSON string to Python:
~~~
import json
result = json.loads(obj)
~~~
* To convert a Python object to JSON:
~~~
import json
asjason = json.dumps(result)
~~~
* To import a JSON file from the web:
~~~
import json
path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
record = [json.loads(line) for line in open(path, 'rb')]
~~~

**XML and HTML: Web Scraping:**
* Many websites make data available in HTML tables for viewing in a browser, but not downloadable as an easily machine-readable format like JSON.
* XML (extensible markup language) is another common structured data format supporting hierarchical, nested data with metadata.

**Reading Microsoft Excel Files:**
xls_file = pd.ExcelFile('data.xls')
table = xls_file.parse('Sheet1')

---
#### Chapter 7: Data Wrangling: Clean, Transform, Merge, Reshape
**Combining and Merging Data Sets:**
* Data contained in pandas objects can be combined together in three separate ways:
    1) `pandas.merge`: connects rows in DataFrames based on one or more keys. This will be familiar to users of SQL or other relational databases, as it implements database join operations.
    2) `pandas.concat`: glues or stacks together objects along an axis.
    3) `combine_first`: instance method enables splicing together overlapping data to fill in missing values in one object with values from another.

**Database-style DataFrame Merges:**
* To merge two databases on a shared column:
~~~
pd.merge(df1, df2, on='column')
~~~
* By default `merge` does an inner join; with the keys in the result are the intersection.
    * Other possible options are 'left', 'right', 'outer'.
        * The outer join takes the union of the keys, combining the effect of applying both left and right joins.
* To change the join type:
~~~
pd.merge(df1, df2, how='outer')
~~~
* To merge with multiple keys, pass a list of column names:
~~~
pd.merge(df1, df2, on=['key1', 'key2'], how='outer')
~~~
* If the two DataFrames you are attempting to merge have overlapping column names, pandas will automatically add on `_x` and `_y` to the end of the two column names.
* To manually specify new names to overlapping columns:
~~~
pd.merge(left, right, on='key', suffixes('_left', '_right'))
~~~
* Merge Argument References:

| Argument  | Description   |
|---|---|
| `left`  | DataFrame to be merged on the left side   |
| `right`  | DataFrame to be merged on the right side  |
| `how`  | One of the `inner`, `outer`, `left`, or `right`. `inner` is by default  |
| `on`  | Column names to join on. Must be found in both DataFrame objects. If not specified and no other join keys given, will use the intersection of the column names in `left` and `right` as the join keys.   |
| `left_on`  | Columns in `left` DataFrame to use as join keys.  |
| `right_on`  | Columns in `right` DataFrame to use as join keys.  |
| `left_index`  | Use row index in `left` as its join key   |
| `right_index`  | Use row index in `right` as its join key  |
| `sort`  | Sort merged data lexicographically by join keys; True by default.   |
| `suffixes`  | Tuple of string values to append to column names in case of overlap; defaults to ('_x', '_y'). For example, if 'data' in both DataFrame objects, would appear as 'data_x' and 'data_y' in result.  |
| `copy` | If False, avoid copying data into resulting data structures in some exceptional cases. By default always copies.|

* To use indexes as the merge key:
~~~
pd.merge(left1, right1, left_on='key', right_index=True)
~~~
* To merge two or more DataFrames with similar indexes:
~~~
pd.join(df1, df2, df3, how='outer' )
~~~
* To do a simple left join:
~~~
df1.join(df2, on='key')
~~~

**Concatenating Along an Axis:**
* To concatenate two numpy arrays together by column (results in a fatter array):
~~~
np.concatenate([arr1, arr2], axis=1)
~~~
* To concatenate two numpy arrays together by row (results in a longer array):
~~~
np.concatenate([arr1, arr2], axis=0) # axis defaults to zero
~~~
* To concatenate multiple DataFrames together by columns (results in a fatter DataFrame):
~~~
pd.concat([df1, df2, df3], axis=1)
~~~
* To concatenate multiple DataFrames together by row (results in a longer DataFrame):
~~~
pd.concat([df1, df2, df3], axis=0) # axis defaults to zero
~~~
* To concatenate multiple DataFrames together by columns using an inner join (only the columns that are present in each DataFrame):
~~~
pd.concat([df1, df2], axis=1, join='inner')
~~~
* Concat Function Arguments:

| Argument  | Description  |
|---|---|
| `objs`  | List or dict of pandas objects to be concatenated. This is the only required argument.   |
| `axis`  | Axis to concatenate along; defaults to 0 (rows).   |
| `join`  | One of 'inner' or 'outer'; defaults to 'outer'. Whether to intersection (inner) or union (outer) together indexes along the other axes.  |
| `join_axes`  | Specific indexes to use for the other n-1 axes instead of performing union/intersection logic.  |
| `keys`  | Values to associate with objects being concatenated, forming a hierarchical index along the concatenation axis. Can either be a list of array if arbitrary values, an array of tuples, or a list of arrays (if multiple level arrays passed in `levels`).  |
| `levels`  | Specific indexes to use as hierarchical index level or levels if keys passed.  |
| `names`  | Names for created hierarchical levels if `keys` or `levels` passed.   |
| `verify_integrity`  | Check new axis in concatenated object for duplicates and raise exception if so. By default (False) allows duplicates.  |
| `ignore_index`  | Do not preserve indexes along concatenation `axis`, instead producing a new `range(total_length)` index.  |

**Reshaping with Hierarchical Indexing:**
* Hierarchical indexing provides a consistent way to rearrange data in a DataFrame in two ways:
    1) `stack`: this "rotates" or pivots from the columns in the data to the rows.
    2) `unstack`: this pivots from the rows into the columns.
* To stack the data, pivoting the columns into the rows:
~~~
df.stack()
~~~
* To unstack a DataFrame, pivoting the rows into the columns:
~~~
df.unstck()
~~~
* Stacking filters out missing data by default.

**Pivoting "long" to "wide" Format:**
* Great example on Pages 190-191

**Removing Duplicates:**
* To see which rows in your DataFrame are duplicates:
~~~
df.duplicated() # Returns boolean True/False
~~~
* To remove duplicate rows in your DataFrame:
~~~
df.drop_duplicates()
~~~
* To drop duplicate rows that have duplicates in a specific row:
~~~
df.drop_duplicates(['column'])
~~~
* To drop duplicate rows that have duplicates in specific rows:
~~~
df.drop_duplicates(['column1', 'column2'])
~~~
* By default `drop_duplicates` keeps the first observed duplicate row. To instead keep the last duplicate row:
~~~
df.drop_duplicates(take_late=True)
~~~

**Replacing Values:**
* To replace a specific value within a column with another specific value:
~~~
df['column'].replace(old_value, new_value)
~~~
* To replace multiple values within a column with a specific value:
~~~
df['column'].replace([old_value1, old_value2], new_value)
~~~
* To replace multiple values within a column with multiple specific values:
~~~
df['column'].replace({old_value1: new_value1, old_value2: new_value2})
~~~

**Renaming Axis Indexes:**
* To transform an axis index via a function:
~~~
df.index = df.index.map(f)
 #OR
df.rename(index=str.upper, inplace=True)
~~~
* To rename certain axis values:
~~~
data.rename(index={'old_value1': 'new_value1', 'old_value2': 'new_value2'}, inplace=True)
~~~

**Discretization and Binning:**
* To bin column values, in this case ages 18-25, 26-35, 36-60, 61-100:
~~~
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
~~~
* To bin column values into named bins:
~~~
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
group_names = ['Youth', 'Young_Adult', 'Middle_Aged', 'Senior']
cats = pd.cut(ages, bins, labels=group_names)
~~~
* To bin column values into n equal-length bins (based on the minimum and maximum vales in the data).
~~~
pd.cut(column, number_of_bins)
~~~
* To bin column values into n equally-distributed bins (based on quantiles):
~~~
pd.qcut(column, 4) # Cut into quartiles
~~~

**Detecting and Filtering Outlier:**
* To remove any value in a column that is greater than or equal to a specific number. For example, to remove any value that is outside of +3 or -3 standard deviations in a column:
~~~
from scipy import stats
df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
~~~

**Permutation and Random Sampling:**
* Randomly reordering (permuting) a Series or the rows in a DataFrame can be done by using `np.random.permutation`:
~~~
df.take(np.random.permutation(len(df)))
~~~
* To select a random subset of a DataFrame without replacement, slice off the first k-elements of the array returned by `permutation`, where k is the desired subset:
~~~
df.take(np.random.permutation(len(df)))[:k]
~~~

**Computing Indicator/Dummy Variables:**
* To convert a categorical variable into a "dummy" or "indicator" matrix:
~~~
pd.get_dummies(df['column'])
~~~
* To add a prefix to the dummy columns:
~~~
pd.get_dummies(df['column'], prefix='dummy_')
~~~
* To add the dummies to the DataFrame:
~~~
pd.loc[:, ['column']].join(dummies)
~~~

**String Object Methods:**
* `split` is often combined with `strip` to trim whitespace and break strings into separate objects:
~~~
val = 'a, b,     guido'
pieces = [x.strip() for x in val.split(',')]
~~~
* To join separate strings with a specific delimiter (or space):
~~~
a = 'hello'
b = 'world'
c = '!'
answer = ' '.join([a, b, c])
~~~
* To check if a substring is in a string:
~~~
a = 'hello'
sentence = 'hello world!'
a in sentence # Returns True
~~~
* To find the index of a substring with a string
~~~
a = 'hello'
sentence = 'hello world!'
sentence.index(a) # returns 0. If the subset is not in the sentence .index() will return a ValueError.
 #OR
sentence.find(a) # returns 0. If the subset is not in the sentence .find() will return -1.
~~~
* To count the number of subsets within a string:
~~~
a = 'hello'
sentence = 'hello world hello'
sentence.count(a) # returns 2
~~~
* To substitute occurrences of one patter for another:
~~~
a = 'hello'
sentence = 'hello world!'
sentence.replace('hello', 'goodbye') # returns 'goodbye world!'
~~~
* You can use .replace() to also delete patterns by passing in an empty string:
~~~
a = 'cruel'
sentence = 'hello cruel world!'
sentence.replace(a, '') # returns 'hello world!'
~~~
* Python built-in string methods:

| Argument  | Description  |
|---|---|
| `count`  | Return the number of non-overlapping occurrences of substring in the string.  |
| `endswith`, `startswith`  | Returns True if string ends with suffix (starts with prefix.)  |
| `join`  | Use string as delimiter for concatenating a sequence of other strings.  |
| `index`  | Return position of first character in substring if found in the string. Raises `ValueError` if not found.  |
| `find`  | Return position of first character of *first* occurrence of substring. Like index, but returns -1 if not found.   |
| `rfind`  | Return position of first character of last occurrence of substring in the string. Returns -1 if not found.  |
| `replace`  | Replace occurrences of string with another string.  |
| `strip`, `rstrip`, `lstrip` | Trim whitespace, including newlines; equivalent to x.strip() (and rstrip, lstrip, respectively) for each element.  |
| `split`  | Break string into substrings using passed delimiter.  |
| `lower`, `upper`  | Convert alphabet characters to lowercase or uppercase, respectively.  |
| `ljust`, `rjust` | Left justify or right justify, respectively. Pad opposite side of string with spaces (or some other fill character) to return a string with a minimum width. |

**Regular Expressions:**
* Regular expressions provide a flexible way to search or match string patterns in text.
* A regex is a single expression formed according to the regular expression language.
* Python's built-in `re` module is responsible for applying regular expressions to strings.
    * The `re` module functions fall into three categories:
        1) Pattern matching
        2) Substitution
        3) Splitting
* To split a string with a variable number of whitespace characters (tab, spaces, and newlines):
~~~
import re
text = "foo     bar\t baz    \tquax"
re.split('\s+', text) # 's\+' is the regex describing one or more whitespace characters.
~~~
* To get a list of all patterns matching the regex:
~~~
import re
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

text = """Dave dave@google.com
     Steve steve@gmail.com
     Rob rob@gmail.com
     Ryan ryan@yahoo.com"""

regex = re.compile(pattern, flags=re.IGNORECASE)
regex.findall(text)
~~~
* To segment a pattern into separate parts, for example to identify an email address and then subset its three components (username, domain name, domain suffix):
~~~
import re
pattern = r'([A-Z0-9.+%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
m =regex.match('wesm@bright.net')
m.groups()
~~~
* Regular Expression Methods:

| Argument  | Description  |
|---|---|
| `findall`, `finditer`  | Return all non-overlapping matching patterns in a string. `finall` returns a list of all patterns while `finditer` returns them one by one from an iterator.  |
| `match`  | Match pattern at start of string and optionally segment pattern components into groups. If the pattern matches, returns a match object, otherwise None.  |
| `search`  | Scan string for match to pattern; returning a match object if so. Unlike `match`, the match can be anywhere in the string as opposed to only at the beginning.  |
| `split`  | Break string into pieces at each occurrence of pattern.  |
| `sub`, `subn` | Replace all sub or first n occurrences of subn of pattern in string with replacement expression. Use symbols \1, \2, ... to refer to match group elements in the replacement string.  |

#### Chapter 8: Plotting and Visualization
* When using matplotlib in Ipython be sure to start Ipython in the following manner:
~~~
ipython --pylab
~~~
* To close a plot window in ipython:
~~~
close()
~~~
* To import convention for matplotlib is:
~~~
import matplotlib.pyplot as plt
~~~
* To use fivethirtyeight's plotting sytle:
~~~
plt.style.use('fivethirtyeight')
~~~
* To create a blank figure:
~~~
fig = plt.figure()
~~~
* To create a single black graph:
~~~
fig, axes = plt.subplots()
~~~
* To create a blank figure with for subplots:
~~~
fig, axes = plt.subplots(2, 2)
~~~
* plt.subplots options:

| Argument  | Description  |
|---|---|
| `nrows`  | Number of rows of subplots  |
| `ncols`  | Number of columns of subplots  |
| `sharex`  | All subplots should use the same X-axis ticks (adjusting the `xlim` will affect all subplots)  |
| `sharey`  | All subplots should use the same Y-axis ticks (adjusting the `ylim` will affect all subplots)  |
| `subplot_kw`  | Dict of keywords passed to `add_subplot` call used to create each subplot  |
| `**fig_kw`  | Additional keywords to `subplots` are used when creating the figure, such as `plt.subplots(2, 2, figsize=(8,6))`  |

**Adjusting the spacing around subplots:**
* By default matplotlib leaves a certain amount of padding around the outside of the subplots and spacing between subplots.
* The spacing can be most easily changed using the `subplots_adjust` method:
~~~
subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
~~~
* `wspace` and `hspace` control the percent of the figure width and figure height, respectively, to use as spacing between subplots.
* matplotlib doesn't check whether axis labels overlap, so you will need to fix the labels yourself on many occasions.

**Colors, Markers, and Line Styles:**
* matplotlib's main `plot` function accepts arrays of X and Y coordinates and optionally a string abbreviation indicating color and line style.
    * For example, to plot x versus y with green dashes:
~~~
x = np.arange(10)
y = np.arange(10)
plt.plot(x, y, 'g--')
~~~
* The same plot could be expressed more explicitly:
~~~
x = np.arange(10)
y = np.arange(10)
plt.plot(x, y, linestyle='--', color='g')
~~~
* To add markers to highlight the actual data points:
~~~
plot(randn(30).cumsum(), color='k', linestyle='dashed', marker='o')
~~~

**Ticks, Labels, and Legends:**
* To manually set the x-axis ticks:
~~~
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
~~~
* To manually set the x-axis labels:
~~~
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
~~~
* To give a name to the X axis:
~~~
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())
ax.set_xlabel('Stages')
~~~
* To add a title to the plot:
~~~
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())
ax.set_title('My first matplotlib plot')
~~~
* To manually set the y-axis ticks:
~~~
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())
ticks = ax.set_yticks([0, 10, 20, 30])
~~~
* To manually set the y-axis labels:
~~~
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())
labels = ax.set_yticklabels(['one', 'two', 'three'], fontsize='small')
~~~

**Adding Legends:**
* To create a legend:
~~~
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum(), 'k', label='one')
ax.plot(randn(1000).cumsum(), 'k--', label='two')
ax.plot(randn(1000).cumsum(), 'k.', label='three')
ax.legend(loc='best')
~~~
* The `loc` argument tells matplotlib where to place the legend. If you aren't picky `'best'` is a good option, as it will choose a locaiton that is most out of the way.
* To exclude one or more of the elements from the legend, pass `_nolegend_` to the label argument as we've done with the second element below:
~~~
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum(), 'k', label='one')
ax.plot(randn(1000).cumsum(), 'k--', label='_nolegend_')
ax.plot(randn(1000).cumsum(), 'k.', label='three')
ax.legend(loc='best')
~~~

**Annotations and Drawing on a Subplot:**
* To add an annotation to a plot:
~~~
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.arange(10)
y = np.arange(10)
plt.plot(x, y, 'g--')
ax.text(5, 5, 'Hello world!', family='monospace', fontsize=20)
~~~
* The above will close 'Hello world!' at the coordinates (5, 5).
* Detailed example of annotations and arrows:
~~~
from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data = pd.read_csv('spx.csv', index_col=0, parse_dates=True)
spx = data['SPX']

spx.plot(ax=ax, style='k-')

crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]

for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 50),
    xytext=(date, spx.asof(date) + 200),
    arrowprops=dict(facecolor='black'),
    horizontalalignment='left', verticalalignment='top')

# Zoom in on 2007-2010
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])

ax.set_title('Important dates in 2008-2009 financial crisis')
~~~
* matplotlib's common shapes are called patches. General patches such as Rectange and Circle are found within matplotlib.pyplot, but the full set is located in matplotlib.patches. To add a patch to a plot:
~~~
fig =plt.figure()
ax = fig.add_subplot(1, 1, 1)

rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2,0.6]], color='g', alpha=0.5)

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
~~~

**Saving Plots to File:**
* The active figure can be saved to file using `plt.savefig`:
~~~
plt.savefig('plot1.png')
~~~
* The file type is inferred from the extension. So to save as .pdf:
~~~
plt.savefif('plot1.pdf')
~~~
* To increase the `dpi` (dots-per-inch) resolution as well as `bbox_inches`(tirm the whitespace around the figure):
~~~
plt.savefig('plot1.png`, dpi=400, bbox_inches='tight')
~~~
* Figure.savefig Options

| Argument  | Description  |
|---|---|
| `fname`  | String containing a filepath or a Python like object. The figure format is inferred from the file extension, e.g. .pdf for PDF or .pnf for PNG.  |
| `dpi`  | The figure resolution in dots per inch; defaults to 100 out of the box but can be configured.  |
| `facecolor`, `edgecolor`  | The color of the figure background of the subplots. 'w' (white), by default.  |
| `format`  | The explicit file format to use ('png', 'svg', 'ps', 'eps'...)  |
| `bbox_inches`  | The portion of the figure to save. If 'tight' is passed, will attempt to trim the empty space around the figure.  |

**matplotlib Configuration:**
* There are two ways to change matplotlib's configuration:
1) Programatically:
~~~
plt.rc('component to change', 'new parameter')
~~~
* An example:
~~~
plt.rc('figure', figzie=(10, 10))
~~~
2) Via the config file.

**Plotting Functions in pandas:**
* While matplotlib requires a lot of information to plot, pandas infers much of that information from the DataFrame. Thus, plotting from pandas only requires concise statements.
* Be default, pandas `plot` method defaults to line plots.
* To create a line plot of one column:
~~~
df['column'].plot()
~~~
* To create a line plot of all columns in the DataFrame:
~~~
df.plot()
~~~
* By default, the index is plotted as the x-axis. To disable this:
~~~
df.plot(use_index=False)
~~~
* Series.plot Method Arguments:

| Argument  | Description  |
|---|---|
| `label`  | Label for plot legend  |
| `ax`  | matplitlib subplot object to plot on. If nothing is passed, uses active matplotlib subplot.  |
| `style`  | Style string, like 'k--', to be passed to matplotlib  |
| `alpha`  | The plot fill opacity (from 0 to 1)  |
| `kind`  | Can be `line`, `bar`, `barh`, `kde`  |
| `logy`  | Use logarithmic scaling on the Y axis.  |
| `use_index`  | Use the object index for tick labels  |
| `rot`  | Rotation of tick labels (0 through 360)   |
| `xticks`  | Values to use for X axis ticks  |
| `yticks`  | Values to use for Y axis ticks  |
| `xlim`  | X axis limits (e.g. [0, 10])  |
| `ylim`  | Y axis limits  |
| `grid`  | Display axis grid (on by default)  |

* DataFrame.plot Method Arguments:

| Argument  | Description  |
|---|---|
| `subplots`  | Plot each DataFrame column in a separate subplot  |
| `sharex`  | If `subplots=True`, share the same X axis, linking ticks and limits  |
| `shary`  | If `subplots=True`, share the same Y axis  |
| `figsize`  | Size of figure to create as tuple   |
| `title`  | Plot title as string  |
| `legend`  | Add a subplot legend (True by default)  |
|  `sort_columns` | Plot columns in alphabetical order; by default uses existing column order.  |

**Bar Plots:**
* To create a vertical-bar plot:
~~~
df.plot(kind='bar')
~~~
* Top create a horizontal-bar plot:
~~~
df.plot(kind='hbar')
~~~
* Bar plots group the values in each row together in a group of bars, side by side, for each value.
* To create a stacked bar plot:
~~~
df.plot(king='bar', stacked=True)
~~~

**Histograms and Density Plots:**
* To plot a histogram:
~~~
df.plot.hist()
~~~

**Scatter Plots:**
* To create a scatter plot between two columns of a DataFrame:
~~~
plt.scatter(df['column1'], df['column2'])
~~~
* To create a scatter plot between all columns within a DataFrame:
~~~
pd.scatter_matrix(df)
~~~

#### Chapter 9: Data Aggregation and Group Operations
**Groupby Mechanics:**
* To compute the summary statistic of a column based on the value of a separate column:
~~~
df['column1'].groupby(df['column2']).mean()
~~~
* To compute summary statistics of each column in the DataFrame based on the value of a separate column:
~~~
df.groupby('column').mean()
~~~
* To compute summary statistics of each column in the DataFrame based on the value of multiple separate columns:
~~~
df.groupby(['column1', 'column2']).mean()
~~~
* A useful groupby method is `size`, which returns a series containing group sizes of the resulting groups:
~~~
df.groupby(['column1']).size()
~~~
* Missing values in a groupby are excluded from the result.

**Iterating Over Groups:**
* The groupby object supports iteration, which can be saved into a dictionary:
~~~
groupby_dict = dict(list(df.groupby('column')))
~~~
* By default, groupby groups on `axis=0`, but you can group on any of the other axes:
~~~
df.groupby('column', axis=1) # Not sure about this example. Check on this.
~~~

**Selecting a Column or Subset of Columns:**
* Indexing a groupby object created form a DataFrame with a column name or array of column names has the effect of selecting those columns for aggregation.
* This means that:
~~~
df.groupby('column2')['column1']
~~~
is equal to:
~~~
df['column1'].groupby(df['column2'])
~~~
* In terms of multiple column aggregation:
~~~
df.groupby(['column1', 'column2'])[['column3']].mean()
~~~

**Data Aggregation:**
* Groupby Methods:

| Function Name  | Description  |
|---|---|
| `count`  | Number of non-NA values in the group  |
| `sum`  | Sum of non-NA values  |
| `mean`  | Mean of non-NA values   |
| `median`  | Arithmetic median of non-NA values  |
| `std`, `var`  | Unbiased (n-1 denominator) standard deviation and variance  |
| `min`, `max`  | Minimum and maximum of non-NA values  |
| `prod`  | Product of non-NA values  |
| `first`, `last`  | First and last non-NA values  |

**Column-wise and Multiple Function Application:**
* To apply multiple aggregate functions to one column:
~~~
df.groupby('column').agg(['min', 'max', 'mean'])
~~~
* To apply multiple aggregate functions to multiple columns:
~~~
df.groupby(['column1', 'column2']).agg(['count', 'mean', 'max'])
~~~
* To apply different functions to one or more columns:
~~~
df.groupby(['column1', 'column2']).agg({'column3': 'sum', 'column4': 'mean'})
~~~
* To apply multiple different functions to one or more columns:
~~~
df.groupby(['column1', 'column2']).agg({'column3': ['min', 'max', 'mean'], 'column4': ['count', 'std']})
~~~

**Pivot Tables and Cross Tabulation:**
* A pivot table is a data summarization tool frequently found in spreadsheet programs like Excel.
    * It aggregates a table of data by one or more keys, arranging the data in a rectangle with some of the group keys along the rows and some along the columns.
* The following two commands are the same:
~~~
df.groupby(['column1', 'column2']).mean()
df.pivot_table(index=['column1', 'column2'])
~~~
* `pivot_table` defaults to use mean.
* To use different aggregation functions:
~~~
df.pivot_table(index=['column1', 'column2'], aggfunc=f)
~~~
* If some combinations are empty, or otherwise NA, pass `fill_value` argument:
~~~
df.pivot_table(index=['column1', 'column2'], aggfunc=f, fill_value=0)
~~~
* Pivot Table Options

| Function Name  | Description  |
|---|---|
| `values`  | Column name or names to aggregate. By default aggregates all numeric columns.  |
| `rows`  | Column names or other group keys to group on the rows of the resulting pivot table.  |
| `cols`  | Column names or other group keys to group on the columns of the resulting pivot table.  |
| `aggfunc`  | Aggregation function or list of functions; 'mean' by default. Can be any function valid in a groupby context.  |
| `fill_value`  | Replace missing values in result table.  |
| `margins`  | Add row/column subtotals and grant total. False by default.  |

**Cross-Tabulation: Crosstab:**
* A crosstab is a special case of a pivot table that computes group frequencies.
~~~
pd.crosstab('column1', 'column2', margins=True)
~~~

#### Chapter 10: Time Series
* Four types of time series:
1) Timestamps: specific specific instants in time.
2) Fixed periods: such as the month January 2007 or the full year 2010.
3) Intervals of time: indicated by a start and end timestamp. Periods can be thought of as special cases of intervals.
4) Experiment or elapsed time: each timestamp is a measure of time relative to a particular start time. For example, the diameter of a cookie baking each second since being placed in the oven.
* The simplest and most widely used kind of time stamp are those indexed by timestamp.

**Data and Time Data Types and Tools:**
* The most import modules for dates and times are:
1) `datetime`
    * Within `datetime` is a method `datetime`.
    * `from datetime import datetime`
2) `time`
3) `calendar`
* To get the current date and time:
~~~
from datetime import datetime
now = datetime.now()
np.year # Returns the year
np.month # Returns the month
np.day # Returns the day
~~~
* To get the difference between two dates and times:
~~~
datetime(2001, 1, 7) - datetime(2008, 6, 24, 8, 15)
~~~
* To add (or subtract) a timedelta to a datetime object:
~~~
from datetime import timedelta
start = datetime(2011, 1, 7) # January 7th, 2011
start + timedelta(12) # add 12 days
start - timedelta(12) # subtract 12 days
~~~
* datetime module:

| Type  | Description  |
|---|---|
| `date`  | Store calendar date (year, month, day) using the Gregorian calendar.  |
|  `time` | Store time of day as hours, minutes, seconds, and microseconds.  |
| `datetime`  | Stores both date and time.  |
| `timedelta`  | Represents the difference between two datetime values (as days, seconds, and microseconds)  |

**Converting between string and datetime:**
* To convert a datetime object to a string:
~~~
stamp = datetime(2011, 1, 3)
str(stamp)
 #OR
stamp.strftime('%Y-%m-d')
~~~
* To convert a string to datetime:
~~~
value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')
~~~
* To convert multiple strings to datetime:
~~~
datestrs = ['7/6/2011', '8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]
~~~
* `datetime.strptime` is the best way to parse a date with a known format.
* However, another option is to use `parse`:
~~~
from dateutil.parser import parse
parse('2011-01-03')
~~~
* `parse` is capable of parsing almost any human-intelligible data representation:
~~~
parse('Jan 31, 1997 10:45 PM')
 #OR
parse('6/11/1991')
~~~
* When dealing with international datetimes, where days appear before months:
~~~
parse('6/12/2011', dayfirst=True)
~~~
* To parse an entire columns, use `pd.to_datetime`:
~~~
pd.to_datetime('column')
~~~
* `to_datetime` changes all missing values to `NaT`
    * `NaT` (Not a Time) is panda's NA value for timestamp data
* Datetime format specifications:

| Type  | Description  |
|---|---|
| `%Y`  | 4-digit year  |
|  `%y`  | 2-digit year  |
| `%m`  | 2-digit month [01, 12]  |
| `%d`  | 2-digit day [01, 31]  |
| `%H`  | Hour (24-hour clock) [0, 23] |
| `%I`  | Hour (12-hour clock) [01, 12]  |
| `%M`  | 2-digit minute [00-59]  |
| `%S`  | Second [00, 61] (seconds 60, 61 account for leap seconds)  |
| `%w`  | Weekday as integer [0(Sunday), 6]  |
| `%U`  | Week number of the year [00, 53]. Sunday is considered the first day of the week, and days before the first Sunday of the year are "week 0"  |
| `%W`  | Week number of the year [00, 53]. Monday is considered the first day of the week, and days before the first Monday of the year are 'week 0.'  |
| `%z`  | UTC time zone offset as +HHMM or -HHMM, empty if time zone naive  |
| `%F`  | Shortcut for %Y-%m-%d, for example 2012-4-18  |
|  `%D` | Shortcut for %m/%d/%y, for example 04/18/12  |

* Local-specific date formatting:

| Type  | Description  |
|---|---|
| `%a`  | Abbreviated weekday name  |
| `%A`  | Full weekday name  |
| `%b`  | Abbreviated month name  |
| `%B`  | Full month name  |
| `%c`  | Fill date and time, for example 'Tue 01 May 2012 04:20:57 PM'  |
| `%p`  | Locale equivalent of AM or PM  |
| `%x`  | Locale-appropriate formatted date; e.g. in US May 1, 2012 yields '05/01/2012'  |
| `%X`  | Locale-appropriate time, e.g. '04:24:12 PM'  |

**Time Series Basics:**
* TimeSeries is a subclass of Series and thus behaves in many of the same ways:
* To create a TimeSeries with datetimes as the index and values in a separate columns:
~~~
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7), datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
~~~
* To index a TimeSeries:
~~~
ts[2] # index of the third timestamp
 #OR
ts[`01/10/2011`] # also index of the third timestamp
 #OR
ts['20110110'] # also index of the third timestamp
~~~
* For longer TimeSeries, you can index based on year:
~~~
long_ts['2001']
~~~
* For longer TimeSeries, you can also index based on month:
~~~
long_ts['2001-05']
~~~
* Slicing with dates works just like with a regular Series:
~~~
long_ts[datetime(2001, 1, 7):]
~~~

**Time Series with Duplicate Indices:**
* To check if the index of a timeseries is unique:
~~~
ts.index.is_unique()
~~~
* To see which items are duplicates:
~~~
grouped = ts.groupby(level=0)
groupbed.count()
~~~

**Data Ranges, Frequencies, and Shifting:**
* To create a fixed-frequency series (meaning same time interval difference between each record; such as every day) from a non-fixed frequency series:
~~~
ts.resample('D').asfreq()
~~~

**Generating Date Ranges:**
* To create a range of dates:
~~~
pd.date_range('4/1/2012', '6/1/2012')
~~~
* By default, `date_range` generates daily timestamps.
* To create a range of dates with just a start or end date:
~~~
pd.date_range(start='4/1/2012', periods=20)
~~~

**Frequencies and Data Offsets:**
* To create a date range based on a specified time frequency, in this case every four hours:
~~~
pd.date_range('1/1/2000', '1/3/2000', freq='4h')
 #OR
pd.date_range('1/1/2000', periods=10, freq='120min')
~~~
* To create a date range including every third Friday of every month:
~~~
pd.date_range('1/1/2000', periods=10, freq='WOM-3FRI')
~~~
* Entire `freq` options on Pages 296-297

**Shifting (Leading and Lagging) Data:**
* "Shifting" refers to moving data backward and forward through time.
* A common use of `shift` is computing percent changes in a time series or multiple time series as DataFrame columns:
~~~
ts/ts.shift(1) - 1
~~~

**Time Zone Handling:**
* In python, time zone information comes from the 3rd party `pytz` library.
* To get a time zone object from pytz:
~~~
tz = pytz.timezone('US/Mountain')
~~~
* By default, time series in pandas are time zone naive, meaning they don't have a time zone attached to them.
* To create a timeseries with a specified localized time zone:
~~~
pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC')
~~~
* To localize the time zone of an existing time series:
~~~
ts_utc = ts.tz_localize('UTC')
~~~
* Once a time series has been localized to a particular time zone, it can be converted to another time zone:
~~~
ts_mtn = ts_utc.tz_convert('US/Mountain')
~~~
* If two time series with different time zones are combined, the result will be UTC.

#### Chapter 12: Advanced Numpy
**Reshaping array:**
* To convert an array from one shape to another:
~~~
arr = np.arange(8)
arr.reshape((4, 2)) # Reshapes the array into two columns and four rows.
~~~
* To reshape a multidimensional array:
~~~
arr.reshape((4,2)).reshape((2,4))
~~~
* To reshape an array by specifying only the column or row dimension and letting numpy figure out the best configuration, pass -1 to the argument:
~~~
arr = np.arange(15)
arr.reshape((5,-1))
~~~
* To reshape an array to mirror the shape of another array:
~~~
arr = np.arange(15)
other_arr = np.ones((3,5))
arr.reshape(other_arr.shape)
~~~
* To flatten (or ravel) an array, meaning to go from a higher-dimension to a one-dimension array:
~~~
arr = np.arange(15).reshape((5,3))
arr.ravel() # ravel does not produce a copy of the underlying data.
 #OR
arr.flatten() # flatten always returns a copy of the data.
~~~

**Concatenating and Splitting Arrays:**
* `numpy.concatenate` takes a sequence (tuple, list, etc.) of arrays and joins them together in order along the input axis.
* To concatenate the long way (along the rows):
~~~
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
np.concatenate([arr1, arr2], axis=0)
~~~
* The above can also be done via `vstack`:
~~~
np.vstack((arr1, arr2))
~~~
* To concatenate the wide way (along the columns):
~~~
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
np.concatenate([arr1, arr2], axis=1)
~~~
* The above can also be done via `hstack`:
~~~
np.hstack((arr1, arr2))
~~~
* To split an array:
~~~
from numpy.random import randn
arr = randn(3,2)
first, second, third = np.split(arr, [1, 2])
~~~
* Array Concatenation Functions:

| Function  | Description  |
|---|---|
| `concatenate`  | Most general function, concatenates collection of arrays along one axis |
| `vstack`, `row_stack`  | Stack arrays row-wise (along axis 0)  |
| `hstack`  | Stack arrays column-wise (along axis 1)  |
| `column_stack`  | Like hstack, but converts 1D arrays to 2D column vectors first  |
| `dstack`  | Stack arrays 'depth'-wise (along axis 2) |
| `split`  | Split array at passed locations along a particular axis  |
| `hsplit`, `vsplit`, `dsplit`  | Convenience functions for splitting on axis 0, 1, and 2 respectively.   |

**Stacking helpers: r_ and c_:**
* Similar to `vstack`, `np.r_` stacks rows onto other rows:
~~~
arr1 = np.arange(6).reshape((3,2))
arr2 = randn(3,2)

np.r_[arr1, arr2]
~~~
* Similar to `hstack`, `np.c_` stacks rows onto other rows:
~~~
arr1 = np.arange(6).reshape((3,2))
arr2 = randn(3,2)

np.c_[arr1, arr2]
~~~
* These helpers can also translate slices to arrays:
~~~
np.c_[1:6, -10:-5]
~~~

**Repeating Elements: Tile and Repeat:**
* The two main tools for repeating and replicating arrays to produce larger arrays are the `repeat` and `tile` functions.
* The `repeat` function replicates each element in an array some number of times, producing a larger array:
~~~
arr = np.arange(3)
arr.repeat(3)
~~~
    * By default, if you pass an integer, each element of the original array will be repeated that number of times.
    * If you pass an array of integers, each element can be repeated a different number of times.
* Multidimensional arrays can have their elements repeated along a particular axis:
~~~
arr = randn(2, 2)
arr.repeat(2, axis=0)
~~~
* The `title` function is a shortcut for stacking copies of an array along an axis. Think of it as laying down tiles.
~~~
arr = randn(2,2)
np.tile(arr, 2)
~~~
* To lay them down the long way:
~~~
np.tile(arr, (2,1))
~~~

**Fancying Indexing Equivalents: Take and Put:**
* Fancy indexing is the practice of subsetting an array using integer arrays.
    * It is the same process as masking:
~~~
arr = np.arange(10)
arr[arr > 4]
~~~
* Numpy offers other ways of fancy indexing:
~~~
arr = np.arange(10)
inds = [7, 1, 2, 6]
arr.take(inds) # Returns an array of the 7th, 1st, 2nd, and 6th items of arr
~~~
* To take along another axes:
~~~
arr.take(inds, axis=1)
~~~
* To replace items in an array with fancy indexing:
~~~
arr = np.arange(10)
inds = [7, 1, 2, 6]
arr.put(inds, 69) # Returns the original array with the 7th, 1st, 2nd, and 6th items replaced with the number 69.
~~~

**Broadcasting:**
* Broadcasting describes how arithmetic works between arrays of different shapes.
* Simplest example of broadcasting occurs when combining a scalar values with an array:
~~~
arr = np.arange(5)
 # Output: array([0, 1, 2, 3, 4])
arr * 4
 # Output: array([ 0,  4,  8, 12, 16])
~~~
    * Here we say that the scalar value 4 has been broadcast to all of the other elements in the multiplication operation.
* To use broadcasting to subtract the mean from each item in a column:
~~~
arr = randn(4,3)
arr_demeaned = arr - arr.mean(0)
~~~
*The Broadcasting Rule: Two arrays are compatible for broadcasting if for each trailing dimension (that is, starting from the end), the axis lengths match or if either of the lengths is 1. Broadcasting is then performed over the missing and/or length 1 dimensions.*

**ufunc Instance Methods:**
* `reduce` takes a single array and aggregates its values, optionally along an axis.
* For example, an alternate way to sum elements in an array is to use np.add.reduce:
~~~
arr = np.arange(10)
np.add.reduce(arr) # Is the same as arr.sum()
~~~
* `accumulate` works similar to cumprod
~~~
arr = np.arange(15).reshape((3,5))
np.add.accumulate(arr, axis=1)
~~~
* ufunc methods:

| Method  | Description  |
|---|---|
| `reduce(x)`  | Aggregate values by successive applications of the operation  |
| `accumulate(x)`  | Aggregate values, preserving all partial aggregates  |
| `reduceat(x, bins)`  | 'Local' reduce or 'groupby'. Reduce contiguous slices of data to produce aggregated array.  |
| outer(x, y)  | Apply operation to all pairs of elements in x and y. Result array has shape x.shape + y.shape  |

**More About Sorting:**
* Like Python's built-in list, the ndarray `sort` instance method is an *in-place* sort, meaning that the array contents are rearranged without producing a new array:
~~~
arr = randn(6)
arr.sort()
~~~
* To sort based on the rows:
~~~
arr = randn(6)
arr.sort(axis=1)
~~~
* When sorting arrays in-place, remember that if the array is a view of a different ndarray, the original array will me modified.
* To avoid this, use `numpy.sort`, which creates a new, sorted copy of an array:
~~~
arr = randn(6)
np.sort(arr)
~~~
* To np.sort on the rows:
~~~
arr = randn(6)
np.sort(arr, axis=1)
~~~
* Neither arr.sort() or np.sort(arr) have a reverse option. To reverse the output:
~~~
arr.sort()[::-1]
np.sort(arr)[::-1]
~~~

**numpy.searchsorted: Finding elements in a Sorted Array:**
* `searchsorted` is an array method that performs a binary search on a sorted array, re-turning the location in the array where the value would need to be inserted to maintain sortedness:
~~~
arr = np.array([0, 1, 7, 12, 15])
arr.searchsort(9)
~~~
* You can also pass an array of values to get an array of indices back:
~~~
arr.searchsorted([0, 8, 11, 16])
~~~
* The default behavior is to return the index at the left side of a group of equal values:
    * To change this to the right side:
~~~
arr.searchsorted([0, 1], side='right')
~~~

#### Appendix: Python Language Essentials
**The Python Interpreter:**
* Python is an interpreted language:
* To run python from the console:
~~~
$ python
~~~
* To run a python script from the console:
~~~
$ python script.py
~~~
* To run ipython from the console:
~~~
$ ipython
~~~
* To run a script within ipython:
~~~
%run script.py
~~~

**Comments:**
* Any text preceded by the hash mark (pound sign) # is ignored by the Python interpreter.
~~~
# This is a comment
~~~
* To comment out an entire section, highlight it and then press Command + /

**Data Types:**
* To check if an object is an instance of a particular type:
~~~
a = 5
isinstance(a, int)
~~~
* To check if an object's type is among multiple types:
~~~
a = 5
isinstance(a, (int, float))
~~~

**Strings:**
* The backslash character `\` is an escape character, meaning that it is used to specify special characters like newline `\n` or unicode characters. To write a string literal with backslashes, you need to escape them:
~~~
s = '12\\34'
print s # Returns 12\34
~~~
* If you have a string with a lot of backslashes and no special characters, you might find this a bit annoying. Fortunately you can preface the leading quote of the string with `r` which means that the characters should be interpreted as is:
~~~
s = r'this\has\no\special\characters'
s # Returns 'this\\has\\no\\special\\characters'
~~~

**Booleans:**
* You can see exactly what boolean value an object coerces to by invoking `bool` on it:
~~~
bool([]) # False
bool('Hello World') # True
~~~

**Dates and Times:**
* The built-in Python `datetime` module provides `datetime`, `date`, and `time` types.
    * The `datetime` type as you may imagine combines the information stored in `date` and `time` and us the most commonly used.
* To create a datetime object:
~~~
from datetime import datetime, date, time
dt = datetime(2011, 10, 29, 20, 30, 21)
~~~
* To access various parts of a datetime object:
~~~
dt.year
dt.month
dt.day
dt.hour
dt.minute
dt.second
~~~
* To extract just the date part of a datetime object:
~~~
dt.date()
~~~
* To extract just the time part of a datetime object:
~~~
dt.time()
~~~
* To format a datetime as a string:
~~~
datetime.strftime(dt,'%m/%d/%Y')
~~~
* To convert (parse) a string to a datetime object:
~~~
datetime.strptime('20091031', '%Y%m%d')
~~~
* To replace certain parts of a datetime object:
~~~
dt.replace(minute=0, second=0)
~~~

**for loops:**
* A `for` loop can be advanced to the next iteration, skipping the remained of the block, using the `continue` keyword. For example, this code sums up integers in a list and skips `None` values:
~~~
sequence = [1, 2, None, 4, None, 5]
total = 0
for value in sequence:
    if value is None:
        continue
    total += value
~~~

**Exception Handling:**
* To handle exceptions:
~~~
def attempt_float(x):
    try:
        return float(x)
    except:
        return 'float() only accepts numerical values.'
~~~
* To handle a specific exception, for instance ValueError:
~~~
def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return 'float() only accepts numerical values'
~~~
* To handle more than one specific exception:
~~~
def attempt_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 'float(x) only accepts numerical values'
~~~
