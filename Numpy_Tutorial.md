### Numpy Tutorial
#### October 2017

---
#### Introduction
* numpy stands for numerical python
* The fundamental idea of numpy is support for multidimensional arrays. So numpy can be considered as the base for numerical computing in Python.

#### Installing numpy
* Python does not come bundled with numpy.
* To install numpy, run the following command in the command prompt:
~~~
pip install numpy
~~~
* To check that numpy was successfully installed, run the following within IPython:
~~~
import numpy
numpy.__version__
~~~

#### The ndarray Object
* The `ndarray` is a fundamental object of numpy.
    * This object is an N-dimensional array, meaning that it contains a collection of elements of the same type indexed using N (dimensions of the array) integers.
* The main attributes of `ndarray` are:
    * data type (`dtype`)
    * `shape`
    * `size`
    * `itemsize`
    * `data`
    * `ndim`
* Example:
~~~
my_array = np.array(((6, 12, 93, 2), (5, 26, 78, 90), (3, 12, 16, 22), (5, 3, 1, 16)))

my_array
array([[ 6, 12, 93,  2],
       [ 5, 26, 78, 90],
       [ 3, 12, 16, 22],
       [ 5,  3,  1, 16]])
~~~
* To return the data type of `my_array`:
~~~
my_array.dtype
~~~
* To return the shape of `my_array`:
~~~
my_array.shape
~~~
* The above will return a tuple of array dimensions (rows, columns).
* To return the size (number of elements) of `my_array`:
~~~
my_array.size
~~~
* To return the itemsize, meaning the size of one array element in bytes of `my_array`:
~~~
my_array.itemsize
~~~
* To return the buffer object that points to `my_array`'s place in memory:
~~~
my_array.data
~~~
* To return the number of the array dimensions of `my_array`:
~~~
my_array.ndim
~~~
* To create an ndarray with five columns and one row:
~~~
my_array = np.array([1, 2, 3, 4, 5])
~~~
* To create an ndarray with four columns and four rows:
~~~
my_array = np.array([[6, 12, 93, 2], [5, 26, 78, 90], [3, 12, 16, 22], [5, 3, 1, 16]])
~~~

#### Selecting Items
* To select the item located on the third row and fourth column:
~~~
my_array[2,3]
~~~
* Remember that numpy uses zero indexing so the third row is index 2 and the fourth column is index 3.
* To access items in an array the convention is array[row, column].

#### Empty (Uninitialized) Arrays
* To create an empty array:
~~~
np.empty(shape, dtype, order)
~~~
* For example:
~~~
np.empty((4,4))
~~~

#### Array Filled With Zeros
* To create an array where the elements are all zeros:
~~~
np.zeroes((4,4), dtype=int)
~~~

#### Array Filled with Ones
* To create an array where the elements are all ones:
~~~
np.ones((4,4), dtype=int)
~~~

#### Array with Evenly Spaced Values Within a Given Range
* To create an array with evenly spaced values within a specific range:
~~~
np.arange(start, stop, step, dtype)
~~~
* For example:
~~~
np.arange(1, 10)
~~~

#### Reshaping an Array
* To give a new shape to an array without changing its data:
~~~
my_array = np.ones((4,4))
np.reshape(my_array, (8,2))
~~~

#### Concatenating Arrays
* To join two or more arrays of the same shape along the rows:
~~~
np.concatenate((array1, array2), axis=0)
~~~
* To join two or more arrays of the same shape along the columns:
~~~
np.concatenate((array1, array2), axis=1)
~~~

#### Splitting Arrays
* To divide an array into multiple sub-arrays:
~~~
np.split(my_array, indices_or_sections, axis=0)
~~~
* For example, to split the following array into three equal parts:
~~~
concatenated_array = np.array(((1, 2),
                       (3, 4),
                       (5, 6),
                       (7, 8),
                       (9, 10),
                       (11, 12)))

split_array = np.split(concatenated_array, 3)
~~~
