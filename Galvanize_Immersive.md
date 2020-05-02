<center> <h1>Galvanize Data Science Immersive</h1> </center>
<center> <h3>Notes</h3> </center>

---

### Python Week 0
#### Day 1 Morning

**Python Environments**
* To create a different Python environment (for me python3):
```
$ conda create -n python3 python=3 anaconda
```
* To activate another environment:
```
$ source activate python3
```
* To deactivate an environment:
```
$ source deactivate
```
* Write everything in Python 3. Beginning in 2019 Python 2 will not be supported.

**Workflow:**
* Create a generic 'galvanize' folder on your desktop:
```
$ mkdir galvanize
```
* Fork the following github repo: https://github.com/gSchool/dsi-python-workshop
    1) Hit Fork on the repo.
    2) Select your own repo.
* Clone that same repo:
```
cd /Users/chrisfeller/Desktop/galvanize
git clone https://github.com/chrisfeller/dsi-python-workshop.git
```
* Move the file slacked out from Frank titled 'lecture_morning' into /Users/chrisfeller/Desktop/galvanize/dsi-python-workshop/day1
* Open the ipython notebook 'lecture_morning.ipynb' in python3
```
$ jupyter notebook
```
* *Sidenote:* To jump to desktop at any point: `F11`
* Acceptable text editors: Atom, vim, emacs, or sublime.
* *Sidenote:* To use command line commands in jupyter notebook, add an ! to the beginning:
```
!ls
```

**Terminology:**
* Python is an interpreted, dynamically-typed language with precise and efficient syntax.
    * Interpreted: not compiled before running it. It's interpreted at runtime. This is good for fast prototyping and portability. However, interpreted programs are slower than compiled programs.
    * Dynamically-typed: Variable names are assigned to objects, not to a type declared beforehand. You don't have to specify what type each variable is.
* Using the Ipython Console, Python has a rapid **REPL** (Read-Evaluate-Print-Loop) that facilitates prototyping and testing snippets of code.
* **NumPy** has powerful array and linear algebra capabilities.
* **SciPy** contains modules for optimization, linear algebra, integration, interpolation, FFT, signal and image processing, ODE solvers and other tasks common in science and engineering.
* **Matplotlib** is a powerful plotting library.
* **Pandas** provides fast and expressive data structures. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python.
* **Sci-kit** learn is a free open-source machine learning library for Python.
* **CPython**, the reference implementation of Python, is open source software and has a community-based development model.
* The python development community is very active - there is a good change that no matter what you are doing, a module or library has already been built.
* Python is a **general-purpose** programming language. It's often used as "glue" in applications written in C, C++, and Fortran.
* The **network-effect**: many data scientists are using Python. See this http://www.oreilly.com/data/free/2016-data-science-salary-survey.csp.

**Functions:**
* A function is a reusable bit of code, defined by:
    * a name,
    * inputs: arguments are passed into the function parameters.
    * an output or return value (can be multiple values in one tuple).
* Example:
```
def is_palindrome(word):
    """ Returns whether the word is a palindrome (the same forwards and backwards).

    INPUT: str
    OUTPUT: bool
    """
    # strip all spaces and put in lowercase
    word = word.replace(" ", "").lower()

    # loop on indexes between 0 and the middle of the word
    for i in range(len(word) // 2):
        # if the character at this index is different
        # from the one on the mirror opposite side
        if word[i] != word[-i - 1]:
            # then it's not a palindrome
            return False

    # if this loop is exited, it means it has never returned False
    # so it should return True
    return True
```
1) What does the example function return?
    * Answer: Boolean (True or False)
2) List the function parameters:
    * Answer: word (which is a string)
3) List the arguments of the function:
    * Answer: 'rever', 'dream', and 'et la marine va venir a Malte'
* To run code in a jupyter notebook: `Shift + Enter`
* Functions are 'first-class' objects in python:
    * They can be passed as arguments to other functions
    * They can be returned as values from other functions
    * Can be assigned to variables
    * Can be stored in other data structures
    * Their type is 'function'
* Follow the D.R.Y principle: Don't Repeat Yourself
* Example #2:
```
def fibonacci(a1, a2, n):
    """Prints the n first elements of a fibonacci suite
    given values a1 and a2 for first two ranks."""

    # eliminating easy cases first
    if (n <= 0):
        return None
    if (n = 1):
        print(a1)
    if (n = 2):
        print(a2)

    # looping on n
    anm2 = a1 # value at rank n-2
    anm1 = a2 # value at rank n-1
    for i in range(2, n):
        an = anm2 + anm1 # value at rank n
        print(an)
        anm2 = anm1      # updating rank n-2
        anm1 = an        # updating rank n-1
    return None

Examples:
print("first 5 values with 1,1")
fibonacci(1,1,5)

print("first 5 values with 1,4")
fibonacci(1,4,5)
```
1) What does the above function return?
    * Answer: None
2) List the function parameters:
    * Answer: a1, a2, and n
3) List the arguments passed to the function:
    * Answer: (1, 1, 5) and (1, 4, 5)
4) Was the fibonacci function every stored to a variable?
    * No
* Example #3:
```
def fibonacci_v2(a1, a2, n, func):
    """Prints the n first elements of a fibonacci suite
    given values a1 and a2 for first two ranks,
    and func the operation for computing rank n."""

    # eliminating easy cases first
    if (n <= 0):
        return None
    if (n = 1):
        print(a1)
    if (n = 2):
        print(a2)

    # looping on n
    anm2 = a1 # value at rank n-2
    anm1 = a2 # value at rank n-1
    for i in range(2, n):
        # NOTICE THE CHANGE HERE ???
        an = func(anm2,anm1) # value at rank n
        print(an)
        anm2 = anm1      # updating rank n-2
        anm1 = an        # updating rank n-1
    return None

some helper functions:
def fibo_operation_add(anm2, anm1):
    return(anm2 + anm1)

def fibo_operation_prod(anm2, anm1):
    return(anm2 * anm1)

See how it works:
print("first 5 values with 1,1 and addition")
fibonacci_v2(1,1,5,fibo_operation_add)

print("first 5 values with 1,4 and addition")
fibonacci_v2(1,4,5,fibo_operation_add)

print("first 5 values with 1,1 and product")
fibonacci_v2(1,1,5,fibo_operation_prod)

print("first 5 values with 1,4 and product")
fibonacci_v2(1,4,5,fibo_operation_prod)
```
1) What is the purpose of the extra parameter in the function definition?
    * Answer: To input the function on which to rank the fibonacci sequence.
2) Is something returned by the main fibonacci function? How about the helper functions?
    * No, there is nothing return by the fibonacci function. Yes, there is something returned by each helper function.
3) Was the fibonacci function ever stored to a variable? How about the helper functions?
    * No, the fibonacci function was not stored in a variable. Yes, the helper functions were each stored into local variables.
4) Why do it this way?
    * To later call them. Makes our fibonacci function more generalizable.

**Modules:**
* Modules:
    * Design and use a module to store functions that you want to reuse
    * Implement a single 'main' module that imports and runs your code
    * Run you code inside a 'main' block
* What is the difference between a library and a module?
    * Answer: A library is installable, meaning it is accompanied by other install files. A module is a simple file.
* The file `my_modle.py` contains:
```
def foo(x, y):
    return x+y

def bar(x, y, z):
    return x - y + z
```
* There are two ways to import these functions:
```
from my_module import foo, bar
print(foo(1, 2))
print(bar(3, 4, 5))
 #OR
import my_module
print(my_module.foo(1, 2))
print(my_module.bar(3, 4, 5))
```
* To check contents of workspace in jupyter notebook:
```
%who
```
* To check contents of workspace in python:
```
dir()
```

**Types:**
* Types of python objects:

| Type  | Description  | Example Values(S)  |
|---|---|---|
| `int`  | integers  | `1, 2, -3`  |
| `float`  | real numbers, float values  | `1.0, 2.5, 102342.32423`  |
| `str`  | strings  | `'abc'`  |
| `tuple`  | an immutable tuple of values, each has its own type  | `(1, 'a', 5.0)`  |
|  `list` | a list defined as an indexed sequence of elements  | `[1, 3, 5, 7]`  |
| `dict`  | a dictionary that maps keys to values  | `{'a':1, 'b':2}`  |
| `set`  | a set of distinct values  | `{1, 2, 3}`  |

* To check the type of an object:
1) `type()` returns the type itself
    ```
    type(some_object)
    ```
2) `isinstance` returns True if the object has the given type
    ```
    isinstance(some_object, some_type)
    ```
    * Generally, `isinstance()` is better to use.
* Example #1:
```
my_string = 'abc'

print('object:', my_string)
print(type(my_string))

print('Is string?', isinstance(my_string, str))
```
* Example #2:
```
my_integer = 123

print('object:', my_integer)
print(type(my_integer))

print('Is integer?', isinstance(my_integer, int))
print('Is float?', isinstance(my_integer, float))
```

**Immutable vs Mutable Types:**
* Immutable: Objects that can't be changed; can only be redefined (replaced, overwrite)
    * int 1, 2, -3
    * float 1.0, 2.5, 102342.32423
    * str 'abc'
    * tuple (1, 'a', 5.0)
* Mutable: Objects that can be changed. Usually they are structured from which elements can be modified, replaced, exchanged.
    * list [1, 3, 5, 7]
    * dict {'a':1, 'b':2}
    * set {1, 2, 3}
* Immutable types will need a new memory block for their value to change (you never modify anything inside that block.)
* Mutable types will let you modify what's inside that block.

---

#### Day 1 Afternoon
**Timing Code Snippets:**
* Which of the three code snippets is fastest:
    1) Version 1 - for loop:
    ```
    absolute_values = []
    for elem in lst:
        absolute_values.append(abs(elem))
    ```
    2) Version 2 - list comprehension:
    ```
    absolute_values = [abs(elem) for elem in lst]
    ```
    3) Version 3 - map:
    ```
    absolute_values = list(map(abs, lst))
    ```
* Use timeit to test:
```
import timeit

# first make a decently sized list of positive and negative integers
import numpy as np
# 5000 random integers from -1000 to 1000
lst = list(np.random.randint(-1000, 1001, size=5000))

# Version 1
s = """
absolute_values = []
for elem in lst:
    absolute_values.append(abs(elem))
"""
num_times = 10000
time_v1 = timeit.timeit(stmt=s, globals=globals(), number=num_times) # globals to access lst, num_times to execute
print("Version 1 takes {0:0.3f} seconds for {1} executions.".format(time_v1, num_times))

# Version 2
s = """
absolute_values = [abs(elem) for elem in lst]
"""
num_times = 10000
time_v2 = timeit.timeit(stmt=s, globals=globals(), number=num_times) # globals to access lst, num_times to execute
print("Version 2 takes {0:0.3f} seconds for {1} executions.".format(time_v2, num_times))

# Version 3
s = """
absolute_values = list(map(abs, lst))
"""
num_times = 10000
time_v3 = timeit.timeit(stmt=s, globals=globals(), number=num_times) # globals to access lst, num_times to execute
print("Version 3 takes {0:0.3f} seconds for {1} executions.".format(time_v3, num_times))
```

**Timing Functions:**
* You can also use timeit to time functions:
```
import timeit

# Version (function) 1
def min_and_max_f1(lst):
    '''
    INPUT: list
    OUTPUT: tuple of two ints/floats

    Given a list of ints and/or floats, return a 2-tuple containing the values
    of the items with the smallest and largest absolute values.

    In the case of an empty list, return None.
    '''
    if lst:
        abs_lst = [abs(elem) for elem in lst]
        i_min, i_max = abs_lst.index(min(abs_lst)), abs_lst.index(max(abs_lst))
        return (lst[i_min], lst[i_max])
    return None

num_times = 10000
time_v1 = timeit.timeit('min_and_max_f1(lst)', globals=globals(), number=num_times)
print("Function 1 takes {0:0.3f} seconds for {1} executions.".format(time_v1, num_times))

# Version (function) 2
def min_and_max_f2(lst):
    '''
    INPUT: list
    OUTPUT: tuple of two ints/floats

    Given a list of ints and/or floats, return a 2-tuple containing the values
    of the items with the smallest and largest absolute values.

    In the case of an empty list, return None.
    '''
    if lst:
        abs_lst = list(map(abs, lst))
        i_min, i_max = abs_lst.index(min(abs_lst)), abs_lst.index(max(abs_lst))
        return (lst[i_min], lst[i_max])
    return None

num_times = 10000
time_v2 = timeit.timeit('min_and_max_f2(lst)', globals=globals(), number=num_times)
print("Function 2 takes {0:0.3f} seconds for {1} executions.".format(time_v2, num_times))
```
* Consider readability (what is pythonic) in addition to which is fastest.

**String Formatting:**
* `str.format()` is the recommended way over `"Hello %s" % 'Chris'`
* Three ways to use `str.format()`:
```
print("I live in {state} near {city}".format(state='WA', city='Seatte'))
print("I live in {0} near {1}".format('WA', 'Seattle'))
print("I love in {} near {}".format('WA', 'Seattle'))
```
* The maximum characters in a line should not exceed 80.

**Float Formatting:**
* To format a float:
```
mse = 126.159320642998
print("Mean Square Error: {0:f}".format(mse))
```
* The 0 matches with the first item in `.format(item)`
    * The f means floating point and defaults to six places after the decimal.
* To format a float to a specific decimal value, in this case two places:
```
mse = 126.159320642998
print("Mean Square Error: {0:.2f}".format(mse))
```


**File I/O:**
* To look at the first few lines of a file within the terminal:
```
$ less file.txt
```
* To look at the entire file within the terminal:
```
$ cat file.txt
```
* To read an entire file within python (this is dangerous):
```
my_file = open('sample_file.txt', 'r')
my_contents = my_file.read()
```
* The preferred way to read in a file is to read it line-by-line, which is more efficient because it avoids loading the full content of the file into memory.
    * This is often done via a generator, which is an iterator that reads in values, one-at-a-time, from an iterable object.
* The correct way to do so:
```
count = 0    # counting the number of characters in the file

with open('sample_file.txt', 'r') as my_file:
    for line in my_file:
        # do something with line...
        # below we count the number of chars, neglecting beginning/ending whitespace
        count += len(line.strip())

print(count)
```

**Reading CSV Files:**
* To read in a .csv, don't split on the comma, instead:
```
import csv

with open('sample_csv_easy.csv') as my_file:
    reader = csv.reader(my_file)
    for line in reader:
        print(line)
```

**Writing Files:**
* To write to the end of a file:
```
lines_to_write = ['abcd', 'blabla', '1234']

with open('file_to_write.txt', 'w') as my_file:
    for line in lines_to_write:
        my_file.write(line)
        my_file.write('\n')
```
* To append to the end of a file:
```
lines_to_append = ['1357', '2468']

with open('file_to_write.txt', 'a') as my_file:
    for line in lines_to_append:
        my_file.write(line)
        my_file.write('\n')
```

---

### Python Week 0
#### Day 2 Morning

**Code Style:**
* To run your python file through Pep8 standards, download pep8 library:
```
conda install pep8
```
* https://pypi.python.org/pypi/pep8

**Converting Python2 to Python3 Code:**
* A magic function that shows you changes needed to change code from Python2 to Python3:
    * `2to3`
    * https://docs.python.org/2/library/2to3.html

**Magic Functions in Jupyter Notebook:**
* To use magic functions, such as timeit in Jupyter Notebook, use double percent-signs:
```
%%timeit
```
* To use magic functions, such as timeit in ipython, use single percent-sing:
```
%timeit
```

**Readability:**
* Avoid using acronyms and abbreviations when naming objects.
* Instead name them the entire word and rely on tab completion to avoid lots of typing.
* This practice acts as a good form of code documentation without commenting to explain what objects are.

**Dictionaries:**
* Dictionaries are key-value pairs.
* They are useful throughout python as a simple table or data structure.
* Three ways to initialize a dictionary:
1)
```
prices = {}
prices['banana'] = 1
prices['steak'] = 10
prices['ice cream'] = 5
```
2)
```
prices_v2 = {'steak': 10, 'banana': 1, 'ice cream': 5}
```
3)
```
prices_v3 = dict([('steak', 10), ('banana', 1), ('ice cream', 5)])
```
* Dictionaries are unordered! The key, value pairs will not come back to you in the same order you put them in. They could even change order each time you print your dictionary.
* Dictionary keys can be of any immutable type.
* Dictionary keys and values are not type checked.

**Dictionary Methods:**
* Most common dictionary methods:
```
d.keys()
d.values()
d.items()
```
* Use `.get()` instead of `d['key']` to avoid KeyErrors:
```
d.get('key')
```
* To return a custom message when a key is missing, while using `.get()`:
```
d.get('key', "Sorry, no key in the dictionary")
```
* To add items to a dictionary use `.update()`:
```
d.update({'key': value})
```
* Exercise: Create a dictionary of favorite movies, including everyone sitting at your table:
```
fav_movies = {'Chris': 'The Goonies', 'Howard': 'Batman', 'Amber': 'Happy Gilmore', 'Dave A': 'Rockstar', 'Dave K': 'Princess  Mono', 'Mickey': 'O Brother Where Art Though', 'Sarah': 'Billy Madison', 'Randy':'Knights Tale'}
```
* List the keys from the `fav_movies` dictionary:
```
list(fav_movies.keys())
```
* List the two ways to display Chris' favorite movie:
1) Worse Version:
    ```
    fav_movies['Chris']
    ```
2) Best Version:
    ```
    fav_movies.get('Chris')
    ```

**Dictionary Comprehension:**
* Example:
    ```
    {x:x**2 for x in (2, 4, 6)}
    ```

**Sets:**
* Think of sets as value-less dictionaries.
    * All items within a set are unique. No duplicates!
* Three ways to initialize a set:
1)
```
groceries = set(['carrots', 'figs', 'popcorn'])
```
2)
```
groceries = {'carrots', 'figs', 'popcorn'}
```
3)
```
groceries = set()
groceries.add('carrots')
groceries.add('figs')
groceries.add('popcorn')
```

**Set Methods:**
* Most common set methods:
```
s.add()
s.union()
s.intersection()
s.difference()
s.update()
s.issubset()
s.issuperset()
s.copy()
```
* Examples:
```
whole_foods = {'kale', 'squash' ,'kombucha', 'granola'}
safeway = {'lettuce', 'carrot', 'seltzer', 'granola'}
bodega = {'lettuce', 'seltzer'}

whole_foods.union(safeway)

whole_foods.intersection(safeway)

whole_foods.isdisjoint(safeway)

whole_foods.remove('granola')

whole_foods.add('hippy granola')

safeway.issuperset(bodega)

bodega.issubset(safeway)

safeway.copy()
```

**Comprehensions:**
* Table of Comprehensions:

| Type  | Constructor  | Output  | Comprehension  |
|---|---|---|---|
| `str`  | `str(1.0)`  | `'1.0'`  |   |
| `tuple`  | `tuple('abc')`  | `('a', 'b', 'c')`  | `(x for x in 'abc')`  |
| `list`  |  `list('abc')` |  `['a', 'b', 'c']` | `[x for x in 'abc']`  |
| `set`  | `set('abc')`  | `{'a', 'b', 'c'}`  | `{x for x in 'abc'}`  |
| `dict`  | `dict([('a', 1), ('b', 2)])`  | `{'a':1, 'b':2}`  | `{x:i for i, x in enumerate('ab')}`  |

* Tuple Comprehension:
```
abc_tuple = (x for x in 'abc')
```
* List Comprehension:
```
abc_list = [x for x in 'abc']
```
* Set Comprehension:
```
abc_set = {x for x in 'abc'}
```

* Using functions within comprehensions:
```
def square(x):
    return x*x

[square(val) for val in [1 ,2, 3, 4, 5]]
```

**defaultdict and Counter:**
* Two common additional functions in dictionary creation are:
    1) defaultdict
    2) Counter
* Both come from the collections library:
```
from collections import defaultdict, Counter
```
* Standard Dictionary Creation:
```
def count_with_if_block(lst):
    '''
    INPUT: list
    OUTPUT: dict

    Return a dictionary whose keys are the items in the list and
    value is the count of the number of times that item occurred in the list.
    '''
    d = {}
    for i, item in enumerate(lst):
        if item in d:
            d[item] += 1
        else:
            d[item] = 1
    return d
```
* defaultdict Dictionary Creation:
```
from collections import defaultdict
def count_with_defualt_dict(lst):
    d = defaultdict(int)
    for i, item in enumerate(lst):
        d[item] += 1
    return d

count_with_defualt_dict(alphabet_soup)
```
* Counter Dictionary Creation:
```
from collections import Counter
Counter(alphabet_soup)
```
* To create a defaultdict with a specific default values:
```
ice_cream = defaultdict(lambda: 'vanilla')
ice_cream['Elliot']
ice_cream.get('Elliot') # Returns vanilla
```

---

#### Day 2 Afternoon
**Object Oriented Programming (OOP):**
* The opposite of OOP is procedure-oriented programming (functions).
* OOP refers to a type of computer programming (software design) in which programmers define not only the date type of a date structure, but also the types of operations (functions) that can be applied to the date structure.

**Classes and Objects:**
* A class creates a new type where objects are instances of the class.
* Objects can store data using variables that belong to a class.
* Variables that belong to an object or class are referred to as fields.
* Objects can also have functionality by using functions that belong to a class. Such functions are called methods of a class.
* This terminology is important because it helps us to differentiate between functions and variables which are independent and those which belong to a class or object.
* Collectively, the fields and methods of a class can be referred to as attributes.

**Example: Dog Object:**
* Dogs can be described in terms of their characteristics:
    * size
    * breed
    * name
    * favorite toy
* Dogs have behaviors and abilities:
    * run
    * bark
    * eat
    * sleep
* *Attributes* are the characteristics of the dog.
* *Methods* are the behaviors of the dog.

**Class vs. Instance:**
* object is used broadly to refer to classes or instances of classes.
* class refers to the set of attributes, methods, etc. that define a group of objects.
* instance refers to a specific example of a class.

**Defining a Class:**
* Example:
```
class Dog(object):
    """Common household pet"""
    def __init__(self, name, breed, favorite_toy):
        """
        Args:
            name (str): the dog's name, ex. "Snoopy"
            breed (str): the breed of the dog, ex. "Beagle"
            favorite_toy (str): something the dog likes to play with most

        """
        self.name = name
        self.breed = breed
        self.favorite_toy = favorite_toy
```
* The `__init__()` method is what creates an instance.
    * It is known as a 'dunder' method because it has duble underscores on each side.

**Instantiating an Object:**
* Example:
```
my_dog = Dog('Snoopy', 'Beagle', 'Frisbee')
```
* Display the attributes of the instance my_dog:
```
my_dog.name
my_dog.breed
my_dog.favorite_toy
```

**Encapsulation:**
* Encapsulation is an object-oriented programming concept that binds together the data and functions that manipulate the data, and that keeps both safe from outside interference and misuse.
* Example:
```
class Car:

    def __init__(self):
        self.__updateSoftware()

    def drive(self):
        print ('driving')

    def __updateSoftware(self):
        print ('updating software')

redcar = Car()
redcar.drive()
```
* In the above example, you are unable to update redcars software because it is a private method (hence the underscore)
    * One underscore means stay away.
    * Two underscore means definitely stay away.

**Polymorphism:**
* Polymorphism is the ability to create multiple instances of the same Class.
```
my_dog = Dog('Snoopy', 'Beagle', 'Frisbee')
your_dog = Dog('Spot', 'Poodle', 'Tennis Ball')
```

**Inheritance:**
* A child class can take in the same methods and attributes as its parent class.
* Example:
```
class SchoolMember:
    '''Represents any school member.'''
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print('(Initialized SchoolMember: {})'.format(self.name))

    def tell(self):
        '''Tell my details.'''
        print('Name:"{}" Age:"{}"'.format(self.name, self.age), end=" ")


class Teacher(SchoolMember):
    '''Represents a teacher.'''
    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)
        self.salary = salary
        print('(Initialized Teacher: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Salary: "{:d}"'.format(self.salary))


class Student(SchoolMember):
    '''Represents a student.'''
    def __init__(self, name, age, marks):
        SchoolMember.__init__(self, name, age)
        self.marks = marks
        print('(Initialized Student: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Marks: "{:d}"'.format(self.marks))
```
* To check if a class is a subclass (child) of another class:
```
issubclass(Teacher, SchoolMember)
```

**Magic Methods:**
* Operator overloading over an existing function within Python.
* Example:
```
# Magic methods: __len__ and __str__
class Dog(object):
    """Common household pet"""

    def __init__(self, name, breed, favorite_toy, has_toy=False, plays_fetch=True):
        """
        Args:
            name (str): the dog's name, ex. "Snoopy"
            breed (str): the breed of the dog, ex. "Beagle"
            favorite_toy (str): something the dog likes to play with most
            has_toy (bool): whether the dog has its favorite toy
            plays_fetch (bool): whether the dog knows how to play fetch
        """
        self.name = name
        self.breed = breed
        self.favorite_toy = favorite_toy
        self.has_toy = has_toy
        self.plays_fetch = plays_fetch

    def __len__(self):
        """The length of a dog is the length of its name"""
        return len(self.name)

    def __str__(self):
        """Name, Breed, Favorite Toy"""
        return '{name}, {breed}, {favorite_toy}'.format(name=self.name,
                                                        breed=self.breed,
                                                        favorite_toy=self.favorite_toy)

dog = Dog('Snoopy', 'Beagle', 'Frisbee')
print (len(dog))
print (str(dog))
```
* Example #2:
```
# Magic method: __add__
class FruitBasket(object):
    """A collection of apples and pears"""

    def __init__(self, num_apples, num_pears):
        """
        Args:
            num_apples (int): number of apples in the basket
            num_pears (int): number of pears in the basket
        """
        self.num_apples = num_apples
        self.num_pears = num_pears

    def __add__(self, other):
        """Combines two baskets into one"""
        num_apples = self.num_apples + other.num_apples
        num_pears = self.num_pears + other.num_pears
        new_basket = FruitBasket(num_apples, num_pears)
        return new_basket

basket1 = FruitBasket(10, 20)
basket2 = FruitBasket(30, 40)
basket3 = basket1 + basket2
print (basket3.num_apples, basket3.num_pears)
```
* Under the hood `__add__` is how addition works with numeric types too:
```
a, b = 1, 2
print(a + b)
print(a.__add__(b))
```

---

### Python Week 0
#### Day 3 Morning

**Differences between Python 2 and Python 3:**
1. print statements use parentheses
2. `xrange` is now `range`
3. `iterkeys` is now `keys`
4. `iteritems`is now `items`
5. classes automatically inherit `object`
6. Unicode is handled differently
7. `raw_input` is now an `input`

**Accessing Documentation:**
* To access documentation via console:
```
$ pydoc itertools.permutations
```
* To access documentation via ipython:
```
? itertools.permutations
```

**Probability Introduction (PDF Slides):**
* Statistics is the inverse of probability.
* Probability refers to the study of pattern.
    * When we solve problems in probability we assume that all basic features of the random process are known, and our goal is to discover other, deeper features.
    * Example: If I have a coin which is known to land head exactly half of the time, it is a problem in probability to determine how often the coin will never land on heads over ten consecutive flips.
* Statistics refers to the study of random process where some basic features of the random process are unknown, and our goal is to infer from observations basic, hidden features of the random process.
    * Example: It is a problem in statistics to determine, when presented with a coin which has landed tails ten consecutive times, whether one should continue to believe it fair.

**Counting:**
* The basic problem solving skill you need to solve problems in probability is counting.
* Basic Counting Principle: If a task can be accomplished as a series of steps, then the number of outcomes of the task is the **product** of the number of outcomes of each individual step.
    * Example: How many ways are there to arrange four letters of the alphabet?
        * Think: How can we accomplish this task as a step-by-step process.
            1) Pick the first letter, write it down.
            2) Pick the second letter, write it down.
            3) Pick the third letter, write it down.
            4) Pick the fourth letter, write it down.
        * Answer: `26 * 26 * 26 * 26 = 456976`
        * Practice Code:
            ```
            import itertools
            short_alpha = 'abc'  # test it small scale first to see if we get what we should!
            num_letters = 2
            arrange_short_alpha = list(itertools.product(short_alpha, repeat=num_letters))
            num_arrange_short = len(arrange_short_alpha)
            print("There are {0} arrangements.".format(num_arrange_short))
            print("Here they are:")
            print(arrange_short_alpha)
            ```
        * Example Code:
            ```
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            num_letters = 4
            arrange_alpha = list(itertools.product(alphabet, repeat=num_letters))
            num_arrange = len(arrange_alpha)
            print("There are {0} arrangements.".format(num_arrange))
            print("Here are the first 20:")
            print(arrange_alpha[:20])
            ```
* However, when you cannot re-use an object from the objet pool we use permutations.
    * This is called selection without replacement.
    * The number of ordered selections of k objects without replacement from a population for k objects is called the number of permutations of k objects taken from n.
    * Practice Code: Same problem but once a letter is picked it's used up:
        ```
        short_alpha = 'abc'  # test it small scale first
        num_letters = 2
        perm_short_alpha = list(itertools.permutations(short_alpha, r=num_letters))
        num_perm_short = len(perm_short_alpha)
        print("There are {0} permutations.".format(num_perm_short))
        print("Here they are:")
        print(perm_short_alpha)
        ```
    * Example Code:
        ```
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        num_letters = 4
        perm_alpha = list(itertools.permutations(alphabet, r=num_letters))
        num_perm = len(perm_alpha)
        print("There are {0} permutations.".format(num_perm))
        print("Here are the first 20:")
        print(perm_alpha[:20])
        ```
    * This is calculating `26 * 25 * 24 * 23 = 358800`
    * However, this code is inefficient because it is creating the actual permutations and then we are counting them. If instead we want to just count instead of create:
        ```
        def num_perms(n, k):
        """# of permutations given n options, k objects"""
        if n =k:
            np = 1
            for _ in range(k):
                np *= n
                n = n - 1
            return np
        raise ValueError('Error: n must be = k')
        return none
        ```
    * Example: You have 25 math and stats books on a bookshelf. How many ways are there to arrange 5 of those books in any order?
        ```
        print(num_perms(25,5))
        ```
* Lastly, if you have a situation where you are selecting without replacement and the order of the selections does not matter you use **combinations**:
    * The number of unordered selections of k objects without replacement from a population for k objects is called the number of combinations of k objects taken from n.
    * Example: How many 5 card hands are possible when drawing from a standard 52 card deck?
        * Notice that the order in which we draw cards is not important.
        * The inefficient way:
        ```
        suits = 'CHSD'
        vals = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        deck = [val + suit for suit in suits for val in vals]

        num_cards = 5 # careful!
        hand_combinations = list(itertools.combinations(deck, r=num_cards))
        num_comb = len(hand_combinations)
        print("There are {0} combinations.".format(num_comb))
        print("Here are the first 13 hands:")
        for hand in hand_combinations[:13]:
            print(hand)
        ```
        * More efficient way:
        ```
        def num_coms(n, k, f):
            return int(f(n, k) / f(k, k))
        print(num_coms(52, 5, num_perms))
        ```
    * SciPy has built-ins that can be useful instead of creating ones own function:
        ```
        from scipy.special import comb
        n = 52
        k = 5
        comb(n, k, exact=False)
        ```

**Cards Example:**
* How many hands of five cards are full houses?
```
# make a simpler deck, don't care about suits
deck = 'A23456789TJQK' * 4
print(deck)
print(len(deck))

num_cards = 5 # careful!
hand_combinations = list(itertools.combinations(deck, r=num_cards))
num_comb = len(hand_combinations)
print("There are {0} combinations.".format(num_comb))
print("Here are the first 13 hands:")
for hand in hand_combinations[:13]:
    print(hand)

def classify_hands(hands):
    """Classifies hand as:
        1 pair (1P)
        2 pair (2P)
        3 of-a-kind (3K)
        Full House (FH)
        4 of-a-kind (4K) or
        other (OT)
        straights are lumped with other

    Return dictionary with keys "1P, 2P, 3K, FH, 4K, or OT"
    and the values as the list of hands with that classification.
    """
    hand_dict = defaultdict(list)
    for hand in hands:
        hand_counter = Counter(hand)
        num_unique = len(hand_counter)
        num_most_common = hand_counter.most_common()[0][1]
        if num_unique == 5:
            hand_dict['OT'].append(hand)
        elif num_unique == 4:
            hand_dict['1P'].append(hand)
        elif num_unique == 3:
            if num_most_common == 3:
                hand_dict['3K'].append(hand)
            else:
                hand_dict['2P'].append(hand)
        else:
            if num_most_common == 3:
                hand_dict['FH'].append(hand)
            else:
                hand_dict['4K'].append(hand)
    return hand_dict

hand_dict = classify_hands(hand_combinations)

print("There were {0} hand combinations.".format(num_comb))
tot_hands = sum([len(v) for v in hand_dict.values()])
print("{0} hands were classified.".format(tot_hands))
for k, v in hand_dict.items():
    print("Hand: {0}, number: {1:7d}".format(k, len(v)))
```

**Probabilities:**
* An outcome is a single thing that can happen.
    * Example: When thinking about poker hands, an outcome is a single hand (any unordered collection of five cards).
* An event is a collection of things that can happen, usually given by a short description.
    * The collection of all full-houses, three of a kinds, etc. are events.
* The probability of an event it:
    ```
    # of ways event can happen / total # of things that can happen
    ```
    * Example - The probability of a full-house:
        ```
        # of full houses / total # of hands
        ```
* Coding Probabilities:
```
# note that P no longer stands for permutation, now probability
p_FH = len(hand_dict['FH'])/tot_hands
print("The probability of a full house is {0:0.4f}.".format(p_FH))
p_FH2 = 13 * num_coms(4,2,num_perms) * 12 * num_coms(4,3,num_perms)/tot_hands # now I believe
print("The probability of a full house is {0:0.4f}.".format(p_FH2))
```
* Example: What is the probability of drawing a hand containing a three of a kind?
```
num_hands = num_coms(52, 5, num_perms)
num_3K = 13 * num_coms(4, 3, num_perms) * num_coms(49, 2, num_perms)
P_3K = num_3K/num_hands
print("The probability of drawing 3K is {0:0.4f}.".format(P_3K))
```
* Example: What is the probability of drawing a hand containing a three of a kind that is not a full house?
```
num_FH = 13 * num_coms(4, 3, num_perms) * 12 * num_coms(4, 2, num_perms)
num_3K_notFH = (num_3K - num_FH)/num_hands
print("The probability of drawing 3K that is not FH is {0:0.4f}.".format(num_3K_notFH))
```
* Example: What is the probability of drawing a hand containing a pair, that does not contain a three of a kind or four of a kind?
```
num_1P = 13 * num_coms(4, 2, num_perms) * num_coms(50, 3, num_perms)
num_4K = 13 * num_coms(4, 4, num_perms) * num_coms(48, 1, num_perms)
p_1P_no3K_no4K = (num_1P - num_3K - num_4K)/num_hands
print("Probability of drawing 1P that is not 3K or 4K is {0:0.4f}.".format(p_1P_no3K_no4K))
```

**Conditional Probability:**
* Suppose we know that on event C has already happened or will happen (the condition), and we want to know the probability of different event B. Then the conditional probability of A given B is defined by:
```
# of ways A and B both happen / # of ways B can happen
```
* Coding conditional probability:
```
num_3ofakind = 13 * num_coms(4, 3, num_perms) * num_coms(49, 2, num_perms)
print("Number hands containing 3 of a kind: {0}".format(int(num_3ofakind)))
num_1pair = 13 * num_coms(4, 2, num_perms) * num_coms(50, 3, num_perms)
print("Number hands containing 1 pair: {0}".format(int(num_1pair)))
p_3ok_1pair = num_3ofakind/num_1pair
print("Given that the hand already contains a pair, the probability")
print("of a 3 of a kind is {0:0.3f}.".format(p_3ok_1pair))
```
* Example: What is the conditional probability that you draw a four of a kind, given that you know you already have a pair?
```
num_4K = 13 * num_coms(4, 4, num_perms) * num_coms(48, 1, num_perms)
num_1P = 13 * num_coms(4, 2, num_perms) * num_coms(50, 3, num_perms)
p_4K_given_1P = (num_4K/num_1P)
print("Cond probability of drawing a 4K given 1P is {0:0.4f}.".format(p_4K_given_1P))
```

**PostgreSQL Cheatsheet:**
* Magic words:
```
psql -U posgres
```
    * If run with the `-E` flag, it will describe the underlying queries of `\` commands (great for learning).
* Most `\d` commands support additional param of `__schema__.name__` and accept wildcards like `*.*`

* `\q`: Quit/Exit
* `\c __database__`: Connect to a database
* `\d __table__`: Show table definition including triggers
* `\dt *.*`: List tables from all schemas (if `*.*` is omitted will only show SEARCH_PATH ones)
* `\l`: List databases
* `\dn`: List schemas
* `\df`: List functions
* `\dv`: List views
* `\df+ __function__`: Show function SQL code
* `\x`: Pretty-format query results instead of the not-so-useful ASCII table.

---

### Python Week 0
#### Day 4 Morning

**Helpful Atom Packages:**
Go to Packages - Settings View - Install Packages/Themes to find these.

* linter-pyflakes: This package is amazing. It highlights syntax errors and other small mistakes in your code before you even run it!
* local-history: This package keeps a history of all changes to your file so you can revert to whatever version you need. Prioritize using Github for version control, but when you inevitably forget to make a commit or two, this will definitely come in handy.
* click-link: So you can actually click on links in md files. Not essential but if you get angry having to copy & paste links, you may appreciate this.
* python-autopep8: Also not essential but it's a good habit to learn how to format your code according to best practices.

**General Advice From Data Science Resident (DSR) -  Kristie:**
* Understanding how to pair-program well is important.
* We love your feedback! You can submit any problems with assignments on the #dsi_errata slack channel.
* This class is designed to be a bit too fast and a bit too difficult. Take advantage of all of the great opportunities to learn, but also find some time for mental health breaks. If you're finishing half of the assignments every day, reviewing the solutions to learn what you missed, and getting at least 50% on the assessments, you're doing great!
* If you're stuck on one small part of an assignment for more than 30-40 minutes, please ask for help. You'll learn so much more if you can find a way to move past the little things and get an understanding of the big picture lessons for the day.
* There may be days when you feel overwhelmed but if you made it this far, I guarantee you have what it takes. And chances are, if you're feeling overwhelmed, you're not the only one!

**Paired Programming Advice:**
* Every 20-30 minutes, the driver and navigator roles will switch. Use the
`countdown` utility in the Terminal to keep time. It will display a Mac OS X
notification when time is up.
* Example:
```
$ countdown 25
Countdown started: 25 minutes
[1] 37336
```

**Checking Memory Usage:**
* Use htop:
```
$htop
```
* To quit: `F10`

**Linear Algebra:**
* Learning Objectives:
    1) Become familiar with linear algebra's basic data structures:
        * **scalar**
        * **vector**
        * **matrix**
        * **tensor**
    2) Create, manipulate, and generally begin to get comfortable with NumPy arrays.
* Reasons why a solid understanding of linear algebra is crutial for a practicing data scientist:
    1) Linear models can concisely be written in vector notation.
    2) Regularization often makes use of matrix norms.
        * Regularization: practice of avoiding overfitting models.
    3) Matrix decompositions are commonlhy used in recommender systems.

**Scalars, vectors, matrices and tensors:**
* **Scaler**:
    * Contents of vectors and matrices.
    * Excel analogy: value in a cell
* **Vector**:
    * Excel analogy: single row or column of a spreadsheet
* **Matrix**:
    * Collection of columns and rows.
    * Excel analogy: a spreadsheet
    * A matrix with *m* rows and *n* columns is a (*m* x *n*) matrix and we refer to *m* and *n* as dimensions
    * Matrices are also tensors.
* **Tensor**:
    * Two-dimensional representation of data.
    * Excel analogy: multiple taps each containing a separate spreadsheet.
    * Matrices are also tensors.

| Machine Learning  | Notation  | Description  |
|---|---|---|
| Scaler  | *x*  | a single real number (ints, floats, etc.)  |
| Vector  | **x** (column vector) or **x^t** (row vector) | a 1D array of numbers (real, binary, integer, etc.)  |
| Matrix   | **X**  | a 2D array of numbers  |
| Tensor  | *f*  | an array generalized to n dimensions  |

**Rank and Dimensions:**
* If we were working with a 4x4 matrix it can be described as a tensor of rank 2 because it has two axes.
* Rank = Axes
* Tensors thus generally have rank  2 because they have more than two axes.
* To find the rank of an object in numpy:
```
a.ndim
```
* Question - What are the dimensions of the following matrix:
```
np.array([[0,0,1,0], [1, 2, 0, 1], [1, 0,0,1]])
```
    * Answer: The matrix dimensions are 3x4
* Question - Given a spreadsheet that has 3 tabs and each tab has 10 rows with 5 columns how might we represent that data with tensor?
    * Answer: The tensor would be of rank 3 and have dimensions 10x5x3

**Notation:**
* Scalars have the standard math notation: `x=1`
* Vectors are denoted by lower case bold letters such as **x**, and all vectors are assumed to be column vectors:
    * A superscript *T* denotes the transpose of a matrix or vector and thus a row vector.
* Matrices are denoted by upper-case bold letters: '**X**'

**Introduction to Numpy:**
* To import the numpy library:
```
import numpy as np
```
* The main object in numpy is the homogeneous, multidimensional array.
* An array is our programmatic way to represent vectors and matrices.
* To create an array:
```
X = np.array([[1,2,3],[4,5,6],[7,8,9]])
```

**Array Attributes:**
* Since arrays are objects, they contain methods and attributes:
    * The methods are functions that act on our matrix.
    * The attributes are data that are related to our matrix/array.
* To get the rank of an array:
```
X.ndim
```
* To get dimensions of an array:
```
X.shape
```
* To return the total number of elements, which is the product of the dimensions:
```
X.size
```

**Array Methods:**
* To get column statistics of an array:
```
X.sum(axis=0)
X.mean(axis=0)
```
* To get row statistics of an array:
```
X.sum(axis=1)
X.mean(axis=1)
```
* To create an array of the sequence 0-9:
    * `arange()` starts counting at zero instead of one.
```
np.arange(10)
```
* To create an array of the sequence 1-10:
```
np.arange(1, 11)
```
* To create an array of the sequence 2-10 by 2's:
```
np.arange(2, 11, 2)
```
* To create a matrix by reshaping an array:
```
np.arange(1,10).reshape(3,3)
```
* To create an array with evenly spaced variables for a specific interval:
```
np.linspace(0, 5, 5)
```
* To create an array of all zeros:
```
np.zeros([3, 4])
```
* To create an array of all ones:
```
np.ones([3, 4])
```

Exercises:
1) Create a 10x10 matrix with values 1-100:
```
1_to_100 = np.arange(1, 101).reshape(10, 10)
```
2) Use the array object from above
```
1_to_100.ndim
1_to_100.size
1_to_100.shape
```
3) Get the mean of the rows and columns:
```
1_to_100.mean(axis=1) #rows
1_to_100.mean(axis=0) #columns
```
4) How do you create a vector that has exactly 50 points and spans the range 11 to 23?
```
11_to_23 = np.linspace(11, 23, 50)
```

**Array Data Types:**
* Arrays may be made of different types of data, but they can only be one data type at a given time.
* To check the data type of an array:
```
x = np.array([1, 2, 3])
x.dtype
```
* To specify the data type when creating an array:
```
x = np.array([1, 2, 3], dtype='float64')
```

**Numpy Quick Reference:**
* Important numpy commands:

| NumPy command              | Note                                                        |
|----------------------------|-------------------------------------------------------------|
| `a.ndim`                     | returns the num. of dimensions or the rank                  |
| `a.shape`                    | returns the num. of rows and colums                         |
| `a.size`                     | returns the num. of rows and colums                         |
| `np.arange(start,stop,step)`    | returns a sequence vector                                   |
| `np.linspace(start,stop,steps)` | returns a evenly spaced sequence in the specificed interval |

**Matrix Operations:**
* Dimensional requirements for matrix multiplication **important**: In order for the matrix product (A x B) to exist, the number of columns in A must equal the number of rows in B.
    * (R1, C1) x (R2, C2) <- C1 and R2 must be equal
        * The output matrix will have dimensions (R1 X C2)
* To transpose an array:
```
np.array([[3, 4, 5, 6]]).T
```
* To horizontally stack columns in a matrix:
```
feature1 = np.array([[99,45,31,14]]).T
feature2 = np.array([[0,1,1,0]]).T
feature3 = np.array([[5,3,9,24]]).T
np.hstack([feature1, feature2, feature3])
```
* To vertically stack rows in a matrix:
```
row1 = np.array([[99,45,31,14]])
row2 = np.array([[0,1,1,0]])
row3 = np.array([[5,3,9,24]])
np.hstack([row1, row2, row3])
```
* To access the first element of the second row:
```
X[0, 2]
```
* To access the first column of a vector:
```
X[:,0]
```
* To access the second through third items of a vector:
```
a = np.arange(10)
a[2:4]
```
* To access every other item in a vector:
```
a[::2]
```
* To reverse the vector:
```
a[::-1]
```

**Dot Product:**
* In order to multiply two matrices, they must be conformable such that the number of columns of the first matrix must be the same as the number of rows in the second matrix.
* Arithmetic operations in numpy work elementwise:
```
a = np.array([3, 4, 5])
b = np.ones(3)
a - b
```
* The * operator does not carry out a matrix product. This is done with the dot function:
```
a = np.array([[1,2],[3,4]])
b = np.array([[1,2],[3,4]])
np.dot(a, b)
```

**Special Addition and Multiplication Operators:**
* Instead of looping through each element in a vector you can apply scalers to an entire vector at once:
```
a = np.zeros((2, 2), dtype='float')
a += 5
a *= 5
a + a
```

**Sorting Arrays:**
* To sort an array:
```
x = np.random.randint(0, 10, 5)
x.sort()
```
* To reshuffle an array:
```
np.random.shuffle(x)
```

**Common Math Functions:**
```
x = np.arange(1, 5)
np.sqrt(x) * np.pi

np.power(2, 4)

np.log(np.e)

x.max() - x.min()
```

**Other Important Numpy Commands:**
* Where:
    * To get all of the items in an array that are greater than 1:
    ```
    a[a1]
    ```
    * This could instead be written as :
    ```
    np.where(a1)
    ```
* Copy:
    * To create a copy of an array:
    ```
    b = a.copy()
    ```
* Missing Data:
```
a = np.array([[1,2,3],[4,5,np.nan],[7,8,9]])
```
* Generating Random Numbers:
```
np.random.randint(0,10,5)      # random integers from a closed interval
# Returns: array([2, 8, 3, 7, 8])

np.random.normal(0,1,5)        # random numbers from a Gaussian
# Returns array([ 1.44660159, -0.35625249, -2.09994545,  0.7626487 ,  0.36353648])

np.random.uniform(0,2,5)       # random numbers from a uniform distribution
# Returns array([ 0.07477679,  0.36409135,  1.42847035,  1.61242304,  0.54228665])
```
* Convenience Functions:
```
np.ones((3,2))
np.zeros((3,2))
np.eye(3)
np.diag([1,2,3])
np.fromfunction(lambda i, j: (i-2)**2+(j-2)**2, (5,5))
```

Exercises:
1) Create a single array for the data (4x4):
```
import numpy as np
row_names = np.array(["A2M", "FOS", "BRCA2","CPOX"])
column_names = np.array(["4h","12h","24h","48h"])
values0  = np.array([[0.12,0.08,0.06,0.02]])
values1  = np.array([[0.01,0.07,0.11,0.09]])
values2  = np.array([[0.03,0.04,0.04,0.02]])
values3  = np.array([[0.05,0.09,0.11,0.14]])

print("\nQuestion 1")
X = np.vstack([values0,values1,values2,values3])
print(X)
print(X.shape)
```
2) Find the mean expression value per gene (per row):
```
print("\nQuestion 3")
time_means = X.mean(axis=0)
print("Time means check: ", X[:,0].mean()==time_means[0])
print(["%s = %s"%(tp,time_means[t]) for t, tp in enumerate(column_names)])
```
3) Find the mean expression value per time point (per column):
```
print("\nQuestion 3")
time_means = X.mean(axis=0)
print("Time means check: ", X[:,0].mean()==time_means[0])
print(["%s = %s"%(tp,time_means[t]) for t, tp in enumerate(column_names)])
```
4) Which gene has the maximum mean expression value?
```
print("\nQuestion 4")
gene_means = X.mean(axis=1)
gene_mean_ind = np.argmax(gene_means)
print("gene with max mean expression value: %s (%s)"%(row_names[gene_mean_ind],gene_means[gene_mean_ind]))
```
5) Sort the gene names by the max expression value:
```
gene_names = row_names
print("sorted gene names: %s"%(gene_names[np.argsort(np.max(X,axis=1))]))
```

---
### Python Week 1
#### Day 1 Morning

#### Introduction to Git and GitHub
**Git vs. GitHub**
* Git:
    * Git is open-source distributed version control software that lets you keep track of file changes in a repository or folder.
    * It runs locally on your computer.
    * Checkpoints (commits) keep track of what was changed, and by whom and when.
    * You can load an earlier checkpoint to reverse changes.
    * The distributed aspect of Git allows teams to collaborate on projects.
* GitHub:
    * GitHub is a web hosting service for Git repositories (repos).
    * Github's mascot is the Octocat
    * It's in the cloud (meaning it's a bunch of server farms connected via the internet).
    **Git exists independently of GitHub, while the converse is not true.**
    * Github repos serve as remotes.
        * A remote is a shared Git repository that allows multiple collaborators to work on the same Git project from different locations.
    * Public repos are free, but private repos you have to pay for.
    * Any private repo that you can fork remains a private repo.

**Key Concepts:**
* Repository (a folder managed by git)
* Workspace (current state)
* Index (staged for commit)
* Commit (take a snapshot)
* Branch (a series of commits)
* Remote (a remote repository that you can push to or pull from)

**Key Commands:**
* `git status`: see the status of the workspace, index, and what branch you're on
* `git add`: add files to the index (commit staging area)
* `git commit`: take a snapshot of the project, committing the files in the index
* `git checkout`: switch to a different branch (use the -b option to switch to a new branch)
* `git branch`: list the branches
* `git reset`: rollback to a previous commit
* `git push`: push up the changes in a local repository to a remote repository
* `git pull`: pull down the changes from a remote repository to the local repository
* `git clone`: copy a remote repository to the local machine

**Typical DSI Workflor:**
Fork - clone - modify some files - add - commit - push

**Important:**
* Fork: used when you want to create a copy of a repo that you can then edit/add to. Once you fork it, you will no longer see updates from the original repo. You will do this as the FIRST STEP 99% of the time in this class.
* Clone: used to create a local copy of a repo on Github.com to your computer. Syncs with changes to the original repo on Github.com. You also do this step 99% of the time after forking the repo first.

*Do not commit large files to Github (anything larger than ~20mb).*

#### Tools and Workflow
**Installing Packages:**
* To install a python package not included in the anaconda distribution:
```
$ conda install package_name
```
* To install mac package utilities:
```
$ brew install package_name
```

**System Keyboard Shortcuts:**
* To open an application, use Spotlight: `CMD + SPACE`
* To switch between applications: `CMD + TAB`
* To quit an application: `CMD + q`
* To close a window of an application: `CMD + w`

**Atom Shortcuts:**
* To open Atom from the command line:
```
$ atom file_or_directory
```
* To open a new file: `CMD + n`
* To close a tab: `CMD + w`
* To save a file: `CMD + s`

**Terminal/iTerm2 Shortcuts:**
* To open a new window: `CMD + n`
* To open a new tab: `CMD + t`
* To move left and right between tabs: `CMD + LEFT ARROW/RIGHT ARROW`
* To split a pane vertically: `CMD + d`
* To split a pane horizontally: `SHIFT + CMD + D`
* To move between panes: `CMD + [/]` (left or right bracket)
* To close a split pane or tab: `CMD + w`
* To clear the terminal screen: `CMD + k`

**Command Line Basic Commands:**
* `ls`: list files in current directory
* `cd directory`: change directories to directory
* `cd ..`: navigate up one directory
* `mkdir new-dir`: create a directory called new-dir
* `rm some-file`: remove some-file
* `man some-cmd`: pull up the manual for some-cmd
* `pwd`: find the path of the current directory
* mv `path/to/file new/path/to/file`: move a file or directory (also used for renaming)
* `find . -name blah`: find files in the current directory (and children) that have blah in their name
* `pwd | pbcopy`: copy working directory path

**Command Line Navigation:**
* To jump to beginning of line: `CRTL + a`
* To jump to the end of line: `CTRL + e`
* To cycle through previous commands: `UP ARROW/DOWN ARROW`

**Interactive Debugging Commands:**
* To open an interactive debugger right after it is called within a script:
```
ipdb.set_trace()
```
* `n`: next line
* `c`: continue to end (or next breakpoint)
* `s`: step into function call
* `b 25`: set a breakpoint at line 25
* `print a`: print the value of a
* `list`: see where you are

---
#### Day 1 Afternoon
### Python Introduction
**A Brief History:**
* Created in the late 1980's by Guido van Rossum
* Named after Monty Python's Flying Circus
* Python 1.0 released in 1994; Python 2.0 in 2000; Python 3.0 in 2008
* End of life (EOL) for Python 2 originally set for 2015, been extended to 2020

**Why Python:**
1) General purpose programming language that supports many paradigms
    * Imperative, object-oriented, procedural, functional
2) Interpreted, instead of compiled
    * Has rapid REPL (Read-Evaluate-Print-Loop)
3) Design philosophy emphasizes code readability
    * White space rather than brackets/braces determine code blocks
4) Efficient syntax
    * Fewer lines of code needed to do the same thing relative to C++, Java
5) Large development community
    * Large and comprehensive standard library (NumPy, SciPy, MatplotLib, Pandas, Scikit-Learn)
    * Open-source development

**Python 2 vs. 3:**
* Principle of Python 3: Reduce feature duplication by removing old ways of doing things.

| Python 2  | Python 3  |
|---|---|
| `print "Hello World"`  | `print("Hello World")`  |
| `3 / 2 = 1`, `(3 / 2.) = 1.5`  | `3 / 2 = 1.5`, `3 // 2 = 1`  |
| `types: str(), unicode()`  | `type: str()`  |
| `range(n)` - makes a list `xrange(n)` - makes an iterator  | `range(n)` - makes an iterator `list(range(n))` - makes a list   |
| `.items()` - makes a list `.iteritems()` - makes an iterator  | `.items()` makes an iterator  |
| `map()` makes a list  | `map()` map object `list(map())` makes a list  |
| `my_generator.next()`  | `next(my_generator)`  |

**Multiple Python Environments:**
* Use `conda` to create an environment in your Anaconda distribution:
```
$ conda create -n py2 python=2 anaconda (if you have Python 3 installed)
$ conda create -n py3 python=3 anaconda (if you have Python 2 installed)
```
* Then to activate the environment:
```
$ source activate py2 (or py3)
```
* And to deactivate the environment:
```
$ source deactivate
```

**Python Workflow:**
* In a script, start with an `if __name__ == __'main'__:` block
* `import` whatever you have to above the `if __name__ == '__main__:'` lock, start writing code below.
* In the Ipython console, run your code and then check to see if you have getting values you expect.
* If you are getting values you expect, start encapsulating your code into functions (and later classes) above the `if __name__ == '__main__':` block
* `import` these functions (and/or classes) into Ipython to make sure they work.

**Python Data Structures:**
* Lists: ordered, dynamic collections that are meant for storing collections of data about disparate objects (e.g. different types). Many list methods. (type list)
* Tuples: ordered, static collections that are meant for storing unchanging pieces of data. Just a few methods. (type tuple)
* Dictionaries: unordered collections of key-value pairs, where each key has to be unique and immutable (type dict) Hash map associates key with the memory location of the value so lookup is fast.
* Sets: unordered collections of unique keys, where each key is immutable (type set). Hash map associates key with membership in the set, so checking membership in a set is fast (much faster than a list).

**Lists:**
* Python lists are a flexible container that holds other objects in an ordered arrangement.
* A list is a general purpose, ordered data structure that allows you to change the data it holds (mutable).

**List Comprehensions:**
* Three ways perform operations on lists:
    * Example: Return the absolute value of each element in the list a.
1) For Loop:
```
abs_a_for = []
for elem in a:
    abs_a_for.append(abs(elem))
```
2) Map (Functional builtin):
```
abs_a_map = list(map(abs, a))
```
3) List Comprehension:
```
abs_a_lstcomp = [abs(x) for x in a]
```
* Functional programming (map, filter, reduce) is often faster, but is not as readable nor as flexible as list comprehensions. Advice: aim to code in comprehensions (list, dictionary, set) and go functional if speed becomes an issue.
* Filtering list comprehensions:
```
a = ['', 'fee', '', '', '', 'fi', '', '', 'foo', '', '', '', '', '', 'fum']
[x for x in a if x]
```

**Zip:**
* Useful way to combine same length iterables:
```
a1 = [1,2,3]
a2 = ['a','b','c']
list(zip(a1, a2))
```

**Tuples:**
* Tuples are a lightweight (meaning relatively small in memory) immutable brother/sister of the list.
* Tuples are immutable, ordered collections.
* Similar to lists, tuples are declared by passing an iterable to the tuple() constructor, with or without the syntactic parenthesis (this works because Python automatically interprets comma separated things that aren't specifically specified otherwise as tuples).
* If you want an ordered, lightweight data structure to hold unchanging data, use tuples.
* Three ways to make a tuple:
1)
```
my_first_tuple = tuple([1, 2])
```
2)
```
my_second_tuple = (1, 2)
```
3)
```
my_third_tuple = 1, 2
```

**Dictionaries:**
* Dictionaries are useful because they link data (the value) to a key for fast look-up. When you want to link data or any object to some entity, use a dictionary.
* To create a dictionary:
```
states_caps_dict = {'Georgia': 'Atlanta', 'Colorado': 'Denver', 'Indiana': 'Indianapolis'}
```
* To return a value from a dictionary:
```
states_caps_dict['Colorado']
```
* A better way to return a value from a dictionary:
```
states_caps_dict.get('New York', 'Not in the dictionary')
```
* Default dictionaries allow a default value to be set:
```
from collections import defaultdict
states_caps = defaultdict(lambda: 'State not found')
states_caps.update(states_caps_dict)
```

**Sets:**
* A set combines some of the features of both the list and the dictionary.
* A set is defined as an unordered, mutable collection of unique items. This means that a set is a data structure where you can store items, without caring about their order and knowing that there will be at most one of them in the structure.
* Sets use a hash to link each item to membership or not. If you are going to check membership in a data structure, use a set.
* To ways to create a set:
1)
```
my_set = set([1, 2, 3])
```
2)
```
my_other_set = {1, 2, 3}
```

**Python Good Pracetice:**
* Variable and functions names are `snake_case`, classes are `CamelCase`
* Avoid extraneous white space
* Lines should not exceed 80 characters
* Create documentation for all functions and classes!
* Use `for loops` instead of indexing into arrays
* Use `enumerate` if you need to index
* Use `with` statements when working with files
* Use list comprehensions
* `(if x:)` instead of `(if x == True:)`

**Zen of Python:**
1.
* This:
    ```
    def make_dict(x, y):
        return {'x':x, 'y': y}
    ```
    * Is more explicit than:
    ```
    def make_dict(*args):
        x, y = args
        return dict(**locals())
    ```
2)
* This:
    ```
    if x == 1:
        print('one')

    cond_1 = <complex comparison 1
    cond_2 = <complex comparison 2
    if (cond1 and cond2):
        # do something
    ```
    * Is more sparse than:
    ```
    if x == 1: print('one')

    if (<complex comparison 1 and <complex comparison 2): # do something
    ```
3)
* This:
    ```
    french_insult = ("Your mother was a hamster, and "
    "your father smelt of elderberries!"
    )
    ```
    * Is preferred line continuation to :
    ```
    french_insult = \
    "Your mother was a hamster, and \
    your father smelt of elderberries!"
    ```

---
### Python Week 1
#### Day 2 Morning
**Opening .tar Files:**
* To open .tar zip files, which we will download frequently from AWS:
1) Download file
2) Move the .tar file to where you want to open it.
3) Unzip it via command line:
```
tar xzf file_name.tar.gz
```
    * You can simply double-click the file on macs.
#### Object-Oriented Programming and Workflows
**Programming Paradigms:**
1) Imperative
    * How to do this and how to do that
    * Makes use of procedural programming (functions)
    * Examples: Fortran, Pascal, BASIC, C
2) Declarative
    * The how being left up to the language
    * Examples: SQL
3) Functional
    * Evaluate an expression and use the result
    * Example: Haskell, Lisp
4) Logic
    * Answer a question via search for a solution
5) Object-Oriented
    * Pass messages between meaningful objects
* All (pure) functional and logic-based programming languages are also declarative.
*  Functional and logical constitute subcategories of the declarative category

**What is Python?**
* Python is an imperative programming language that has both functional and object-oriented aspects. It is also interpreted, interactive, and iterative.
    * Interpreted languages are programming languages in which programs may be executed from source code form, by an interpreter.
        * This is in contrast to compiled languages.
    * In theory any language can be compiled or interpreted - it is just that one is generally done more often than the other.
    * Iterative languages are built around or offering generators.
* Like many popular languages today it is multiparadigm.

**Object-Oriented Programming:**
* Object-oriented programming (OOP) is a programming paradigm based on the concept of objects.
* In python, objects are data structures that contain:
    1) data (attributes)
    2) procedures (methods)
*Languages are not objects-oriented as much as the language environment supports OOP.*
    * Scala is another great example of a language that supports both functional and object-oriented scripting.
* Four Features of OOP:
    1) Message Passing: Objects communicate by sending a receiving messages (methods)
    2) Polymorphism: also called subtype polymorphism (can create instances of classes)
    3) Encapsulation: A language construct that facilitates the bundling of data with the methods operating on that data
        * Python does not support full encapsulation
    4) Inheritance: reuse of base classes to form derived classes
* To search all methods of an object that:
```
print([t for t in dir(np.array([1, 2, 3])) if 'set' in t.lower()])
```

**args and kwargs:**
* Convention and shorthand to refer to a variable number of arguments:
* For regular arguments, use args, which is a list:
```
**def a_func(*args): takes multiple arguments and returns a list
```
* For keyword arguments, use kwargs:
```
def a _func(**kwargs): takes multiple keyword arguments and returns a dictionary
```

**Magic Methods:**
* Special methods, indicated by double underscore (dunder), that you can use to give ubiquitous functionality of some operators to objects defined by your class.
* `__init__(self, ...)`: Constructor, initializes the class
* `__repr__(self)`: Defines format for how object should be represented
* `__len__(self)`: Returns number of elements in an object
* `__gt__(self, other)`: Provides functionality for the  operator
* `__add__(self, other)`: Provides functionality for the + operator
* `__iadd__(self, other)`: Provides functionality for the += operator

**Functions, Classes, Modules, and Packages:**
* Function: A block of organized, reusable code that is used to perform a single related action
* Class: A template of reusable code that creates objects containing attributes and methods
* Module: A file containing Python definitions and statements (e.g mylib.py)
* Package: Packages are a way of structuring Python’s module namespace
* Library: A generic term for code designed to be usable by many applications
* Script: a executable module
--
* All packages are modules, but not all modules are packages
* Any module that contains a __path__   attribute is considered a package. Packages may be installable via `run.py` and registered in PyPI

**OOP Design:**
* Build classes via:
    * Composition/aggregation:
        * Class contains an object of another class with the desired functionality
        * Often, just a basic type: str, float, list, dict, etc.
        * *HasA* - use members, aggregation
    * Inheritance:
        * Class specializes behavior of a base class
        * *IsA* - use inheritance
        * In some cases, derived class uses a mix-in class only to provide functionality, not polymorphism

**Interfaces:**
* An interface is a contract between the client and the service provider.
* Isolates client from details of implementation
* Client must satisfy preconditions to call method/function
* Respect boundary of interface:
* Library/module provides a service
* Clients only access resource/service via library
* Then bugs arise from arise incorrect access or defect in library

**Test Driven development (TDD):**
* Make sure your interface is intuitive and friction-free.
* Use unit tests or specification test
    * To verify interface is good before implementation
    * To exercise individual functions or objects before application is complete
    * Framework can setup and tear-down necessary test fixture
* Stub out methods using pass
* Test Driven Development (TDD):
    * Red/Green/Refactor
        1) Write unit tests
        2) Verify that they fail (red)
        3) Implement code (green)
        4) Refactor code (green)
* Use a unit test framework = unittest (best), doctest, or nose

**Class Constructor Template:**
```
"""
A generic template
"""

__author__ = "Mr. Baggins"

class SomeClass(object):
    """
    A generic class
    """

    def __init__(self):
        """
        Constructor
        """

    def __str__(self):
        """
        Identifing string
        """

        return("some class itentifier")


if __name__ == "__main__":
    print("\nRunning...")
    sc = SomeClass()
    print(sc)
    print(sc.__doc__)
```

**Python Debugger:**
* Steps:
1) Import:
```
import pdb
```
2) Place into a script (right before a problem line):
```
pdb.set_trace()
```
3) Run the file
4) The file will stop where the trace was set and you can examine the current namespace.
5) To continue to next line:
```
n
```
6) To continue to the next set_trace:
```
c
```
7) To quit the debugger:
```
q
```
* Common Commands:
`h` -- help
`b` -- set a break-point
`where` --  show call stack
`s` --  execute next line, stepping into functions
`n` --  execute next line, step over functions
`c` --  continue execution
`u` -- move up one stack frame
`q` -- quit
`d` -- move down one stack frame

---
### Python Week 1
#### Day 3 Morning
#### Introduction to SQL
**RDBMS:**
* A Relational Database Management System (RDBMS) is a type of database where data is stored in multiple related tables.
* The tables are related through primary and foreign keys.
* Every table in a RDBMS has a primary key that uniquely identifies that row
* Each entry must have a primary, and primary keys cannot repeat within a table
* Primary keys are usually integers but can take other forms

**Foreign Keys and Table Relations:**
* A foreign key is a column that uniquely identifies a column in another table
* Often, a foreign key in one table is a primary key in another table
* We can use foreign keys to join tables

**Why RDBMS:**
* RDBMS provides one means of persistent data storage.
    * Survives after the process in which it was created has ended
    * Written to non-volatile storage (stored even if unpowered)
    * Frequently accessed and unlikely to change in structure
    * Example: A company database that contains records of customers and purchases.
* RDBMS provide the ability to:
    * Model relations in data
    * Query data and their relations efficiently
    * Maintain data consistency and integrity

**Terminology:**
* Schema defines the structure of a tables or a database
* Database is composed of a number of user-defined tables
* Each table has columns (or fields) and rows (or records)
* A column is of a certain data type such as an integer, text, or date

**SQL Query Basics:**
* SQL queries always return tables.
* SQL is a declarative language, unlike Python, which is imperative. With a declarative language, you tell the machine what you want, instead of how ,and it figures out the best way to do it for you.
* SQL Queries have three main components:
    1) `SELECT`
        * What data (columns) do you want?
        * Example:
        ```
        SELECT
            *
        FROM
            table;
        ```
        * The asterisk means 'everything'
    2) `FROM`
        * From what location (table) you want it?
    3) `WHERE`
        * What data (rows) do you want?
        * WHERE specifies criterion for selecting specific rows (row filter)
        * Note that the WHERE statement must reference the original column name, not the alias.
            * However, WHERE can reference a table column that is not in SELECT
        * Example:
        ```
        SELECT
            column1
        FROM
            table
        WHERE
            condition;
        ```
        * We can specify multiple conditions on the WHERE clause by using AND/OR:
        ```
        SELECT
            column1
        FROM
            table
        WHERE
            condition1 AND condition 2
        ```
        * Note that comparison operator uses a single equal sign (= instead of ==)

**Formatting SQL Statements:**
* Unlike Python, whitespace and capitalization do not matter (except for strings)
* Convention is to use ALL CAPS for keywords.
* Line breaks and indentation help make queries more readable; use the following formet:
```
SELECT
    column1,
    column2,
FROM
    table;
```
**Aliases:**
*  Aliasing can be used to rename columns and even tables (more on this later).
* “AS” makes code clearer but is not necessary.
* Be careful not to use keywords (e.g. count) as aliases!
* In postgres, aliases can NOT be used in WHERE or HAVING clauses.
    * Aliases can be used in GROUP BY clauses
* Example:
```
SELECT
    column1 AS col1
FROM
    table;
```

**LIMIT and ORDER BY:**
* LIMIT specifies the number of records returned.
* ORDER BY is ascending by default; specify DESC for reverse sorting.
* Example:
```
SELECT
    *
FROM
    table
ORDER BY
    column
LIMIT 3;
```

**SELECT DISTINCT:**
* SELECT DISTINCT grabs all the unique records.
* If multiple columns are selected, then all unique combinations are returned.
```
SELECT DISTINCT
    columns
FROM
    table;
```

**Arithmetic Operators:**
* Arithmetic operators are similar to Python (except SQL uses ^ for exponents)
* Can be used with multiple columns (for example, adding one column value to another)
```
SELECT
    column1 * 2,
    column1 + column 2
FROM
    table;
```

**CASE WHEN:**
* CASE WHEN statement is the SQL version of an if-then-else statement.
* Used in the SELECT clause
* Can combine multiple WHEN statements and/or multiple conditionals
```
SELECT
    column 1,
    CASE WHEN column2 = 'value' THEN 1 ELSE 0 END AS new_column
FROM
    table;
```

**Aggregators:**
* Aggregators combine information from multiple rows into a single row.
* Other aggregators include MIN, MAX, SUM, OCUNT, STDDEV, etc.
```
SELECT
    COUNT(*)
    MAX(column1)
FROM
    table;
```

**Group By:**
* The GROUP BY clause calculates aggregate statistics for groups of data
* Any column that is not an aggregator must be in the GROUP BY clause
* Any column in the GROUP BY by clause must also appear in the SELECT clause (true of Postgres but not MySQL)
```
SELECT
    column1,
    COUNT(column2)
FROM
    table
GROUP BY
    column1
```
* Use HAVING instead of WHERE when filtering rows after aggregation
* WHERE clause filters rows in the root table before aggregation
* Like WHERE clause, HAVING clause cannot reference an alias (in Postgres, at least)

**Joining Tables:**
* The JOIN clause allows us to use a single query to extract information from multiple tables.
Every JOIN statement has two parts:
    1) Specifying the tables to be joined (JOIN)
    2) Specifying the columns to join tables on (ON)
* Join Types:
    * (INNER) JOIN: Discards any entries that do not have match between the keys specified in the ON clause. No null/nan values.
        * Discards records that do not have a match in both tables
    * LEFT (OUTER) JOIN: Keeps all entries in the left (FROM) table, regardless of whether any matches are found in the right (JOIN) tables. Some null/nan values.
        * Retains all records from the left(FROM) tables and includes records from right (JOIN) table if they are available.
    * FULL (OUTER) JOIN: Keeps the rows in both tables no matter what. More null/nan values.
        * Retains all records from both tables regardless of matches.

**Query Components vs. Order of Evaluation:**

| Order of Components  | Order of Evaluation  |
|---|---|
| `SELECT`  |  5 - Targeted list of columns evaluated and returned |
|   `FROM`  | 1 - Product of all tables is formed  |
| `JOIN/ON`  |   |
| `WHERE`  | 2 - Rows filtered out that do not meet condition  |
| `GROUP BY`  | 3 - Rows combined according to GROUP BY clause and aggregations applied  |
| `HAVING`  | 4 - Aggregations that do not meet that HAVING criteria are removed  |
| `ORDER BY`  | 6 - Rows sorted by column(s)  |
| `LIMIT`  | 7 - Final table truncated based on limit size  |
| `;`  |  8 - Semicolon included as reminder |

**Subqueries:**
* In general, you can replace any table name with a subquery.

**Subquery vs Temp Table vs Create/Drop Table:**
* All three of the following approaches yield the same result. The best one might depend on how many times you will reference the new table.
1) Subquery:
```
SELECT
    newTable.col1,
    newTable.col2
FROM (SELECT
        col1,
        col2,
        col3
        FROM
        anotherTable
    ) AS newTable;
```

2) Temp Table:
```
WITH newTable AS
    (SELECT
     col1,
     col2,
     col3
FROM anotherTable)

SELECT
  newTable.col1,
  newTable.col2
FROM
  newTable;
 ```

 3) Create New Table
 ```
 CREATE TABLE newTable AS
    (SELECT
        col1,
        col2,
        col3
    FROM anotherTable);

SELECT
  newTable.col1,
  newTable.col2
FROM
  newTable;
DROP TABLE newTable;
```

**Load .sql file into a DB:**
* One-time step to create a database and load .sql file. From the command line:
```
$ psql
$ CREATE DATABASE dbname;
$ \q
$ psql dbname < file.sql
```

**Running psql Scripts:**
* To run a .sql script in psql (make sure you're in the correct directory or use the full path):
```
$ \i file.sql
```
* To display the current database schema:
```
$ \d
```
* To display the schema of an individual table:
```
$ \d table
```

**Postgres Commands:**
* `\l` - list all databases
* `\d` - list all tables
* `\d` <table name - describe a table’s schema
* `\h` <clause - Help for SQL clause help
* `q` - exit current view and return to command line
* `\q` - quit psql
* `\i` script.sql - run script (or query)

---
#### Day 3 Afternoon
#### SQL Best Practices
1) Don't use SELECT * unless you are learning about the data and trying to see what is in a table.
    * People used to cite performance issues as a main reason for this. With today's technology that is not totally true any more.
    * But what is true is that SQL is already pretty slow, and no reason to make it pull in every column if you don't need them all.
    * It has "code smell" which means it's not wrong, it's just not a best practice.
    * It makes your code unreadable to anyone else skimming it (i.e... on GitHub)
2) The most important line of any SQL query you will ever write is your "FROM" statement.
    * Your FROM statement dictates how the rest of the code is going to be written.
        * Joins that link back to the FROM table instead of other join tables run are much less computationally intensive because SQL is not running through all of FROM and all of the other tables to finally get the records it needs.
    * I [Jordan] have never once had to write a RIGHT JOIN. If you have to, you can likely move that table to be your FROM table, and LEFT JOIN to the table you need to.
        * Not that this really matters, it's just easier to read.
    * Your FROM table should be a small-medium concise table. (i.e... a site directory).
3) Do not make your joins in your WHERE statement.
```
SELECT
     table1.this,
     table2.that,
     table2.somethingelse
FROM
     table1, table2
WHERE
     table1.foreignkey = table2.primarykey
AND (some other conditions)
```
* Way more computationally intensive, and much harder to read.
4) Don't use subselects (subqueries) if you can avoid it.
    * Again, there are computational and readability reasons.
    * Sometimes it’s necessary - but most of the time you can make it a temp table!
    * Faster!
    * Prettier!
    * Easier to read!
5) SQL isn't case sensitive - so make your code pretty.
    * This is different for everyone!
    * I [Jordan] have strong opinions on how "SELECT, CASE, WHEN, END, FROM, WHERE, ORDER BY, HAVING, and GROUP BY" should all be all capitalized. But that's just a personal preference.
    * Some people like commas in the SELECT before the columns, I prefer them to all be after the column.
    * Some people are crazy and like all their columns on one line, I like each one on it's own line.
    * Whatever you do just be consistent!

---
### Python Week 1
#### Day 4 Morning
#### Python SQL
**psycopg2:**
* A Python library that allows for connections with PostgresSQL databases to easily query and retrieve data for analysis.

**General Workflow:**
1) Establish connection to Postgres database using psycopg2
2) Create a cursor
3) Use the cursor to execute SQL queries and retrieve data
4) Commit SQL actions
5) Close the cursor and connection

**Create a database from an admin account:**
* Database creation should be reserved for only administrators.
    * Each database should have a list of non-admin users that are specific to that database.
    * Keeping this separation of roles is a setup that helps with security.
* Database setup:
```
$ psql -U postgres

CREATE USER ender WITH ENCRYPTED PASSWORD 'bugger';
CREATE DATABASE golf WITH OWNER ender;
```
* Check to see if the new database exists:
```
\list    # lists all the databases in Postgres
\connect # connect to a specific database
\dt      # list tables in the currently connected database
\q       # quit
```

**Connect to the database:**
* Connections must be established using an existing database, username, database IP/URL, and maybe passwords
```
import psycopg2
import getpass

upass = getpass.getpass()
conn = psycopg2.connect(database="golf", user="chrisfeller", password=upass, host="localhost", port="5432")
print("connected")
```

**Instantiate the Cursor:**
* A cursor is a control structure that enables traversal over the records in a database
* Executes and fetches data
* When the cursor points at the resulting output of a query, it can only read each observation once. If you choose to see a previously read observation, you must rerun the query.
* Can be closed without closing the connection
```
cur = conn.cursor()
```

**Commits:**
* Data changes are not actually stored until you choose to commit
* You can choose to have automatic commit by using autocommit = True
* When connecting directly to the Postgres Server to initiate server level commands such as creating a database, you must use the autocommit = True option since Postgres does not have "temporary" transactions at the database level
```
query = '''
        SELECT *
        FROM golf
        LIMIT 30;
        '''

cur.execute(query)
```
* Ways to look at the data:
```
cur.fetchone()
cur.fetchmany(10)
results = cur.fetchall()
```
* You can also iterate over the cursor:
```
cur.execute(query)
for record in cur:
    print ("Outlook: {} Temperature: {} Humidity: {}".format(record[1], record[0], record[2]))
```
* Transactions can be rolled back until they're committed:
```
conn.rollback()
```
* Close your connection:
```
conn.close()
```

**Review:**
* Connections must be established using an existing database, username, database IP/URL, and maybe passwords
* If you have no created databases, you can connect to Postgres using the dbname 'postgres' to initialize db commands
* Data changes are not actually stored until you choose to commit. This can be done either through conn.commit() or setting autocommit = True. Until commited, all transactions is only temporary stored.
* Autocommit = True is necessary to do database commands like CREATE DATABASE. This is because Postgres does not have temporary transactions at the database level.
* If you ever need to build similar pipelines for other forms of database, there are libraries such PyODBC which operate very similarly.
* SQL connection databases utilizes cursors for data traversal and retrieval. This is kind of like an iterator in Python.
* Cursor operations typically goes like the following:
    * execute a query
    * fetch rows from query result if it is a SELECT query because it is iterative, previously fetched rows can only be fetched again by rerunning the query
    * close cursor through .close()
* Cursors and Connections must be closed using .close() or else Postgres will lock certain operation on the database/tables to connection is severed.

---
### Python Week 1
#### Day 5 Morning
#### Basic Plotting
* Before every plot you make you should have a guess as to what it will look like.
    * This will help you catch bugs.

**matplotlib:**
* The most frequently used plotting package in Python.
* Written in pure Python
* Heavily dependent on numpy.
* Goals:
    * Plots should look great i.e. publication quality
    * Text should look great (antialiased, etc.)
    * Postscript output for inclusion with TEXT documents
    * Embeddable in a GUI for application development
    * Code should be easy to understand and extend
    * Making plots should be easy
    * When using in publications or white paper - `[1]`
* Seaborn lives on top of matplotlib in the same way that pandas lives on tip of numpy.

**Two Ways to Use matplotlib:**
1) pylab interface:
```
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabe('some numbers')
plt.show()
```
2) artist (matplotlib frontend):
```
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_ylabel('some numbers')
ax.plot([1, 2, 3, 4])
plt.show()
```

**matplotlib basic example:**
```
import matplotlib as plt
plt.figure(figsize=(8,6))
ax = plt.add_subplot(1,1,1)
...
ax.set_title('foo')
ax.set_ylabel('y')
ax.set_xlabel('x')
plt.savefig('foo.png', dpi=400)
```

**matplotlib plotting functions:**

| Command  | Description  |
|---|---|
| `plot`  | plot lines and/or markers  |
| `bar`  | bar plot  |
| `error bar`  | error bar plot  |
| `boxplot`  | boxplot  |
| `histogram`  | histogram  |
| `pie`  | pie charts  |
| `imshow`  | heatmaps/images  |
| `scatter`  | scatter plots  |

**matplotlib customization functions:**

| Command  | Description  |
|---|---|
| `text`  | add text to an axis  |
| `table`  | embed a table in the axes  |
| `subtitle`  | figure title  |
| `ylim`/`xlim`  | get/set the limits of x and y  |
| `imshow`  | heatmaps/images  |
| `xticks`/`yticks`  | get/set limits of tick locations  |
| `tight_layout`  | tries to make whitespace look right  |

**Axes Subplots:**
* To create four (1x1) subplots within one (4x4) figure:
```
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
plt.subplot(2, 2, 1) #Top Left Corner
plt.subplot(2, 2, 2) #Top Right Corner
plt.subplot(2, 2, 3) #Bottom Left Corner
plt.subplot(2, 2, 4) #Bottom Right Corner
```

**Style Sheets:**
* Two ways to implement a style sheet:
1)
```
with plt.style.context(’fivethirtyeight’):
plt.plot(x, np.sin(x) + x + np.random.randn(50))
plt.plot(x, np.sin(x) + 0.5 * x + np.random.randn(50))
plt.plot(x, np.sin(x) + 2 * x + np.random.randn(50))
```
2)
```
plt.style.use('fivethirtyeight')
```
* Be careful with this one in Jupyter

---
### Python Week 2
#### Day 1 Morning
#### Probability
**Why Learn Probability?**
* Machine learning came from statistics/probability.
* "Data Scientist (n.): Person who is better at statistics than any software engineer and better at software engineering than any statistician."
* Five Steps of Data Science:
    1) Obtain Data - Pointing and clicking does not scale
        a. Python and Unix command line functions
        b. Databases
    2) Scrub Data - The world is a messy place
        a. Python and Unix command line programs (sed & awk - text parsers, grep - searching text)
    3) Explore Data - You can see a lot by looking
        a. Python - Matplotlib & Pandas
        b. EDA - histograms, scatterplots
        c. Dimensionality Reduction & Clustering (Python's sklearn)
    4) Model Data - "All models are wrong, but some are useful"
        a. Python - sklearn
        b. Cross-validation
    5) iNterpret Data - "The purpose of computing is insight, not numbers"
        a. How can you learn from what your model is telling you to plan next steps, new approaches.
    * For #3, #4, and #5 above, you'll need your Probability, Statistics, and/or Machine Learning skills.
* Summary of motivation behind learning statistics/probability:
    1. As a data scientist, probability and statistics are a right of passage.
    2. Most ML algorithms have some basis in probability/statistics. The best ML folks know how each model works.
    3. Questions about probability and statistics are commonly asked in data science interviews.

**Logic Symbology:**

| Symbol       | Meaning           |
|--------------|-------------------|
| $\vee$       | or                |
| $\wedge$     | and               |
| $\neg$       | not               |
| $\implies$   | implies (if-then) |
| $\iff$       | if and only if    |
| $|$          | such that         |
| $\forall$    | for all           |
| $\therefore$ | therefore         |


**Formal Logic:**
* Statements are either True or False
$a \implies b$, means:
- If $a$ is True, then $b$ must be True.
- If $b$ is False, then $a$ must be False.

$a \wedge b$, means:
- (sometimes written $ab$)
- $a$ and $b$ are both True.

$a \vee b$, means:
- (sometimes written $a+b$)
- Either $a$ is True, or $b$ is True, or both are True.

$\neg a$, means:
- $a$ is False.

... and more.

A few extra rules which follow from above:

- $aa = a$
- $a+a = a$
- $ab = ba$
- $a + b = b + a$
- $a(bc) = (ab)c = abc$
- $a + (b + c) = (a + b) + c = a + b + c$
- $a(b + c) = ab + ac$
- $a + bc = (a + b)(a + c)$

**DeMorgan's Law:**
- $\neg (a \vee b) \iff \neg a \wedge \neg b$
- $\neg (a \wedge b) \iff \neg a \vee \neg b$

**Shortcoming of Formal Logic:**
* Formal Logic doesn't allow for the inference we want to do. Instead of absolute True/False, we want to reason about the plausibility of propositions.
* Instead of Formal Logic we want to capture the certainty of an outcome *x*, which we'll call 'probability' and write as *P(x)*.
$$0 \le P(x) \le 1$$

Where:
- $P(x) = 0 \iff$ the outcome $x$ is impossible, and
- $P(x) = 1 \iff$ the outcome $x$ is certain.
- $P(x) = 0.5 \iff$ the outcome $x$ will happen half of the time.

**Sample Space:**
* The set of all possible outcomes of an event.
* Also referred to as 'support' or the 'domain'.
* Examples:

| Event                                       | Sample Space |
|---------------------------------------------|--------------|
| coin flip                                   | {H, T}       |
| Human heights                               | $\mathbb{R}^+$ |
| Number of slices of pizza eaten before 10am | $\mathbb{N}$   |

* Example - What is the sample space of a 6-sided die?
    * Answer: 1, 2, 3, 4, 5, 6, null
        * The set of outcomes always includes the null value.
* Example - What is the sample space of your starting hand in blackjack?
    * Answer: All two-hand combinations of deck of cards: C(52, 2) plus null.

**Set Review:**
* Symbols:

| Symbol       | Meaning        |
|--------------|----------------|
| $\in$        | in             |
| $\cap$       | intersection   |
| $\cup$       | union          |
| $\emptyset$  | the empty set      |

* Operations:


| Operation                  | Definition                                               |
|----------------------------|----------------------------------------------------------|
| Union   | A $\cup$ B               |
| Intersection               | A $\cap$ B            |
| Difference                 | A \ B     |
| Complement                 | A^C                        |
| Disjoint                   | $A \cap B=\emptyset$             |

**Set Operations in Python:**
```
A = set([1, 3, 7, 9])
B = set([2, 4, 6, 7, 9])

| union, & intersection, - difference, ^ in one or the other but not both

Use one of the above to complete the following statements
print("{} are the numbers in A or B, but not both.".format(A ^ B))
print("{} are both in A and B.".format(A & B))
print("{} are the numbers in A after removing numbers that were also in B.".format(A - B))
print("{} are all the numbers in A and B".format(A | B))
```

**Probability Definitions:**
Ryan's definitions:
"The outcome of an unrealized event."
"A function which generates outcomes with a fixed, internal probability of each outcome."

Wikipedia:
"In probability and statistics a random variable ... is a variable whose possible values are numerical outcomes of a random phenomenon."
"A random variable is defined as a function that maps outcomes to numerical quantities (labels), typically real numbers."

Frank' definition:
"A random variable is a value drawn randomly from a probability distribution."

Random variables are hard to define...

A random variable is written as a capital letter. We write _specific outcomes_ as lowercase letters. That leads to the following notation:

$$P(X=x) = p$$

which we interpret as the probability of the random variable $X$ being realized as the _specific_ outcome $x$ is $p$.

**Shorthand Notation:**
When reasoning about probability, we often only write the random variable and leave it implied that we reference an unnamed, unspecific outcome.

$$P(A) = P(A=a)$$

Both notations above are unspecific about the outcome in question, but the former leaves the outcome name omitted.

Furthermore, we often omit the _and_ operator. All below are equivalent in meaning:

$$P(AB) = P(A,B) = P(A \cap B) = P(A=a \wedge B=b)$$

Similarly, we often write the _or_ operator with an `addition` symbol. All below are equivalent in meaning:

$$P(A+B) = P(A \cup B) = P(A=a \vee B=b)$$

**Laws of Probability:**
* Conditional Probability:
    $$P(A|B)$$

    We interpret this notation as the probability over the random variable $A$ given that we know the outcome of the random variable $B$. More concretely we can write:

    $$P(A=a | B=b) = p$$

    to say _specifically_ the probability of an outcome $a$ given a _specific_ outcome $b$ is $p$.

    The following is always true:

    $$P(AB) = P(A) P(B|A)$$

    $$P(AB) = P(B) P(A|B)$$

    * Conceptually, how can knowing that B occurred affect the probability of A?
        * Answer: Knowing that B occurred changes the sample space. If A overlaps with B, then it will increase the probability of A.
    * Define conditional probability in your own words. And then answer: Is P(A) ever equal to P(A|B)?
        Answer: The probability that A happens give that the sample space is confined to B. P(A) = P(A|B) when A is independent (does not effect) of B or A is a complete subset of B.

* Chain Rule:
    * If you apply the conditional probability rule over-and-over, you get the chain rule:

```
P(A, B, C) = P(A, B) * P(C | A, B) = P(A) * P(B|A) * P(C | A,B)
```



**Bayes' Rule:**
* The axioms above lead to Bayes' Rule:

$$P(A | B) = \dfrac{P(B | A) P(A)}{P(B)}$$

* What is the probability you are doped give you test positive? (Use Bayes' Rile and the Law of Total Probability)

|Conditional Events|Probability|
| --------- | ----------- |
| Probability test positive given doped | .99 |
| Probability test positive given clean | .05 |
| Prior probability of doped | .005 |

```
P(D) = Probability you doped
P(P) = Probability you tested positive
P(D|P) = (P(P|D) * P(D)) / P((P|C) * P(C) + P(P|D) * P(D))

(.99 * .005)/(.05 * (1-.005) + (.99 * .005))) = 0.09049
```

**Independence:**
If two random variables $A$ and $B$ are independent, then the following is true:

$$P(A|B) = P(A)$$

that is, the probability of $A$ is not affected by the outcome of $B$.

Independence implies that:

$$P(A \cap B) = P(A) P(B)$$

**Mutual Exclusivity:**
If two events $A$ and $B$ **cannot** occur simultaneously, then they are said to be "mutually exclusive" events. This implies that:

$$P(A \cap B) = 0$$

$$P(A \cup B) = P(A) + P(B)$$

**Combinatorics:**
* Combinatorics is the mathematics of ordering, choosing sets, etc. It is useful for counting events in your sample space.

**Factorial:**

$$n!=\prod\limits_{i=1}^{n} i$$

$$0! = 1$$

* There are $n!$ unique ways to arrange $n$ objects into a sequence (where order matters).
* Four ways to calculate factorial:
```
def my_factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

def factorial_recursive(n):
    if n == 1:
        return 1
    else:
        return n * factorial_recursive(n-1)

from functools import reduce

def fact_functional(n):
    return reduce(lambda x, y: x*y, range(1, n+1))

from math import factorial
factorial(52)
```

**Permutations:**
$$P(n,k)=\dfrac{n!}{(n-k)!}$$

There are $P(n,k)$ unique ways to arrange $k$ of $n$ objects into a sequence (where order maters).

* Code:
```
def perm(n, k):
    return math.factorial(n)/math.factorial(n-k)
```

**Combinations:**

$$C(n,k)={{n}\choose{k}} = {{n!}\over{(n-k)!k!}}$$

There are $C(n,k)$ unique ways to choose $k$ items from a set of $n$ items (where order doesn't matter).

* Code:
```
def comb(n, k):
    return math.factorial(n)/ (math.factorial(n-k) * math.factorial(k))
```

**Expected Value:**
$$E(X) = \sum_{s \in S} s * P(X=s)$$

It is the possible outcomes weighted by their respective probabilities of occurring.

**Variance:**
It is the expected value of $(X-E(X))^2$.

$$Var(X) = \sum_{s \in S} (s-E(X))^2 * P(X=s)$$

Variance gives the amount of "spread" in the possible outcomes.

**Covariance and Correlation:**
Covariance is the expected value of $(X - E(X))(Y - E(Y))$.

$$Cov(X, Y) = \dfrac{\sum_{i = 1}^n (X_i - E(X))(Y_i - E(Y))} {n-1}$$

Correlation is covariance normalized by the standard deviations of $X$ and $Y$.

$$Corr(X, Y) = \dfrac{\sum_{i = 1}^n (X_i - E(X))(Y_i - E(Y))} {\sqrt{ \sum_{i=1}^n (X_i - E(X))^2  \sum_{i=1}^n (Y_i - E(Y))^2 }}$$

**Problems with Correlation:**
* It only captures linear relationship, not other relationships
    a. Spearman's rho captures non-linear monotonic relationship.
* It doesn't capture slope at all; it only captures linear relationships (minus noise).
* There are many datasets which have the same correlation even though they are way different.

---

#### Day 1 Afternoon
#### Probability
**Continuous vs. Discrete (Random) Variables:**
* Discrete:  there is a positive, minimum difference between two values the variable can take
* Continuous: between two values the variable can take, there are uncountably infinite other values the variable can take
    * Another way to put it: There are measurable "gaps" between values of a discrete variable, where the gaps between values of a continuous variable can be made infinitesimal.

**Probability Mass Function (PMF):**
* r.v. is an abbreviation for random variable.
* The PMF of a r.v.  XX  gives the probabilities of every outcome in the support  SS  of r.v.  X

**Probability Density Function (PDF):**
* The PDF of a r.v.  XX  gives the relative likelihood of a random variable's support. PDFs should not be interpreted the same as a PMF; with a PDF you only can interpret area-under-the-curve.

**Discrete Distributions:**
* Bernoulli
    * Example: A single coin flip turns up heads with probability *p*.
        PMF: $P[success] = p$ , $P[failure] = 1-p$
    * Mean: *p*
    * Variance: *p*(1 - *p*)

* Binomial
    * Example: The number of coin flips out of n which turn up heads. *p* is the probability of heads for each trial.
        PMF: $P[X=k] = p(1-p)^{k-1}$
    * Mean: *np*
    * Variance: *np*(1-*p*)

* Geometric
    * Examples: The number of trials until a coin flip turns up heads.
        PMF: $P[X=k] = p(1-p)^{k-1}$
    * Mean: 1/*p*
    * Variance: 1-*p*/*p*^2

* Poisson
    * The Poission distribution gives the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate and independently of the time since the last event.
    * For instance: The number of taxis passing a street corner in a given hour (on avg, 10/hr, so  λ=10λ=10 ).
    * The number of calls a call center receives per minute is another example.
        PMF: $P[X=k] = \frac{ \lambda^k e^{-\lambda} }{ k! }$
    * Mean: $\lambda$
    * Variance: $\lambda$

**Continuous Distributions:**
* Uniform
    * Examples: Degrees between hour hand and minute hand ( a=0,b=360a=0,b=360 ).
        PDF: $f(x) = \frac{1}{b-a}$
    * Mean: $\frac{a+b}{2}$
    * Variance: $\frac{(b-a)^2}{2}$

* Normal (a.k.a., Gaussian)
    * Examples: IQ Scores (if $\mu = 100, \sigma = 10$)
        PDF: $f(x) = \frac{ 1 }{ \sigma \sqrt{2 \pi} } \exp(- \frac{ (x-\mu)^2 }{ 2 \sigma^2 })$
    * Mean: $\mu$
    * Variance: $\sigma^2$

* Exponential
    * Examples: Number of minutes until a taxi will pass street corner (if on average 10 taxis pass per hour; $\lambda=10/60$ the number of taxis per minute)
        CDF: $f(x) = \lambda \exp(\lambda x)$
    * Mean: $\frac{1}{\lambda}$
    * Variance: $\frac{1}{\lambda^2}$

**Summary of Distributions:**
Discrete:

    - Bernoulli
        * Model one instance of a success or failure trial (p)

    - Binomial
        * Number of successes out of a number of trials (n), each with probability of success (p)

    - Poisson
        * Model the number of events occurring in a fixed interval
        * Events occur at an average rate (lambda) independently of the last event

    - Geometric
        * Sequence of Bernoulli trials until first success (p)


Continuous:

    - Uniform
        * Any of the values in the interval of a to b are equally likely

    - Gaussian
        * Commonly occurring distribution shaped like a bell curve
        * Often comes up because of the Central Limit Theorem (to be discussed later)

    - Exponential
        * Model time between Poisson events
        * Events occur continuously and independently

**Joint Probability Distribution:**
* The probability of pairs of events from two (or more) random variables:
    $$P(A=a, B=b)$$
* If two random variables, also called a bivariate distribution or if more random variables, called a multivariate distribution.

---
### Python Week 2
#### Day 2 Morning
#### Estimating Distributions
* One of the major applications of statistics is estimating population parameters from sample statistics.
* By using probability distributions, and the few parameters needed to describe them, you can potentially make a simple model that describes the observed sample data:
    `sample_data = model + residuals`
        * sample data: a subset of the population
        * model: how data items relate to one another
        * residuals = the difference between the model and the measured data.

**scipy.stats Methods:**
* .pdf: Probability Density Function
    * Returns the probability that the variable x takes a specific value (more correctly: lies between a range of values)
* .cdf: Cumulative Distribution Function
    * Given a value x cdf returns the cumulative probability that x gets a value less or equal to x, or in other words 'lies in the interval (-inf, x}.'
    * Given an x value, what is the probability that the random variable variable takes a value less than or equal to the x value.
    * Example: What is the probability that the random variable is less than or equal to 5?
* .ppf: Percent Point Function (Inverse of CDF)
    * ppf returns the value of x of the variable that has a given cumulative distribution probability (cdf). Thus, given the cdf(x) of a x value, ppf returns the value of x itself, therefore, operating as the inverse of cdf.
* .sf: Survival Function (1 - CDF)
* .isf: Inverse Survival Function (Inverse of SF)
* Link: http://pytolearn.csd.auth.gr/d1-hyptest/11/distros.html
* Link: https://stackoverflow.com/questions/37559470/what-do-all-the-distributions-available-in-scipy-stats-look-like#37559471

**Parametric vs. Non-Parametric Models:**
* Parametric: We assume an underlying distribution, then we use our data to estimate the parameters of that underlying distribution.
    * Method of Moments (MOM)
    * Maximum Likelihood Estimation (MLE)
    * Maximum a Posteriori (MAP)
* Nonparametric: We do not assume any single underlying distribution, but instead we fit a combination of distributions to the observed data.
    * Kernel Density Estimation (KDE)

**When to use Parametric vs. Non-parametric Methods:**
* Parametric methods:
    1) Based on assumptions about the distribution of the underlying population and the parameters from which the sample was taken.
    2) If the data deviates strongly from the assumptions, could lead to the incorrect conclusion.
* Non-parametric methods:
    1) NOT based on assumptions about the distribution of the underlying population.
    2) Generally not as powerful - less inference can be drawn.
    3) Interpretation can be difficult...what does the wiggly curve mean?

**Distribution Estimation (Parametric):**
* After visually inspecting the data (e.g., via a histogram) we pick a distribution. Then we can use one of these three methods to estimate the parameters of the distribution:
    1) Method of Moments (MOM)
    2) Maximum Likelihood Estimation (MLE)
    3) Maximum a Posteriori (MAP)

**Method of Moments (MOM):**
* Based on the assumption that sample moments should provide good estimates of the corresponding population moments.
* Older, classic method. May be less efficient.
* Overview:
    1) Assumes an underlying distribution for your domain:
        E.g., Poisson, Bernoulli, Binomial, Gaussian
    2) Compute the relevant sample moments:
        E.g., Mean, Variance
    3) Plug the sample moments into the PMF/PDF of the assumed distribution.
* Example: Your website visitor log shows the following number of visits for each of the last seven days: [6, 4, 7, 4, 9, 3, 5]. What't the probability of zero visitors tomorrow?
    * Which underlying distribution should we assume?
        * Answer: Poisson, because it models # of events per unit of time.
    * What relevant sample moments do we compute:
        Answer: The sample mean is the only parameter required for a Poisson distribution: $\lambda$
        mean = 5.4286
    * Plug the mean into the scipy command:
    ```
    lambda = 5.4286
    scipy.stats.poisson.pmf(0, lambda)
    ```
* Example #2: You flip a coin 100 times. It comes up heads 52 times. What's the MOM estimate that in the next 100 flips the coin will be heads <= 45 times?
    * Which underlying distribution should we assume?
        * Answer: Binomial, because we are looking for a number of successes in a series of Bernoulli trials.
    * What relevant sample moments do we compute:
        Answer: The mean.
        mean = 52
    * Plug the mean into the scipy command:
    ```
    n = 100
    mu = 52
    p = 52/100
    binom = scipy.stats.binom(n,p)
    binom.pmf(45)
    ```

**Maximum Likelihood Estimation (MLE):**
* Trying to find the values of the parameters which would have most likely produced the observed data.
    * One way to maximize the likelihood equation is through taking the derivative.
* Overview:
    1) Assume an underlying distribution for your domain (just like with MOM)
        E.g. Poisson, Bernoulli, Binomial, Gaussian
    2) Define the likelihood function:
        * We want to know the likelihood of the data we observe under different distribution parameterization.
    3) Choose the parameter set theta that maximizes the likelihood function.

**Maximum a Posteriori (MAP):**
* Same thing as MLE, except that it includes a distribution of prior knowledge, Bayesian style.

**Kernel Density Estimation (KDE):**
* A non-parametric techniques
* Question: How can we model data that does not follow a known distribution?
    * Answer: Use a nonparametric technique.
* KDE is a nonparametric way to estimate the PDF of a random variable. KDE smooths the histogram by summing 'kernal functions' (usually Gaussians) instead of binning into rectangles
* Kernal functions have a bandwidth parameter to control under- and over-fitting.

---
#### Day 2 Afternoon
### Population Inference & Sampling
**Population Inference:**
* You take a sample of a population, calculate statistics on that sample and infer that the population has those exact statistics.

**Collecting Data: Taking a Sample:**
* A sample should be representative of the population.
* Random sampling is often the best way to achieve this.
    * Ideally, each subject has an equal change of being in the sample.

**Confidence Intervals:**
* A confidence interval (CI) is an interval estimate of a population parameter.
* How do we establish confidence intervals?
    1) Use a known relationship between sampling statistics and their distributions to estimate population statistics and place bounds around that estimate.
    2) Bootstrapping.

**Bootstrap Sampling:**
* Estimates the sampling distribution of an estimator by sampling with replacement from the original sample.
* Advantages:
    1) Completely automatic
    2) Available regardless of how complicated the estimator may be
* Often used to estimate the standard errors and confidence intervals of an unknown population parameter.
* Method:
    1) Start with your dataset of size n
    2) Sample from your dataset with replacement to create 1 bootstrap sample of size n
    3) Repeat B times
    4) Each bootstrap sample can then be used as a new sample for estimation and model fitting.
* You don't use CLT in bootstrapping
* When to Bootstrap:
    * When the theoretical distribution of the statistic (parameter) is complicated or unknown (e.g., median or correlation)
    * When the sample size is too small for traditional methods.
    * When collecting more data is cost-prohibitive.
    * When you favor accuracy over computational cost.
* Summary: In bootstrap sampling, we synthesize new samples that, collectively, form a stronger basis for measuring the accuracy of sample statistics than can be achieved from a single sample alone.

---
### Python Week 2
#### Day 3 Morning
#### Hypothesis Testing I
**What is a Hypothesis Test:**
* Hypothesis test evaluates two mutually exclusive statements about a population to determine which statement is best supported by the sample data.

**Hypothesis Testing:**
* Estimation
    * The value of the parameter is unknown
    * Goal is to find en estimate and confidence interval for the likely value
* Hypothesis Testing
    * The value of the parameter is stated
    * Goal is to see if the parameter value is different than the stated value.

**Hypothesis Testing Steps (Short Version):**
1) Formulate your two, mutually explosive hypotheses.
    * Null, Alternate
2) Choose a level of significance
    * alpha
3) Choose a statistical test and find the test statistic
    * t or z, usually
4) Compute the probability of your results (or more extreme results) assuming the null hypothesis is true
    * p-value
5) Compare p and alpha to draw a conclusion
    * p <= alpha, Reject Null in favor of alternative
    * p  alpha, Fail to reject Null

**Hypothesis Testing Steps (Long Version):**
1) Formulating your hypotheses
    * Null Hypothesis (Ho)
        * Typically a measure of the status quo (no effect)
        * The null hypothesis is assumed to be true
    * Alternative Hypothesis (Ha)
        * Typically the effect that the researcher hopes to detect
    * Your hypotheses must be mutually exclusive
2) Choosing your significance level
    * Significance level (alpha) is the probability of rejecting the null hypothesis when it is true.
    * Another way to think about it - if you pick a significance level of 0.05, that means you are comfortable with the fact that there is a 5% chance that you incorrectly reject the null hypothesis.
    * It's the change of making a type I error.
        * Type I error: False Positive
        * Type II error: False Negative
2.5) Adjusting your significance level
    * When making multiple comparisons, need to adjust significance rates of individual tests (αi) so that the overall experimental significance level remains the same (αE)
    * Called the Bonferroni correction
    * It's a straightforward, very conservative correction:
    αi = αE/*m*, where *m* is the number of comparisons
3) Choose your test and test statistic
    * Conceptually the test statistic represents the non-dimensonalized distance (in terms of standard deviations) between your two hypotheses.
    * Large number (~2) of standard deviations means they are far apart, low number of standard deviations (~0.5) means they are close together
4) Compute your p-value
    * p-value: The probability of your results (or more extreme) assuming the Null Hypothesis is true.
    * How:
        * Using tables (Z or t)
        * Using Python (CDF in scipy stats)
    * Your p-value will depend on what type of Hypothesis test you are performing:

|  Direction | Ho  | Ha  | P-value  |
|---|---|---|---|
|  2-sided test | =  | !=  | One half of P-value in each tail  |
| left-tail  | =  | <  | All of P-value in left tail  |
| right-tail  | <=  |   | All of P-value in right tail  |

5) Draw a conclusion
    * p <= alpha, reject null in favor of alternative
    * p  alpha, fail to reject null
    * Do you ever prove null hypothesis?
        * No, we assume it is true to begin with.
    * Do you ever prove the alternative hypothesis?
        * No, we just reject the null hypothesis in favor of the alternative hypothesis

---
#### Day 3 Afternoon
#### Hypothesis Testing II
**Chi-Squared Test:**
* The chi-squared test is a hypothesis test where:
    * the sampling distribution of the test statistic is a chi-squared distribution when the null hypothesis is true.
        * compare to z and t tests where the sampling distribution is assumed to follow the normal or t distributions.
    * Use: Is there a significant difference between expected and observed frequencies in one or more categories?
        * Goodness-of-fit
            * Example: Is a die fair based on 120 rolls
        * Independence
            * Is the amount you smoke independent of your fitness level?

**Experimental Studies vs. Observational Studies:**
* Experiments:
    * Randomly assign subjects to treatments
        * minimizes effect of confounding variables (more in a bit)
    * Apply treatments to subjects
    * CAN be used to determine causality
* Observational:
    * Subjects self-select into treatment groups
    * Confounding variables often a problem
    * CanNOT be used to establish causality

**Confounding Variables:**
* An attribute (third variable) correlated with both the dependent and the independent variable that affects the association between them.
    * Example:
        * Studying relationship between birth order (1st child, 2nd child, etc.) and the presence of Down's Syndrome in the child.

**AB-Testing:**
* A/B testing (sometimes called split testing) is comparing two versions of a web page to see which one performs better. It’s a Hypothesis Test.
* Steps:
    1) Get a good baseline (CTR of Null Hypothesis)
    2) Construct an alternative hypothesis
        * Changing the size of the “checkout” button will increase the CTR
    3) Decide for how long you will test (statistical power)
        * You want your test to be able to reject a false null hypothesis (detect an effect if it exists)
    4) Start the test, and while testing randomly direct users to one of the two web pages.
    5) Don’t peak!
        * But of course people do - and that get’s them into trouble
    6) Make your conclusion

---
### Python Week 2
#### Day 4
#### Power Calculation and Bayes
**Review:**
* Recall that a critical value is the point (or points) on the scale of the test statistic beyond which we reject the null hypothesis, and is derived from the level of significance.
    * Alternatively, you could see if the test statistic falls outside of a confidence interval that corresponds to your level of *a*.

**A Cheat Sheet for working with hypothesis tests:**

| Description  | Python  |
|---|---|
| Cumulative Distribution Function (CDF)  | `stats.norm.cdf(1.96) = 0.975`  |
| Inverse of CDF (Percent Point Funciton)  | `stats.norm.ppf(0.975) = 1.96`  |
| Evaluate the PDF  | `stats.norm.pdf(1.95) = 0.05`  |
| Calculate a 2-sided p-value with Z=1.99  | `2(1 - stats.norm.cdf(1.99)) = 0.047`  |

**Inverse of the cdf (percent point function):**
```
import scipy.stats as stats
alpha = 0.05
z = stats.norm.ppf(1-(alpha))
```
    * Answer: 1.64
* To find the percentile along the CDF associated with that critical value:
```
phi_z = stats.norm.cdf(z)
```
    * Answer: 0.95

**Exercises:**
1) Assume X follows a normal distribution. What is the P(90 < X < 95) where $\mu$=80, $\sigma$=12?
```
import scipy.stats as stats
stats.norm.cdf(95, 80, 12) - stats.norm.cdf(90, 80, 12)
```
    * Answer: 0.0966
    * Another way of doing it:
        ```
        import scipy.stats as stats
        dist = stats.norm(80, 12)
        dist.cdf(95) - dist.cdf(90)
        ```
2) What is the 2-sided p-value associated with a Z value of 2.0?
```
import scipy.stats as stats
2 * (1 - stats.norm.cdf (2.0))
```
    * Answer: 0.0455
    * Another way of doing it:
            ```
            2 * (stats.norm.sf(2.0))
            ```
* Coin Flip Example: /Users/chrisfeller/Desktop/galvanize/Week2/Day4/coin_flip.py

**Power Calculation:**
* $\alpha$
    * **Type I Error**
    * Reject *Ho* when *Ho* is True
    * False Positive
* $\beta$
    * **Type II Error**
    * Fail to reject *Ho* when *Ho* is False
    * False Negative
* 1 - $\beta$
    * **Power**
    * Reject *Ho* when *Ho* is False

**Why you need power:**
* To determine the number of samples needed to power a given study.
* We can obtain *n* samples so what level of statistical power will we have?
* Power is generally thought about with one of the following questions in mind:
    1) What power will my study have if...?
    2) If I want my study to have X power, what sample size will I need?

**Statistical Power:**
* The likelihood that we call something significant when there is something there to be detected.
* Equation: Power = 1 - $\beta$
* Another definition: The ability to detect signal, when signal exists.
* [Great Power Explanation](http://my.ilstu.edu/~wjschne/138/Psychology138Lab14.html)
* [Visualizing Power](http://rpsychologist.com/d3/NHST/)
* Python Calculation:
```
def calc_power(data, null_mean, ci=0.95):
   m = data.mean()
   se = standard_error(data)
   z1 = scs.norm(null_mean, se).ppf(ci + (1 - ci) / 2)
   z2 = scs.norm(null_mean, se).ppf((1 - ci) / 2)
   return 1 - scs.norm(data.mean(), se).cdf(z1) + scs.norm(data.mean(), se).cdf(z2)
```

**Four Things that Affect Statistical Power:**
1) Significance level
2) Effect Size - |$\mu$0 - $\mu$1|
3) Standard Deviation
4) Sample Size

**Decision Table for Hypothesis Testing:**

|  Statistical Decision | Ho True (No Effect)  | Ho False (Effect)  |
|---|---|---|
|  **Do not reject Ho (No Effect)** | Correct  | Type II Error (False Negative/$\beta$)  |
| **Reject Ho (Effect)**  | Type I Error ($\alpha$, False Positive)  | Correct   |

**Review:**
1) What is statistical power?
    * Answer: Ability to reject Ho when Ho is True
2) Name the factors that affect power?
    1) Significance Level
    2) Effect Size
    3) Standard Deviation
    4) Sample Size
3) When do we use a z-score vs. a t-score?
    * When we are comparing means, we use a z-score when n = 30; and t-score when n < 30. When we are comparing proportions we used a z-score when we have proportion  10 for both sides of the argument.

**Calculating Power:**
* Power calculations can be hard coded in python or computed with functions in R.

---
#### Day 4 Afternoon
#### An Introduction to R
**Introduction:**
* R and Python are the two big languages in data science.
* Some things are easier on one language than they are in the other one.
    * R is better for:
        * Linear models
        * Power calculations
* Python is a fairly general purpose language while R is specific for statistics.

**Data Types:**
```
x <- 4
class(x)
is.integer(x)

y <- as.integer(3.1)
z <- as.integer(TRUE)
z <- x  y

a <- as.character(99.9)
b <- paste("sam","gamgee",sep=" ")
c <- sprintf("%s has %d lembas breads", "sam", 59)
d <- sub("friendly with", "wary of", "Sam is friendly with spiders.")
```

**Basic Containers (Vectors, Matrices):**
```
x <- c(1,2,3,4)
y <- c("a", "b", "c", "d", "e")
z <- c(x,y)

x + 5
x + 1:4

u = c(10, 20, 30)
v = c(1, 2, 3, 4, 5, 6, 7, 8, 9)
u + v

A = matrix(
   c(1, 2, 3, 4, 5, 6), # the data elements
   nrow=2,              # number of rows
   ncol=3,              # number of columns
   byrow = TRUE)        # fill matrix by rows
A[2,3]
A[2,]
A[,c(1,3)]
```

**Vectors Containing Other Objects:**
```
n = c(2, 3, 5)
s = c("aa", "bb", "cc")
b = c(TRUE, FALSE, TRUE)
df = data.frame(n, s, b)

mtcars
mtcars["Mazda RX4", "cyl"]
nrow(mtcars)
ncol(mtcars)
head(mtcars)
mtcars[9]
mtcars[[9]]
mtcars["am"]
mtcars$am

mydata = read.csv("mydata.csv")
```

---
### Python Week 2
#### Day 5 Morning
#### Introduction to Probabilistic Programming
**Probabilistic Programming:**
* A probabilistic programming language makes it easy to:
    1) Write out complex probability models
    2) Subsequently solve these models automatically.
        * This is accomplished by:
            1) Random variables are handled as a primitive
                * Primitive: Smallest 'unit of processing' available to a programmer of a given machine.
            2) Inference is handled behind the scenes
            3) Memory and processor management is abstracted away
* One of the really nice things about probabilistic programming is that you do not have to know how inference is performed, but it can be useful.
* A probabilistic programming language will interpret the function as the distribution itself, and will give the programmer various tools for asking questions about the distribution.

**Why you would want to use probabilistic programming:**
1) Customization: We can create models that have built-in hypothesis tests
2) Propagation of uncertainty: There is a degree of belief associated with the prediction and estimation
3) Intuition: The models are essentially 'white-box,' which provides insight into our data.

**Why you might *NOT* want to use probabilistic programming:**
1) Deep Dive: Many of the online examples will assume a fairly deep understanding of statistics
2) Overhead: Computational overhead might make it difficult to be production ready.
3) Sometimes simple is enough: The ability to customize models in almost a plug-n-play manner has to come with some cost.

**Bayesian Terminology:**
* Prior *P(θ)* - one's belief about a quantity before presented with evidence.
* Posterior *P(θ|x)* - probability of the parameters given the evidence
* Likelihood - *P(x|θ)* - probability of the evidence given the parameters
* Normalizing constant - *P(x)*

**PyMC3:**
* Current best package for baysian probabilistic programming.
    * However, a package called edward might soon be the standard.
* Import syntax:
```
import pymc3 as pm
```

**Markov Chain Monte Carlo (MCMC):**
* It is an family of algorithms for obtaining a sequence of random samples from a probability distribution for which direct sampling is difficult.
* The sequence can then be used to approximate the distribution
* It allows for inference on complex models

**Bayes Factor:**
* It is the Bayesian way to compare models (comparable to the p-value in frequentist statistics).

---
#### Day 5 Afternoon
#### Multi-Armed Bandit
**Steps in Frequentist A/B Testing:**
1) Define a metric
2) Determine parameters of interest for study (number of observations, power, significance threshold, and so on)
3) Run test, without checking results, until number of observations have been achieved.
4) Calculate p-value associated with hypothesis test
5) Report p-value and suggestion for action

**Steps in Bayesian A/B Testing:**
1) Define a metric
2) Run test, continually monitor results
3) At any time, calculate probability that A >= B or vice versa
4) Suggest course of action based on probabilities calculated

**Subjective vs. Objective Priors:**
* Bayesian priors can be classified into two classes: objective priors, which aim to allow the data to influence the posterior the most, and subjective priors, which allow the practitioner to express his or her views into the prior.

**Empirical Bayes:**
* Empirical Bayes combines frequentist and Bayesian inference.
    * The prior distribution, instead of being selected beforehand is estimated directly from the data generally with frequentist methods.
    * This is poor practice as you are double-dipping into your data.
* It it not a true Bayesian method.

**The Gamma and Beta Distribution:**
* The two most common prior distributions in Bayes are Gamma and Beta.
* They each provide great flexibility in the shape of the distribution.
* They each work very well as conjugate priors.

**Conjugate Priors:**
* For a given likelihood distribution (binomial, poisson, etc), there is a type of distribution that you can choose for your prior that will be able to directly morph/update with new information into the posterior distribution: https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions
* Conjugate families of priors arise when the likelihood times the prior produces a recognizable posterior kernel
* For mathematical convenience, we construct a family of prior densities that lead to simple posterior densities.
* Conjugate prior distributions have the practical advantage, in addition to computational convenience, of being interpretable as additional data
* Probability distributions that belong to an exponential family have natural conjugate prior distributions

**Exploration vs. Exploitation:**
* Multi-armed Bandit Problem:
    * https://en.wikipedia.org/wiki/Multi-armed_bandit
    * Applies beyond A/B testing to internet display advertising, ecology, finance, clinical trails, and psychology.
* Exploration: Trying our different options to try and determine the reward associated with the given approach (i.e. acquiring more knowledge)
* Exploitation: Going with the approach that you believe to have the highest expected payoff (i.e. optimizing decisions based on existing knowledge)
* Types of strategies  to solve multi-armed bandit problem:
    1) Epsilon-Greedy Strategy:
        * Explore with some probability epsilon (often 10%)
        * All other times we will exploit (i.e. choose the bandit with the best performance so far)
        * After we choose a given bandit we update the performance based on the result.
    2) Upper Confidence Bound (UCB1):
        * Choose whichever bandit that has the largest value.
    3) Softmax:
        * Chooce the bandit randomly in proportion to its estimated value.
    4) Bayesian Bandits
        * The Bayesian bandit algorithm involves modeling each of our bandits with a beta distribution with the following shape parameters:
            * $\alpha$ = 1 + number of times bandit has won
            * $\beta$ = 1 + number of times bandit has lost
        * We will then take a random sample from each bandit's distribution and choose the bandit with the highest value.

---
### Python Week 3
#### Day 1 Morning
#### Linear Algebra
**Quick Reference:**

| NumPy command              | Note                                                        |
|----------------------------|-------------------------------------------------------------|
| `a.ndim`                     | returns the num. of dimensions or the rank                  |
| `a.shape`                    | returns the num. of rows and colums                         |
| `a.size`                     | returns the num. of rows and colums                         |
| `arange(start,stop,step)`    | returns a sequence vector                                   |
| `linspace(start,stop,steps)` | returns a evenly spaced sequence in the specificed interval |
| `dot(a,b)`                   | matrix multiplication                                       |
| `vstack([a,b])`              | stack arrays a and b vertically                             |
| `hstack([a,b])`              | stack arrays a and b horizontally                           |
| `where(a>x)`                 | returns elements from an array depending on condition       |
| `argsort(a)`                 | returns the sorted indices of an input array                |

**Transposes:**
* A matrix transpose is an operation that takes an *m x n* matrix and turns it into an *n x m* matrix where the rows of the original matrix are the columns in the transposed matrix, and visa versa.
* Two ways to transpose:
```
np.array([[3,4,5,6]]).T

np.array([[3,4,5,6]]).transpose()
```
* A transpose can be thought of as the mirror image of a matrix across the main diagonal.

**Dot Product:**
* If we have two vectors **x** and **y** of the same length *n*, then the dot product is given by matrix multiplication.
* If you’re multiplying a 3x2 matrix by a 2x1 matrix:
    * The two inner numbers must match (columns of first = 2, rows of second = 2)
    * The matrix produced will have dimensions equal to the outer numbers (rows of the first = 3, columns of the second = 1); you get a 3x1 matrix!

**Matrix Determinant:**
* The determinant is a useful value that can be computed for a square matrix.
    * Just as the name implies a square matrix is any matrix with an equal number of rows and columns.
* Matrices are sometimes used as the engines to describe processes. Each step of the process may be considered a transition or transformation and the determinant in these cases serves as a scaling factor for the transformation.

**Identity Matrix:**
* An identity matrix is a matrix that does not change any vector when we multiply that vector by that matrix.
* We construct one of these matrices by setting all of the entries along the main diagonal to 1, while leaving all of the other entries at zero.
```
np.eye(4)
```
* If such a matrix exists, then XX is said to be invertible or nonsingular otherwise XX is said to be noninvertible or singular

**Axes:**
* It's perhaps simplest to remember it as 0=down and 1=across.
* This means:
    * Use `axis=0` to apply a method down each column, or to the row labels (the index).
    * Use `axis=1` to apply a method across each row, or to the column labels.
* Best Explanation/Visual: https://stackoverflow.com/questions/25773245/ambiguity-in-pandas-dataframe-numpy-array-axis-definition
* One way to remember which axes to apply a function to in numpy - Think of a simple function like summing up your data in a given direction. If you are getting totals that you could add to the bottom of your data as a “totals row”, then you are applying the function along the rows (axis=0). If you are getting totals that you could add to the right side of your data as a “totals column”, then you are applying the function along the columns (axis=1).


---
#### Day 1 Afternoon
#### Pickling Files
* Example:
1) Load the file
```
trip_data = pd.read_csv('/data/201402_trip_data.csv', parse_dates=['start_date', 'end_date'])
```
2) Date Munge
```
trip_data['month'] = trip_data['start_date'].dt.month
trip_data['dayofweek'] = trip_data['start_date'].dt.dayofweek
trip_data['date'] = trip_data['start_date'].dt.date
trip_data['hour'] = trip_data['start_date'].dt.hour
```
3) Save/Load Pickle File
```
trip_data.to_pickle('/data/201402_trip_data.pkl')
trip_data_clean = pd.read_pickle('/data/201402_trip_data.pkl')
```

* Additional Example:
```
import pickle,os

results = {'a':1,'b':range(10)}
results_pickle = 'foo.pickle'

save it
print("...saving pickle")
tmp = open(results_pickle,'wb')
pickle.dump(results,tmp)
tmp.close()

load it
print("...loading pickle")
tmp = open(results_pickle,'rb')
loadedResults = pickle.load(tmp)
tmp.close()
```

* Saving Files with Numpy:
```
import numpy as np

a = np.arange(10)
b = np.arange(12)
file_name = 'foo.npz'
args = {'a':a,'b':b}
np.savez_compressed(file_name,**args)

npz = np.load(file_name)
a = npz['a']
b = npz['b']
```

---
### Python Week 3
#### Day 2 Morning
#### Linear Regression
**Assumptions of Linear Regression:**
1) Sample data is representative of the population.
2) True relationship between X and y is linear.
3) Feature matrix X has full rank (rows and columns are linearly independent)
4) Residuals are independent
5) Residuals are normally distributed
6) Variance of the residuals is constant (homoscedastic)
*Note: Linear regression does NOT assume anything about the distribution of x and y, it only makes assumptions about the distribution of the residuals, and this is all that's needed for the statistical tests to be valid.*

**Colinearity of the Feature Matrix:**
* Colinearity occurs in a dataset when two or more features (columns of X) are highly correlated. These features provide redundant information.x
* Perfect colinearity violates the full-rank assumption, and the feature matrix becomes singular or degenerate.
* Signs of colinearity include:
    * Opposing signs for the coefficients of the effected variables, when it’s expected that both would have the same sign.
    * Standard errors of the regression coefficients of the effected variables tend to be large.
    * Large changes to the regression coefficients when a feature is added or deleted (unstable solution).
    * Rule of thumb: a variance inflation factor (VIF) > 5 indicates a multicollinearity problem.
* Remedies to colinearity include:
    * Regularization (Ridge and Lasso)
    * Principal Component Analysis (PCA)
    * Engineering of feature that combines the affected features
    * Simply dropping one of the features (lazy, but viable option)

**Model Selection:**
* R^2 (model fit) is insufficient - more features means larger R^2.
    * Alternatives:
        * Adjusted R^2
        * Mallow's Cp
        * AIC
        * BIC

**Homoscedasticity/Heteroscedasticity**
![Residual Analysis for Homoscedasticity](http://slideplayer.com/slide/8135406/25/images/23/Residual+Analysis+for+Homoscedasticity.jpg)

* Homoscedasticity means “having the same scatter.” For it to exist in a set of data, the points must be about the same distance from the line.
* The opposite is heteroscedasticity (“different scatter”), where points are at widely varying distances from the regression line.

**Linear Model Example:**
* Function to create a linear regression summary.
* y is target variable
* x is predictor variable(s)
```
def lin_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    summary = model.summary()
    return summary
```
* Plot Residuals
```
def plot_student_resids(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    plt.scatter(model.fittedvalues, model.outlier_test()['student_resid'])
    plt.axhline(0, color='r', alpha=0.6, linestyle='dashed')
    plt.show()
```

**Outliers vs. High Leveraged Data Points:**
* An outlier is a data point whose response y does not follow the general trend of the rest of the data.
* A data point has high leverage if it has "extreme" predictor x values. With a single predictor, an extreme x value is simply one that is particularly high or low. With multiple predictors, extreme x values may be particularly high or low for one or more predictors, or may be "unusual" combinations of predictor values.
* [Outliers vs. High Leveraged Explanation](https://onlinecourses.science.psu.edu/stat501/node/337)

---
### Python Week 3
#### Day 3 Morning
#### The Bias-Variance Tradeoff
**Terminology:**
* Bias: The amount the expected value of the results (of a model) differ from the true value.
    * Expected value: The long-term average value of the result of an experiment or function (i.e. model).
* Variance: The expected value of the squared deviation of the results from the mean of the results.
* [Bias-Variance Video](http://work.caltech.edu/library/081.html)

**(Supervised) Machine Learning:**
* In supervised machine learning, features (a.k.a. predictors, exogenous variables, independent variables) are used to predict targets (a.k.a. endogenous variables, dependent variables).
* In supervised machine learning, we’d like to train a model on past data, and then predict the outcome on new data.
* The goal of unsupervised predictive models is to predict accurately on unseen data.
* The error of a model on unseen data is made up of three things:
    1) Bias
    2) Variance
    3) Sampling Error (Uncontrollable)
    * We never know how much of the error in our model is each of these things.

**Signal and Noise:**
* Most data is a combination of:
    1) Signal (that we want to model)
    2) Random noise (that we don't)
* Not knowing either of these quantities exactly affects our ability to make good predictive models.

**Unknown Signal & Noise leads to Possible:**
* Underfitting: The model doesn't fully capture the relationship between predictors and the target data
    * The model has *not* learned the data's underlying signal or true value.
* Overfitting: The model has captured noise in the data.
    * The model has learned the data's signal and the noise.

**Bias vs. Variance:**
* As you increase model complexity (higher order fit, or more features, etc.) to fit training data, you'll be able to fit the data better, but you also become more sensitive to noise in the data. Therefore as model complexity goes from simple to more complex, you'll likely transition from a model that gives high bias, low variance predictions to a model that gives low bias, higher variance predictions.
* The goal is to find the complexity sweet spot that has the lowest combination of bias and variance.

**Verbalizing the Bias-Variance Trade-Off:**
* A low bias model accurately predicts the population's underlying true value, or signal, and vice-versa.
* A low variance model's predictions don't change much when it is fit on different data from the underlying population (and vice-versa).
* A trade-off often exists between bias and variance because some amount of model complexity is often required to match the underlying population signal, but this same complexity also makes the fit more sensitive to variations in the data the model is fit on. So as bias decreases, variance often increases (and vice-versa).

#### Cross Validation
**Introduction:**
* So how is the 'correct' model complexity chosen if there is a tradeoff between bias and variance?
![Bias-Variance Tradeoff Visualized](http://scott.fortmann-roe.com/docs/docs/BiasVariance/biasvariance.png)

* Model complexity can be the order of fit, the number of features, interaction of features, number of splits (decision tree), number of neurons/layer in a neural net, number of layers...
* We can't do anything to reduce the sampling error from the population, but can we find the model complexity that minimizes the sum of the bias and variance?
* To do this, we have data with targets that the model has not trained on, we can plot the expected residuals for a given model complexity, and choose the complexity that gives the lowest residual.

**Cross Validation:**
* Definition: A model validation technique for assessing how the result of a statistical analysis will generalize to an independent data set.
* We use cross-validation for two things:
    1) Attempting to quantify how well a model (of some given complexity) will predict on an unseen data set.
    2) Tuning hyperparameters of models to get best predictions.
* Process:
    1) Split your data (after splitting out hold-out set) into training/validation sets.
        * 70/30, 80/20, or 90/10 splits are commonly used
    2) Use the training set to train several models of varying complexity.
        * e.g. linear regression (w/ and w/out interaction features), neural nets, decision trees, etc.
    3) Evaluate each model using the validation set.
        * calculate R2, MSE, accuracy, or whatever you think is best
    4) Keep the model that performs best over the validation set.
* K-Fold Cross Validation is the go-to.

**What to do if your model is overfitting:**
1) Get more data....(not usually possible/practical)
2) Subset selection: keep only a subset of your predictors (i.e., dimensions)
3) Regularization: restrict your model's parameter space
4) Dimensionality Reduction: project the data into a lower dimensional space

**Subset Selection:**
1) Best subset
    * Try every model. Every possible combination of *p* predictors
    * Computationally intensive. 2^*p* possible subsets of *p* predictors.
    * High chance of finding a 'good' model by random chance.
2) Stepwise
    * Iteratively pick predictors to be in/out of the final model.
    * Forward, backward stepwise
        * Forward: starting with just one and adding more features, one-by-one.
        * Backward: starting with them all, and removing one-by-one.
    * Sklean features only backward recursive elimination.

---
#### Day 3 Afternoon
### Regularized Linear Regression
**Difficulties in getting goo results from 'vanilla' regression:**
* Typically, in linear regression there are “many” features (columns), where “many” can be as few as 10.
* Difficulty #1: With many features, even your simplest model can be very flexible and overfit.
* Difficulty #2: With many features, unless you have lots of data there is a good chance that your data is sparse in the region you are trying to predict. Without training data in the region you are trying to predict, how can you make a good predictions?

**Curse of Dimensionality:**
* As the number of dimensions increase, the volume that data can occupy grows exponentially.
* [Curse of Dimensionality Explanation](http://blog.galvanize.com/how-to-manage-dimensionality/)




**Ridge Regression and LASSO:**
* Problem: As model complexity (e.g. order, # features) increases, the magnitude of the coefficients tends to increase AND there is more of a tendency to overfit.
* Solution: Limit overfitting by limiting the magnitude of the coefficients
* Changing the hyperparameter lambda changes the amount that large coefficients are penalized.
* Increasing lambda increases the model’s bias and decreases its variance.
* Ridge forces parameters to be small + Ridge is computationally easier (faster!) because it’s differentiable
* Lasso tends to set coefficients exactly equal to zero:
    * This is useful as a sort-of “automatic feature selection” mechanism,
    * Leads to 'sparse' models
    * Serves a similar purpose to stepwise features selection
* In scikit-learn:
    * lambda is equal to alpha
    * L1 = Lasso
    * L2 = Ridge
* Lasso penalties are better at recovering sparse signals.
* Ridge penalties are better at minimizing prediction error.

```
sklearn.linear_model.LinearRegression(...)
sklearn.linear_model.Ridge(alpha=my_alpha, ...)
sklearn.linear_model.Lasso(alpha=my_alpha, ...)
sklearn.linear_model.ElasticNet(alpha=my_alpha, l1_ratio = !!!!, ...) Wow!
```
* All have these methods:
```
.fit(X, y)
.predict(X)
.score(X, y)
```

**Summary:**
* Use regularization!
    * Helps prevent overfitting
    * Helps with collinearity
    * Gives you a knob to adjust bias/variance trade-off
    * Essentially, regularization penalizes the classifier for paying attention to features that don't help much toward answering the question at hand, while favoring features that provide relevant information.
* Don't forget to standardize your data!
    * Column-by-column, de-mean and divide by the standard deviation.
    ```
    from sklearn.preprocessing import StandardScaler

    always scale your data before regularization!
    instantiate StandardScaler object
    ss = StandardScaler()

    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)
    ```
* Lambdas control the size (L1 & L2) and existence (L1) of feature coefficients.
    * Large lambdas mean more regularization (fewer/smaller coefficents) and less model complexity.
* You can have it all! (ElasticNet)

**Clarification of Terminology:**
* Normalization rescales the values into a range of [0,1]. This might be useful in some cases where all parameters need to have the same positive scale. However, the outliers from the data set are lost.
* Standardization rescales data to have a mean (μ) of 0 and standard deviation (σ) of 1 (unit variance). For most applications standardization is recommended.
* Regularization is a technique to avoid overfitting when training machine learning algorithms, which works by penalizing more complex models with more features.

---
### Python Week 3
#### Day 4 Morning
#### Regular Expressions
**Introduction:**
* Available through the re module.
* Allows you to answer the questions, 'Does this string match the pattern? or Is there a match for the pattern anywhere in this string?'
* Email Address Example:
```
import re
my_str = "This (frodo@gmail.com) is what you are looking for, but here is my other address: samwise@minis-tirith.edu"
pattern = r"[A-Za-z0-9\.\+_-]+@[A-Za-z0-9\._-]+\.[a-zA-Z]*"
print(re.findall(pattern, my_str))
```
* Three things that go into each regular expression syntax call:
    1) A call to the standard library package `re`
    2) A pattern to find
    3) A string that we are searching against

**re module methods:**
* `re.search` - Does the pattern exist anywhere in the string?
* `re.match` - Does the pattern exist at the beginning of the sting? (useless?)
    * Search and match are similar
* `re.split` - Split the string on the occurance of a pattern
* `re.findall` - Return all the matches
* `re.finditer` - Like findall, but returns an iterator
* `re.sub` - Find a pattern and upon matching substitute it with another

**Basic Example:**
* Search for the world hello in the string 'hello hell world':
```
s = "hello hello world"
m = re.search("world",s)
print(m.group(0))
re.findall("world",s)
```
* Use `re.findall()` over `re.search()` unless you are only trying to see if a pattern is in a document.

**Basic Regular Expression Syntax:**
* `\d` Matches any decimal digit; this is equivalent to the class [0-9].
* `\D` Matches any non-digit character; this is equivalent to the class [^0-9].
* `\s` Matches any whitespace character; this is equivalent to the class [ `\t\n\r\f\v`].
* `\S` Matches any non-whitespace character; this is equivalent to the class [`^ \t\n\r\f\v`].
* `\w` Matches any alphanumeric character; this is equivalent to the class [`a-zA-Z0-9_`].
* `\W` Matches any non-alphanumeric character; this is equivalent to the class [`^a-zA-Z0-9_`].

**Examples:**
```
s = "Today you are you! That is truer than true! There is no one alive who is you-er than you! (Quote 42)"
```
* Match All of the 'T's:
```
print(re.findall(r"T",s))
```
* Match all upper case letters:
```
print(re.findall(r"[A-Z]",s))
```
* Match only the numbers:
```
print(re.findall(r"[0-9]",s))
```
* Match any word:
```
print(re.findall(r"\w+",s))
```
* Match any non-word
```
print(re.findall(r"\W+",s))
```

**Regular Expression Special Characters:**
* '.' - Matches any character except a newline.
* '^' - Matches the start of the string
* '$' - Matches the end of the string
* '* ' - Match 0 or more repetitions of the preceding RE
* '+' - Match 1 or more repetitions of the preceding RE
* '?' - Match 0 or 1 repetitions of the preceding RE
* '* ?, +?, ??' - non-greedy form of The '* ', '+', and '?' qualifiers
* {m} - Match exactly m copies of the previous RE
* {m,n} - Match from m to n repetitions of the preceding RE
* [  ] - Used to indicate a set of characters.
* '|' - A|B, match either A or B.

**Examples:**
```
s = "You have brains in your head. You have feet in your shoes. You can steer yourself any direction you choose.  --Dr. Suess"
```
* Match all words:
```
print(re.findall(r"\w+",s))
```
* Match only the first word:
```
print(re.findall(r"^\w+",s))
```
* Match the last word:
```
print(re.findall(r"\w+$",s))
```
* Match all letters that come after a single whitespace character
```
print(re.findall(r"\s(\w)",s))
```
* Create a pattern that will match 'you word'?
```
print(re.findall(r"You \w+|you \w+",s))
```

**The Backslash Scourge:**
* Regular expressions use the backslash character ('\') to indicate special forms or to allow special characters to be used without invoking their special meaning.
* To match a literal backslash, one might have to write '`\\`' as the pattern string, because the regular expression must be `\`, and each backslash must be expressed as `\` inside a regular Python string literal.

#### Indexing in NumPy
**np.where vs. masking:**
* `np.where` returns just the indices where the equivalent mask is true
    * this is useful if you need the actual indices (maybe for counts)
    * `np.where` returns a tuple
* Masking returns the actual values.

**Chaining Logic Statements:**
* `np.intersect1d` find the intersection of two arrays:
```
x = np.arange(10)
y = np.arange(5,15)
np.intersect1d(x, y)
```
* Another use case for `np.intersect1d`
```
x = np.arange(50,60)
np.intersect1d(np.where(x > 53),np.where(x<57))
```
* Doing the same thing using `np.where`:
```
np.where((x > 53) & (x < 57))
```
* Doing the same thing with masking:
```
x[(x > 53) & (x < 57)]
```

#### Tricks w/ Lists
**Introduction:**
* lambda - shorthand to create an anonymous function
* zip - take iterables and zip them into tuples
* map - applies a function over an iterable

**List Comprehensions w/ lambda and map:**
* Map vs. List Comprehension:
```
a = range(-5,5)
list(map(abs,a))
[abs(x) for x in a]
```
* Map w/ lambda vs. Comprehension:
```
a = range(-5,5)
[x**2 for x in a]
list(map(lambda x: x**2, a))
```
* Filter w/ lambda vs. List Comprehension
```
a = ['', 'fee', '', '', '', 'fi', '', '', '', '', 'foo']
list(filter(lambda x: len(x) > 0,a))
[x for x in a if len(x) > 0]
```

**Nested List Comprehensions:**
* Example:
```
l = [['40', '20', '10', '30'], ['20', '20', '20'], ['30', '20'], ['100', '100', '100', '100']]
print([[float(y) for y in x] for x in l])
```

**Zip:**
* Examples:
```
a1,a2 = [1,2,3],['a','b','c']
print(list(zip(a1,a2))
print(list(zip(*[a1,a2]))
```
* To create a dictionary from two lists:
```
a1,a2 = [1,2,3],['a','b','c']
dict(zip(a2,a1))
```

**Common Interview Questions:**
* Transpose a list of lists using map:
```
a = [[1,2,3],[4,5,6]]
list(map(list, zip(*a)))
```
* Transpose a list of lists using a list comprehension:
```
a = [[1,2,3],[4,5,6]]
[[row[i] for row in a] for i in range(len(a[0]))]
```

---
#### Day 4 Afternoon
#### Logistic Regression
**Linear Regression Review:**
* Regularization: technique to control overfitting by introducing an a penalty term over the error function to discourage coefficients from reaching large values
* Why do we use the term shrinkage when discussing regularization?
    * Regularization is also referred to as shrinkage because it reduced the values of the coefficients
* L1 and L2 Penalties
    * Interpretation: When two predictors are highly correlated L1 penalties tend to pick one of the two while L2 will take both and shrink the coefficients
    * In general L1 penalties are better at recovering sparse signals
    * L2 penalties are better at minimizing prediction error
    * So what type of regression is good for eliminating correlated variables?
        * L1 (Lasso)
    * And if you just want to reduce the influence of two correlated variables?
        * L2 (Ridge)
    * What if you want to use both?
        * Elastic Net

**Classification Problems:**
* Just like the regression problem, except that the values y we now want to predict take on only a small number of discrete values.
* We are doing Supervised learning: models using labels paired with features which can roughly be broken into:
    * Regression y is continuous (price, demand, size)
    * Classification: y is categorical or discrete (fraud, churn)
* Logistic regression is classification?
    * The output of a logistic regression model is (a transformation of) (Y|X). So in a sense it is still regression.
* Terminology:

    | Machine Learning  | Other Fields  |
    |---|---|
    | Features X  | Covariates, independent variables, regressors  |
    | Targets y  | dependent variable, regressand  |
    | Training  | learning, estimation, model fitting  |

**Comparing linear and logistic regression:**
* In linear regression, the expected values of the target variable are modeled based on combination of values taken by the features
* In logistic regression the probability or odds of the target taking a particular value is modeled based on combination of values taken by the features.
* Linear regression: If you increase X by 1 unit, y is predicted to increase by the given coefficient value (B1) respectively if the other features are held constant.
* Logistic regression: If you increase X by 1 unit, the log odds increases by the given coefficient value (B1) respectively if the other features are held constant.

**Validation:**
* Accuracy:
    * Overall proportion correct
    * `(TN + TP)/ (FP +FN +TN +TP)`
* Precision:
    * Proportion called true that are correct
    * `TP/(TP + FP)`
* Recall:
    * Proportion of true that are correct
    * `TP/(TP + FN)`
* sklearn classification report:
```
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
```

**Confusion Matrix:**
* A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.
* Example below: predicting if patients have a disease.
    * True Positive (TP): We predict they have the disease and they in fact do have the disease.
    * True Negative (TN): We predicted they do not have the disease and they in fact do not have the disease.
    * False Positive (FP): We predict that they do have the disease and the in fact do not have the disease.
    * False Negative (FN): We predict that they do not have the disease when in fact they do have the disease.
* Rates Computed from a Confusion Matrix:
    * Accuracy: Overall, how often is the classifier correct?
        * (TP+TN)/total
    * Misclassification Rate: Overall, how often is the classifier wrong?
        * (FP+FN)/total
    * True Positive Rate: When a patient has a disease, how often does the classifier predict they have the disease?
        * TP/actual yes
        * Also called 'Sensitivity' or 'Recall'
            ```
            number of true positives         number correctly predicted positive
            -------------------------- = -------------------------------------
            number of positive cases              number of positive cases
            ```

    * False Positive Rate: When a patient does not have a disease, how often does the classifier actually predict they do have the disease?
        * FP/actual no
            ```
            number of false positives     number incorrectly predicted positive
            --------------------------- = ---------------------------------------
             number of negative cases           number of negative cases
            ```
    * Specificity: When a patient does not have a disease, how often does the classifier predict they do not have the disease?
        * TN/actual no
        * Equivalent to 1 - False Positive Rate
    * Precision: When it predicts a patient has the disease, how often is it correct?
        * TP/predicted yes
    * Prevalence: How often does the yes condition actually occur in our sample?
        * actual yes/total
* Other terminology:
    * Positive Predicted Value: This is very similar to precision, except that it takes prevalence into account. In the case where the classes are perfectly balanced (meaning the prevalence is 50%), the positive predictive value (PPV) is equivalent to precision.
    * Null Error Rate: his is how often you would be wrong if you always predicted the majority class. (In our example, the null error rate would be 60/165=0.36 because if you always predicted yes, you would only be wrong for the 60 "no" cases.) This can be a useful baseline metric to compare your classifier against. However, the best classifier for a particular application will sometimes have a higher error rate than the null error rate, as demonstrated by the Accuracy Paradox.
    * Cohen's Kappa: This is essentially a measure of how well the classifier performed as compared to how well it would have performed simply by chance. In other words, a model will have a high Kappa score if there is a big difference between the accuracy and the null error rate.
    * F Score: This is a weighted average of the true positive rate (recall) and precision.
    * ROC Curve: This is a commonly used graph that summarizes the performance of a classifier over all possible thresholds. It is generated by plotting the True Positive Rate (y-axis) against the False Positive Rate (x-axis) as you vary the threshold for assigning observations to a given class.
* [Confusion Matrix Explanation](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)

![Shape Explanation](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/440px-Precisionrecall.svg.png)
* Precision has all the P's

---
### Python Week 3
#### Day 5 Morning
#### Data Visualization
**Exploratory and Explanatory:**
* Exploratory:
    * Understanding data without any expected agenda or narrative.
* Explanatory:
    * Using data-viz to editorialize and communicate a narrative.
    * Can be author-driven or viewer-driven.

**Types of Data:**
* Nominal scales are simply labels. They are used for labeling categorical variables without any quantitative value. For example: eye color (blue, green, brown, hazel, etc..); country of origin (US, Canada, Mexico, etc..)
* Oridnal scales represent order or rank. For example: Rate your satisfaction on a scale of 1-5; Olympic medalists Gold, Silver, Bronze; etc...
* Interval scales are measurable and have constant difference between values (as well as being orderable). For example, temperature given in degress Celcius, or time given in seconds. The difference between 0 and 30 deg. C is the same as between 70 and 100 deg. C.
* Ratio scales have order, constant difference between values and an absolute zero. Examples include temperature in Kelvin, height and weight.

**Types of Graphs:**
* [Types of Graphs](https://homes.cs.washington.edu/~jheer/files/zoo/)
* [Visual Encoding](https://homes.cs.washington.edu/~jheer/files/zoo/)

**Color Scales:**
* The default color scale is usually 'jet'?
    * It creates false boundaries between color.
    * Not well attunded to the human perception of colors, i.e., when people look at difference in this scale, they don't correctly perceive the differences in the encoded data.

**Color Specifications:**
* RGBa - Red, Green, Blue, alpha.
    * This gives how much of each color gets added from each channel (with alpha being the opacity). Since most displays work from adding red, green, and blue pixels this is pretty standard for digital graphics. RGB values will typically be in 8-bit scale (0-255), and rarely be scaled to [0, 1]. Alpha values are almost always [0, 1], or a percentage equivalent.
* Hexidecimal
    * This is an RGB encoding that scales values to two Hexidecimal values per channel (so ranging from '00' to 'FF'). It is usually denoted with a '#' in front. For example red looks like this: '#FF0000', and green: '#00FF00'. This is a common standard for web / HTML / CSS applications.
* CMYK - Cyan, Magenta, Yellow, Black (or Process Color).
    * For physically printed material, it will often go through a four process printing press layering ink from these four colors.
* HSV - Hue, Saturation, value.
    * This is often referred to as a 'cylidrical color geometry'. Hue is the angular component (with primary red at 0deg, primary green at 120deg, and primary blue at 240deg). Saturation is the radial dimension with range[0,1] with zero saturation being white for all hues, and a saturation of one being the full expression of that hue. Value is the central or axial dimension and is in the range [0,1] with black at zero value and the full hue at a value of one.
* Gamut
    * This is the range of all colors that can be represented. This is limited by the colorspace used, the process in which it is being printed, or just by human perception.
* Pixel
    * The smallest portion of an image is the pixel, a portmanteau of "picture" and "element". The 3d volume equivalent is a voxel.
* DPI and LPI (or Resolution)
    * Dots per inch and lines per inch. This is how many pixels will occupy a physical space. Screen resolution will typically be 72 or 92 dpi depending on the OS. For printing purposes, a dpi above 300 is desirable. Resolution, literally is the ability to resolve two items being distinct (think about statistical Power here).

**Color Tips:**
* Respect colorblindness A common condition that should affect your use of colors in data visualization.
    * Affects about 1/12 men and 1/200 women. ~4.5% of total population
    * Inability to distinguish red/green, or blue/green are most common.
    * Good choices are blue-yellow gradients
* Accurately Recreates Data e.g. Perceptually uniform The difference in perceived intensity of the color in the eye should match the difference in intensity of the data at all parts of the spectrum.
    * The new matplotlib color default "viridis" is chosen with this criteria in mind
* Prints in black and white e.g. perceptually uniform for lightness. Matplotlib has considered this too.

---
### Python Week 4
#### Day 1 Morning
#### k-Nearest Neighbors (kNN)
**Parametric vs. Non-Parametric Models:**

| Parametric  | Non-Parametric  |
|---|---|
| Fixed model structure  | Flexible model structure  |
| Fixed type/number of parameters  | Flexible type/number of parameters  |
| Some assumptions about the data  | Fewer assumptions about the data  |
| Easy to understand & interpret  | Harder to interpret  |
| Can train the model very quickly  |  Model takes longer to train |
| Can handler smaller datasets  |  Needs larger amounts of data |
| Performance can be poor  | Generally good performance  |
| Typically less overfitting problems  | Often problems with overfitting  |

* Regression is a parametric model, while kNN is non-parametric.


**kNN Introduction:**
* Uses information about similar datapoints to predict information about given datapoint.
* kNN can be used for both classification and regression
    * Example 1: Predicting type of animal (dog or horse) based on animals with similar characteristics (**classification**)
    * Example 2: Predicting the selling price of a house based on houses with similar characteristics (**regression**)

**Overview of kNN Algorith:**
1) Choose a value for the hyperparameter k - how many neighbors do you want to look at for a given data point?
2) Calculate the distance all data points are from each other
3) Find the closest k points to each data point, i.e. its neighbors
4) Make a prediction for each data point
    a. For classification, assign a data point’s category based on what category the majority of its neighbors are (e.g., if 3 neighbors are dogs and 1 neighbor is a horse, then you classify that point as a dog)
    b. For regression, calculate a data point’s value by taking the average value of its neighbors

**Ways to Measure Distance:**
1) Euclidean Distance
2) Manhattan Distance
3) Cosine Distance
     * Equal to 1 - Cosine Similarity
     * Best for text analysis
     * Small cosine distance means the objects are similar and a large cosine distance means the objects are farther apart.

**Choosing a Value for k:**
* Common starting point is n**0.5 (square-root of n)
* Use cross validation to find the best value for k.
* Smaller values for k are likely to overfit the model
* Larger values of k lead to less complex models

**Standardizing Features:**
* Always standardize your features: http://scikit-learn.org/stable/modules/preprocessing.html

**A Variation of kNN: Point Weighting:**
* Points that are closer are considered more important than points that are farther away
* Shorter distance - higher weighted “vote” for its own category

**kNN & The Curse of Dimensionality:**
* kNN does not work well with 5+ dimensions.
* In higher dimensions, data is more sparse/spread out, which leads to overfitting.
* To hedge against high-dimensional concerns when using kNN, use a higher k value to avoid overfitting.

**Advantages & Disadvantages of kNN:**
Advantages:
* Simple to implement
* The training phase is just storing the data in memory
* Works with any number of classes/categories
* Easy to add in additional data
* Only two hyperparameters - k & the distance metric

Disadvantages:
* Poor performance in high dimensions
* Very slow to run, especially with large datasets
* Categorical features don't work well with kNN

---
#### Day 1 Afternoon
#### Decision Trees
**Overview:**
* Supervised learning
* Non-parametric model
* A series of sequential splits
* Each split based on a single feature
* Splits are chosen to best separate the target
variable
* Target variable can be either categorical
(classification tree) or numerical (regression
tree)
* Overall goal: minimize error in your predictions
 of your target variable

 **Defining Concepts:**
 * Entropy - a measurement of the diversity in a sample
    * High entropy - a sample that is partly made up of dogs and partly made up of horses
    * Low entropy - a sample that is 100% dogs
* Decision trees split the data on features to decrease entropy as much as possible
    * We are trying to separate the classes, e.g., split the dogs from the horses
* Information gain - a way to measure how much we reduced the entropy by splitting the data in a particular way
    * If we decrease the entropy by a large amount, then we have a large information gain
    * `Information gain = parent entropy - mean child entropy`

**The Decision Tree Algorithm:**
1) Consider all possible splits on all features
    * If a feature is categorical, split on value or not value.
    * If a feature is numeric, split at a threshold: >threshold or <=threshold
2) Calculate & choose the 'best' split
    * Classification trees - the best split is the split that has the highest information gain when moving from parent to child nodes
    * Regression trees - the best split is the split that has the largest reduction in variance when moving from parent to child nodes

**Comparing Types of Information Gain:**
* Shannon Entropy: Measures the *diversity* of a sample.
* Gini Index: Measures the *probability of misclassifying* a single element if it was randomly labeled according to the distribution of classes in the sample.
* Gini index and shannon entropy will give you very similar answers.
* sklearn uses gini index instead of shannon entropy by default, but you can change that in the arguments.

**Recursion:**
* When a function calls itself

**Pruning: Preventing Overfitting:**
* Trees will overfit by default unless you direct them otherwise
* Pruning involves a bias-variance trade-off
* Pre-pruning ideas (pruning while you build the tree)
    * Leaf size: stop splitting when the number of samples left gets small enough
    * Depth: stop splitting at a certain depth (after a certain number of splits)
    * Purity: stop splitting if enough of the examples are the same class
    * Gain threshold: stop splitting when the information gain becomes too small
* Post-pruning ideas (pruning after you've finished building the tree)
    * Merge terminal nodes if doing so decreases error in your test set
    * Set the maximum number of terminal nodes; this is a form of regularization

**Algorithm Options for Decision Trees:**
* ID3: category features only, information gain, multi-way splits
* C4.5: continuous and categorical features, information gain, missing data okay, pruning
* CART: continuous and categorical features and targets, gini index, binary splits only

**Advantages & Disadvantages of Decision Trees:**
Advantages:
* Easy to interpret
* Non-parametric/more flexible model
* Can incorporate both numerical and categorical features
* Prediction is computationally cheap
* Can handle missing values and outliers
* Can handle irrelevant features and multicollinearity

Disadvantages:
* Computationally expensive to train
* Greedy algorithm - looks for the simplest, quickest model and may miss the best model (i.e., converges at local maxima instead of global maxima)
* Often overfits
* Deterministic - i.e., you’ll get the same model every time you run it

**Decision Trees in sklearn:**
* Uses the CART algorithm
* Uses Gini Index by default (but you can change it to entropy if you'd like)
* You can prune by varying the following hyperparameters: max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes
* Must use dummy variables for categorical features
* Does not support missing values (even though CART typically does)
* Only supports binary splits

---
### Python Week 4
#### Day 2 Morning
#### Ensemble Methods (Part 2): Bagging and Random Forests
**Bagging**
* A single, fully-grown decision tree is low bias, high variance.
    * If you gave it different data it would predict wildly.
* Bootstrapping: Take a bunch if low-bias, high-variance model and combine them to reduce the variance.
* Bagging is simply predicting an average of predictors made from bootstrapped samples.
* Boostrapping Basics:
    1) Bootstrapped trees provides unbiased, high variance predictors
    2) Averaged estimators are lower variance than single predictors
    3) Averaged predictors are still unbiased
    4) Bootstrapping explores the “data set”/“predictors” space: then uses the average of predictors we might see under sampling
* Bagging is a general ensemble procedure often used with trees

**Random Forests:**
* Individual decision trees are 'correlated' because:
    1) Bootstrapping samples are about the same (nothing we can do)
    2) Influential features tend to be the same (this we can change)
* Random Forest Basics:
    * Tree is constructed by recursive best splits
    * We only use a random subset of features at each split
    * Temporary exclusion: features includable again at each split.
    * Typically select the square root of p for classification and p/3 for regression, where p is the number of features.
* Random forests are often robust to tree characteristics: very large number of 'bushy' trees typically used and approach state of the art performance out of the box.

**Random Forest Tuning:**
* `g(p)`: Number of features considered at each split
* *`m`*: Number of trees (i.e. number of bootstrapped samples)
* *`nb`*: Sample size of each boostrapped sample
* `X`: Tree Characteristics (Penalizing/Regularization)

**Random Forest Computational Notes:**
* Trees can be computationally expensive to fit, but they can be fit in parallel
* Cross-validation can be computationally demanding but is still needed for methodology comparisons.

**Other:**
* Bootstrap by rows for bagging.
    * Bootsrapping *with* replacement
* Bagging by features for random forests.
    * Bootsrapping *without* replacement
* Tree < Bagging < Random Forest

---
### Python Week 5
#### Day 1 Morning
#### Support Vector Machines
**Support Vector Machines:**
* Have decreased in popularity since the popularization of deep learning and neural networks.
* Main objective: Find a hyperplane that separates classes in feature space.
    * If we cannot do this in a satisfying way:
        1) We introduce the concept of soft margins.
        2) We enrich and enlarge the feature space to make separation possible.
* Beneficial Characteristics of SVM:
    * Look for this hyperplane in a direct way
    * SVMs are a special instance of kernel machines
    * Kernel methods exploit information related to inner products
    * A kernel trick helps make computation easy
    * SVMs make implicit use of a L1 penalty
* [Video Explanation of SVM](https://www.youtube.com/watch?v=-Z4aojJ-pdg)



**High-level Overview:**
* SVM is similar to linear regression because at its center is a linear function.
* SVMs do not naturally provide probabilities, just a class prediction.

**Hyperplane:**
* A hyperplane in *p* dimensions is a flat affine subspace of dimensions *p*-1
* An optimal separating hyperplane is a unique solution that separates two classes and maximizes the margin between the classes.
* So what is a hyperplane?

| Dimensions  | Hyperplane  |
|---|---|
| 1D  | A Point  |
| 2D  | A Line  |
| 3D  | A Plane  |
|  *p*D | *p* - 1 dimensional hyperplane  |

**Margin:**
* The margin is defined as the perpendicular distance between the decision boundary and the closest of the data points.
* Maximizing the margin leads to a particular choice of decision boundary.
* The location of this boundary is determined by a subset of the data points, known as support vectors.
![Hyperplanes, Margins, and Support Vectors Visualized](https://3.bp.blogspot.com/-12I3KUZYAZU/WHI90_mZokI/AAAAAAAAFzg/qaaiCYvhwT41_rp0PEQjE7GFkPEtNrzkwCLcB/s1600/SVM%2Bin%2BR.png)

**Dot Product:**
* In python:
```
import numpy as np
a = np.array([[1,2,3,4]]).T
b = np.array([[4,5,6,7]]).T
print(np.dot(a.T,b)[0][0])
print(np.array([a[i]*b[i] for i in range(a.shape[0])]).sum())
```

**Norm of a Vector:**
* In python:
```
import numpy as np
x = np.array([1,2,3,4])
print(np.sqrt(np.sum(x**2)))
print(np.linalg.norm(x))
```

**Cosine Similarity:**
* In python:
```
import numpy as np
a = np.array([[1,2,3,4]]).T
b = np.array([[4,5,6,7]]).T
cos_sim = np.dot(a.T,b)/(np.linalg.norm(a.T)*np.linalg.norm(b))
```

**Maximum Margin Classifier:**
* The maximum margin hyperplane is defined only by the closest points. Support vectors can be useful, because we only need to use a subset of the inputs to make decisions.
* Potential issues with this:
    * Outliers can heavily influence which support vectors we choose
    * What if the data are not linearly separable?

---
#### Day 1 Afternoon
### Support Vector Machines
**Soft Margins:**
* Relax the idea of separating hyperplane
* Adjusting the margins is a way of regularizing, because the margin is now determined by more than just the closest point.
* Modifying the margins by tuning *C* hyperparameters.
    * *C* is a regularization parameter that modulates the misclassification error penalty.
    * Large *C* - hard margins (accuracy more important)
    * Small *C* - soft margins (generalization more important)
        ```
        from sklearn import svm
        clf = svm.SVC(kernel=’linear’,C=1.0)
        clf.fit(X,y)
        ```
    * The more points that are invovled in the estimation of the margin (the smaller value of *C*), the more stable the orientation of the margin.

**Kernels:**
* Enlarges the feature space by increasing dimensionality
* The decision boundary takes a more complex form (squares, cross-product, etc.)
* In the enlarged space the decision boundary is linear, but projected back down to two dimensions we get a non-linear function.
* Polynomials get crazy kinda fast
    * kernels are the way to account for this.

**The 'Kernel Trick':**
* Saves some computation
* Opens new possibilities. A kernel can operate in infinite dimensions!
* There are certain kernels that can be expressed as an inner product in much higher dimensional space. The inner product gives us the notion of similarity which is related to the inverse of distance function we already know how to compute.
* SVMs are not the only class of machine learning algorithm that can benefit from the kernel trick. The category of algorithms that use this are known as Kernel Machines.
*  [Linear and RBF Kernel Tutorial/Visualization](https://chrisalbon.com/machine-learning/svc_parameters_using_rbf_kernel.html)
* Gaussian kernal is the RBF kernal

**Grid Search:**
```
from sklearn.grid_search import GridSearchCV
svc_rbf = SVC(kernel=’rbf’)
param_space = {’C’:     np.logspace(-3, 4, 15),
’gamma’: np.logspace(-10, 3, 15)}
grid_search = GridSearchCV(svc_rbf, param_space,
                           scoring=’accuracy’, cv=10)
grid_search.fit(x, y)
print grid_search.grid_scores_
print grid_search.best_params_
print grid_search.best_score_
print grid_search.best_estimator_
```

**Bias-Variance Tradeoff:**
* Bias
    * A high-'bias' model makes many assumptions and prefers to solve problems a certain way.
    * E.g. A linear SVM looks for dividing hyperplanes in the input space only.
    * For complex data, high-bias models often *underfit* the data.
* Variance
    * A high-'variance' model makes fewer assumptions and has more representational power.
    * E.g. An RBF SVM looks for diving hyperplanes in an infinite-dimensional space.
    * For simple data, high-variance models often overfit the data.

**SVMs vs. Logistic Regression:**
* Logistic Regression maximizes the Binomial Log Likelihood function.
* SVMs maximize the margin.
* When classes are nearly separable, SVMs tends to do better than Logistic Regression.
* Otherwise, Logistic Regression (with Ridge) and SVMs are similar.
* However, if you want to estimate probabilities, Logistic Regression is the better choice.
* With kernels, SVMs work well. Logistic Regression works fine with kernels but can get computationally too expensive.
* With kernels, SVMs work well. Logistic Regression works fine with kernels but can get computationally too expensive.
* In higher dimensional space SVMs are generally preferred
* SVMs work best with wide dataframes

**Must be able to explain what maximum margin classifier is and how it relates to logistic regression. Must also know how the tuning parameter works.**

---
### Python Week 5
#### Day 2 Morning
#### Gradient Descent
**Review:**
* Regularization: Technique to control overfitting by introducing a penalty term to the error function in order to shrink the magnitude coefficients.
    * Regularization is also referred to as shrinkage because it reduces the values of the coefficients.
* Two kinds of regularization:
    1) L1 (Lasso)
    2) L2 (Ridge)
        * When two predictors are highly correlated L1 penalties tend to pick one of the two while L2 will take both and shrink the coefficients.
        * In general, L1 penalties are better at recovering sparse signals
        * L2 penalties are better at minimizing prediction error
        * L2 (Ridge) commonly outperforms L1 (Lasso) in regards to predictions.

**Logistic Regression:**
* In logistic regression we are trying to model the probabilities of the K classes via linear functions in x.
* Those models are usually fit by MLE
* Rather than model the response directly (like in linear regression ) logistic regression models the probability that y belongs to a category.
* Example: `P(asthma|years_smoked)` is between 0 and 1 for any `years_smoked`

**Optimization Methods:**
* What are we trying to accomplish in optimization?
    * Find the parameters of a model which maximize the likelihood of the data
    * Find the parameters of a model which minimize a cost function.
* Cost Function:
    * Maybe there is a cost (quality of care) associated with the total number of patients in an emergency room
    * We are interested in predicting profit at a business and there is some coset associated with producing the product
* Optimizing refers to the task of either minimizing or maximizing some function *f(x)* by altering x. This function is called an objective function and it we are minimizing it it has several names: cost function, loss function, error function.

**Checking Calculus w/ Sympy:**
```
import sympy
x = sympy.symbols(’x’)
sympy.diff(4*x**2)
```
* Answer: `8*x`

**Gradient Descent:**
* There are three main variants of gradient descent
    1) Batch gradient descent: computes the gradient of the cost function for the entire data set
    2) Stochastic gradient descent (SGD): performs gradient descent for each training example in x along with its corresponding y
    3) Mini-batch gradient descent: performs an update for every mini-batch of training examples

**Using Intuition:**
* The gradient is the multivariate analogue of the derivative.
* Geometrically the gradient is the direction of steepest descent.
* Derivative is the slope or rate of change.
![Visualizing Gradient Descent](http://blog.datumbox.com/wp-content/uploads/2013/10/gradient-descent.png)

**First and Second Derivatives:**
* A visual of first & second derivatives:
* Let’s say you have a graph of some function (far left).
* If you take the first derivative, you get how the function changes. Positive numbers if function is increasing/slope is positive. Negative numbers if the function is decreasing/slope is negative. Where the first derivative is 0 means the slope of the function is 0 and we’re at either a peak or a valley in the original function.
* Second derivative - how the slope changes (i.e., how the changes in the function change…) Positive numbers here if concave up (it’s at least a local minimum). Negative numbers here if it’s concave down (it’s a maximum).

![Visualize First and Second Derivatives](http://sites.math.rutgers.edu/~greenfie/currentcourses/math151/gifstuff/orig_complex3.gif)

**Gradient Descent Algorithm:**
* Pseudocode:
```
new_params = dict((i,0) for i in range(k))
while not has_converged:
    params = copy(new_params)
for theta in params:
update = alpha * gradient(theta, params)
new_params[theta] -= update
```

**Things to Remember:**
* Needs a differentiable cost/likelihood function
* Local maxima/minima
* Need to scale the features
* Although SGD seems nicer it can have performance issues associated with oscillation

---
### Python Week 5
#### Day 3 Morning
#### Gradient Boosting
**Class Example:**
* Each row takes a test in sequential order. Every new person in the row worked on incorrect or unfinished problems until the end of the row or all problems had been completed.
    * Takewaways: Sequential learners focus effort on outcomes predicted poorly by previous earners as opposed to those predicted accurately.
    * Weak learners consistently predict outcomes but with accuracy only slightly better than chance (so high bias/low variance)

**Gradient Boosting Intro:**
* Question that led to gradient boosting: Can many sequential weak learners be used to make a powerful prediction tool? And if so, will that prediction tool suffer from overfitting.
* Gradient Boosting is typically applied using trees.
    * Shallow trees (stumps?) provide weak learners/reduce training
    * A highly-pruned tree (highly-constricted) is known as a stump (aka weak learner).
* Gradient boosting is an ensemble of sequential 'weak learners'
* Pros:
    * Highly predictive without overfitting
* Cons:
    * Extensive tuning required
    * Unlike bagging/random forests, increasing the number of trees used for boosting can cause overfitting
    * Inherently non-parallelizable

**AdaBoost:**
* Adaptive Boosting

**XGBoost (eXtreme Gradient Boosting):**
* Percentiles binned features: faster/more explicit splitting
* Handles missing data
* Smart memory management/out-of-core computation

**Bagging/Random Forests/Boosting Review:**
* Bagging: The average of many low bias, high variance trees created sequentially; the trees are highly correlated.
* Random Forest: The average of many low bias, high variance trees created sequentially; selects a subset of features at each split to create trees with more differences.
* Boosting: The average of many high bias, low variance trees created sequentially; given that there are only a few splits per tree the trees have some differences.

**Grid Search Code:**
* Gridsearch to find optimal hyperparameters for Random Forest model
```
random_forest_grid = {'max_depth': [3, None],
                  'max_features': ['sqrt', 'log2', None],
                  'min_samples_split': [2, 4],
                  'min_samples_leaf': [1, 2, 4],
                  'bootstrap': [True, False],
                  'n_estimators': [10, 20, 40, 80],
                  'random_state': [1]}

rf_gridsearch = GridSearchCV(RandomForestRegressor(),
                             random_forest_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='mean_squared_error')
rf_gridsearch.fit(X_train, y_train)

print("best parameters:", rf_gridsearch.best_params_)
```

---
### Python Week 5
#### Day 5 Morning
#### Neural Networks: Multi-layer Perceptron
**Intro Links:**
*  [DL4J's Introduction to Deep Neural Networks Overview](https://deeplearning4j.org/neuralnet-overview)
*  [Carnegie Mellon's C.S. lecture: "Neural Networks: A Simple Problem](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/nn.pdf)
* [i am trask's "Neural Network in 11 lines of Python"](https://iamtrask.github.io/2015/07/12/basic-python-network/)

**Neural Net Motivation: Our Brain the Ultimate Parallel Computer:**
* If we want a computer to be able to perform the same tasks as our brain, we should look to how the brain works for inspiration.

**Neural Net History:**
* Biomimicry
    * McCulloch-Pitts first neuron model in 1943
* Improvements to MCP neuron
    * 1949, Donald Hebb, neuropsychologist
    * 1957, Frank Rosenblatt invents the Perceptron
        * Initializes the weights to random values (e.g. -1.0 to 1.0)
            * Weights are the quantified strength of a connection between neurons
        * Weights change during supervised learning according to the delta rule, ~(yi - yp). After a certain number of training passes through the whole training set (a.k.a. the number of epochs) stop changing the weights.
        * Implements a learning rate that affects how quickly weights can change.
        * Adds a bias to the activation function that shifts the location of the activation threshold and allows this location to be a learned quantity (by multiplying it by a weight).
* Set-back: the XOR affair
    * Perceptrons, an introduction to computational geometry (book by Minsky and Papert 1969)
    * “critics of the book state that the authors imply that, since a single artificial neuron is incapable of implementing some functions such as the XOR logical function, larger networks also have similar limitations, and therefore should be dropped.”
* XOR affair solution: go deeper (multi-layer)
    * Single layer perceptron networks are limited to being linear classifiers. Not true of deeper MLP (multi-layer perceptron) networks.
* Breakthroughs for multi-layer networks:
    * Back propogartion
    * Compute Power
* (Artificial) Neural Networks:
    * Artificial neural networks (ANNs) are a family of models inspired by biological neural networks
    * Systems of interconnected ‘neurons’ which exchange messages...
    * The connections have numeric weights that can be tuned based on experience, making neural nets ... capable of learning.
* [Neural Network Visualized](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.50821&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

**Types of Artificial Neural Nets:**
* Multilayer Perceptron (MLP): Standard algorithm for supervised learning. Used for pattern, speech, image recognition.
* Autoencoder: Used for unsupervised learning of efficient codings, usually for dimensionality reduction, though recently to make generative models.
* Convolutional Neural Network: Node connections are inspired by visual cortex. Uses kernels to aggregate/transform information in network. Used for image, video recognition, natural lang. proc..
* Connections between nodes can be cyclical, which gives the network memory. Used for sequences: handwriting, speech recognition, time series.

**'Vanilla' Neural Network: Multilayer Perceptron (MLP):**
* The Parts:
    * Computation unit: perceptron (neuron)
        * Input, weights, summation, activation function, prediction
    * Use a single neuron to explain:
        * Feed-forward
        * Backpropogation (get gradients)
        * Gradient descent and its flavors (stochastic, batch, minibatch)
    * Gradient descent solution optimizer
    * Mulit-layer network

**Perceptron (Neuron, neurode, node, unit):**
![In the Brain](http://cs231n.github.io/assets/nn1/neuron.png)
![In a Neural Net](http://cs231n.github.io/assets/nn1/neuron_model.jpeg)

**Activation Functions:**
* Sigmoid
* ReLU
* tanh
* Leaky ReLU
*Your choice of an activation function for a given layer of your network should be informed by literature and your experience.*

**Computations:**
* Goal: Minimize the error or loss function - RSS (regression), misclassification rate (classification) - by changing the weights in the model.
* Back propogation: the recursive application of the chain rule back along the computational network, allows the calculation of the local gradients, enabling....
* Gradient descent: to be used to find the changes in the weights required to minimize the loss function

**MLP pseudocode from scratch:**
* Showing stochastic approach
```
for a desired number of epochs:
    for each epoch:
         for each row of X, y in inputs, targets:
             Feed-forward to find:
                 node activations
                 prediction
             Calculate loss
             Backpropagate to find the gradient of the loss w.r.t. the weights
             Use gradient descent to update the weights
         print the training error
# now that the weights are trained
for all test data:
    Feed-foward to find the predictions
print the test error
```

**Defining a model - available hyperparameters:**
* Most are particular to your application: read the literature and start with something that has been shown to work well
* Structure: the number of hidden layers, the number of nodes in each layer Activation functions
* Weight and bias initialization (for weights Karpathy recommends Xavier init.)
* Training method: Loss function, learning rate, batch size, number of epochs
* Regularization: Likely needed! (NN very complex models that will overfit)
    * Weight decay (L1 & L2), early stopping, dropout
* How do you figure out the right parameters:
    1) Search the literature for similar use case and architecture to define a baseline model.
    2) Once you have that, start varying the hyperparameters and check performance using cross-validation.

---
#### Day 5 Afternoon
#### Neural Networks: Autoencoders
**Autoencoders:**
* Form of self-supervised learning
    * It's outputs are the inputs
* Autoencoders encode (embed) the input into a lower dimensional space
    * It's lossy
* Can be used with many types of data: typical rows & columns, images, sequences.
* Common uses:
    * Reduction in dimensionality of data before sending to another algorithm
    * Denoising (noise added to input, output is input without noise)
    * Recently, generative (learns parameters of distributions)
* Example: Face Completion

---
### Python Week 6
#### Day 1 Morning
#### Convolutional Neural Networks (CNNs)
**Neural Network Architecture:**
[Link](http://www.asimovinstitute.org/neural-network-zoo/)

![CNN](https://www.mathworks.com/content/mathworks/www/en/discovery/convolutional-neural-network/jcr:content/mainParsys/image_copy.adapt.full.high.jpg/1508999490138.jpg)

[Link](https://ezyang.github.io/convolution-visualizer/index.html)

**functions of CNNs:**
* Classification
* Retrieval
* Detection
* Segmentation

**Convolution Layering**
* Steps:
    1) Line up the filter (i.e., mini piece of the image) and the original image.
    2) Go through each pixel in the original image one by one and multiply it by the corresponding pixel in the filter
    3) Find the total sum of all of these pixel multiplications
    4) Divide this sum by the total number of pixels in the
    filter to determine its value for the activation map.
* For the convolution layer, you cannot apply a filter with a stride that does not fit clearly.
* In practice, it is common to zero pad the border to make the stride fit.
* Calculating Output Size:
    N - Original image dimension
    F - Filter Dimension
    S - Stride
    P - Padding
    Outut Size: ((N - F + 2P)/S) + 1

**Pooling: Shrinking the image stack:**
* Steps:
    1) Pick a window size (usually 2x2 or 3x3 pixels)
    2) Pick a stride (usually 2 pixels)
    3) Walk your window in strides across your filtered images.
    4) From each window, take the maximum value

**Concept Review: Activation Layer:**
* CNNs use activation function(s) just like MLPs
* Types of activation functions in this layer:
    * ReLU - maps all values between 0 and ∞
    * Leaky ReLU - instead of mapping all negative values to 0, these values have a small negative slope of about 0.01
    * Tanh - maps all values between -1 and 1
    * Sigmoid - maps all values between 0 and 1

**Concept Review: Fully Connected Layer:**
* Computing our votes for X & O through feed forward calculations
* The votes are the summation of the dot products between the activations & the weights
* We classify the image as an X or O depending on which has the higher value for a vote
* Both the filters used (e.g., the 3x3 diagonal line images) and voting weights are updated through backpropagation
* You can think of this as an MLP just stuck onto the end of the CNN

**Advantages of CNNs:**
* CNNs are great at finding patterns and using these patterns to classify images.
* If you can make your data look like images, then they are super useful!
* You can use any 2D, 3D, or even higher dimensional data in a CNN.

**Disadvantages of CNNs:**
* CNNs only capture local “spatial” patterns in data.
* If your data is just as useful after swapping any of your columns with each other, then you can’t use CNNs.

---
#### Day 1 Afternoon
#### Recurrent Neural Networks (RNNs)
**Applications for RNNs & LSTMs:**
* Pattern recognition: handwriting, captioning images
* Sequential data: speech recognition, stock price prediction, and generating text and news stories
* Translating between...
    * Speech
    * Text
    * Difficult languages
    * Audio
    * Video
* Physical processes, including robotics
* Anything embedded in time

**LSTMs:**
* “Long short-term”
    * Long term memory refers to the learned weights
    * Short term memory refers to the values related to gates that change with each step through the LSTM
* LSTMs have a more complex structure than RNNs
* There are different types of LSTMs, that have different components or connections
* They extend the ability of RNNs to remember (and forget) deeper into the past
* LSTMs seek to address RNNs’ exploding/vanishing gradients problem
    * Continuing to multiply a quantity by a number >1 or <1 can cause that quantity to become very large (exploding) or very small (vanishing)
    * Because the layers of RNNs relate to each other through multiplication, their derivatives can have this problem
    * If we don’t know the gradients, we can’t adjust the weights to continue learning

**LSTM Terms:**
* Input gate: how much of the newly computed ideas you want to let through
* Forget gate: how much of the previous information you want to let through
* Output gate: how much of the internal information you want to move forward to the next steps
[LSTM Introduction](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---
### Python Week 6
#### Day 2 Morning
#### Natural Language Processing
**What is NLP?:**
* Conversational Agents
    * Siri, Cortana, Google Now, Alexa
    * Talking to your car
    * Communicating with robots
* Machine Translation
    * Google Translate
* Speech Recognition, Speech Synthesis
* Lexical Semantics, Sentiment Analysis
* Dialogue Systems, Question Answering
* NLP is just a way to take text data that isn’t in the form we’re used to (i.e., with features as columns and rows as samples) and change the data to that typical format so we can use it in all our other algorithms.

**NLP and AI:**
* The ultimate goal of NLP is to the fill the gap in how the humans communicate (natural language) and what the computer understands (machine language).
* Why deep learning is needed in NLP:
    * It uses a rule-based approach that represents Words as ‘One-Hot’ encoded vectors.
    * Traditional method focuses on syntactic representation instead of semantic representation.
    * Bag of words - classification model is unable to distinguish certain contexts.

**Challenges:**
* Ambiguity
    * The problem of determining which sense was meant by a specific word is formally known as word sense disambiguation.
    * Syntactic disambiguation (The 'I made her duck' example)

**Knowledge of Language:**
* Phonetics & Phonology (linguistic sounds)
* Morphology (meaningful components of words)
* Semantics (meaning)
* Pragmatics (meaning wrt goals and intentions)
* Discourse (linguistic units larger than a single utterance)

**NLP Terminology:**
* A collection of documents is a corpus
* Each document is a collection of tokens
* Blocks of n words are referred to as n-grams

**Things to consider:**
* Parts of speech tagging:
    * Stopwords
    * Sentence Segmentation
* N-grams
* Normalization
* New York, POTUS, ...
* Stemming - the process of reducing words to their word stem, base or root form
* Lemmatization - removes inflectional endings only and to return the base or dictionary form of a word
    * Lemmatization works with tenses, while stemming does not.

**Text Processing:**
* A typical processing procedure might look like:
    1) Lower all of your text (although you could do this depending on the POS)
    2) Strip out misc. spacing and punctuation
    3) Remove stop words (careful they may be domain or use-case specific)
    4) Stem/Lemmatize our text
    5) Part-Of-Speech Tagging
    6) Expand feature matrix with N-grams

**Stop Words:**
* Stop words are words which have no real meaning but make the sentence grammatically correct. Words like ’I’, ’me’, ’my’, ’you’...
Scikit-Learn’s contains 318 words for the English set of stop words.
* sklearn:
```
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
STOPLIST = ENGLISH_STOP_WORDS
processed = [lemmatize_string(doc, STOPLIST) for doc in corpus]
```

**Making text machine consumable:**
* In order to perform machine learning on text documents, we first need to turn the text content into numerical feature vectors.
    * Convert our corpus of text data into some form of numeric matrix representation.
    * The most intuitive way to do so is the 'bags of words' representation:
        1) assign a fixed integer id to each word occurring in any document of the training set (for instance by building a dictionary from words to integer indices).
        2) for each document `#i`, count the number of occurrences of each word w and store it in `X[i, j]` as the value of feature `#j` where `j` is the index of word w in the dictionary
    * Bags of words are typically high-dimensional sparse datasets
        * To correct for this, we can save memory by only storing the non-zero parts of the feature vectors via `.fit_transform()` in scikit-learn
* In a Term-Frequency matrix each column of the matrix is a word and each row is a document. Each cell therein contains the count of that word in a document.
* Term Frequency does not account for:
    1) Underused tokens
    2) Overused tokens
    3) Length of the document

**sklearn:**
```
from sklearn.feature_extraction.text import TfidfVectorizer
```
* Hyperparameters:
    * `max_df`: Can either be absolute counts or a number between 0 and 1 indicating a proportion. Specifies words which should be excluded due to appearing in more than a given number of documents.
    * `min_df`: Can either be absolute counts or a between 0 and 1 indicating a proportion. Specifies words which should be excluded due to appearing in less than a given number of documents.
    * `max_features`: Specifies the number of features to include in the resulting matrix. If not None, build a vocabulary that only considers the top max features ordered by term frequency across the corpus.
* Example:
```
from sklearn.feature_extraction.text import TfidfVectorizer

c_train = [’Here is my corpus of text it says stuff and things’,
’Here is some other document’]
c_test = [’Yet another document’,
’This time to test on’]

tfidf = TfidfVectorizer()
tfidf.fit(c_train)
test_arr = tfidf.transform(c_test).todense()

print(tfidf.get_feature_names())
```

**spaCy:**
* The spaCy package is an industrial-strength Natural Language Processing tool in Python. spaCy can be used to perform lemmatization, part-of-speech tagging, sentence extraction, entity extraction, and more; all while excelling at large-scale information extraction tasks. Leveraging the power of Cython, spaCy is the fastest syntactic parser in the world and is capable of parsing over 13,000 words per minute.
* Lemmatization in spaCy:
```
nlp = spacy.load(’en’)
doc = nlp("And what would you do if you met a jaboo?")
lemmatized_tokens = [token.lemma_ for token in doc]
print(lemmatized_tokens)
```

**word2vec:**
* Using vector representations of words that are also known as called word embeddings.
* Group of related models that are used to produce word embeddings
* Typically they are two-layer neural networks that are trained to reconstruct linguistic contexts of word
* Input is a large corpus of text and output is a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space.
* Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space

---
### Python Week 6
#### Day 3 Morning
#### K-Means Clustering
**Supervised vs. Unsupervised Learning Review:**
* Supervised:
    * Have a target/label that we model
    * Models look like function that take in features (X) and predict a label (y)
    * Have an error metric that we can use to compare models
    * Used to predict future unlabeled data
* Unsupervised
    * No target/label to predict
    * Goal is to find underlying structure, patterns, or organization in data
    * No start error metric to compare models - determining if you have the optimal solution is very challenging. Cross validation often not applicable.
![Algorithm Cheat Sheet](http://scikit-learn.org/stable/_static/ml_map.png)

**Basic K-Means Algorithm:**
1) Initialize k centroids
2) Until convergence
    * Assign each data point to the nearest centroid
    * Recompute the centroids as the mean of the data points
* Pseudocode:
    ```
    initialize centroids
    while not converged:
        assigned_data_to_centroids
        compute_new_controid_means
    ```
![K Means](http://www.learnbymarketing.com/wp-content/uploads/2015/01/method-k-means-steps-example.png)
* [K-Means Visualized](http://www.onmyphd.com/?p=k-means.clustering)
* [K-Means Example](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

**Centroid Initialization Methods:**
1) Randomly choose k points from your data and make those your initial centroids (simplest)
2) Randomly assign each data point to a number 1-k and initialize the kth centroid to the average of the points with the kth label (what happens as N becomes large?)
3) k-means++ chooses well spread initial centroids. First centroid is chosen at random, with subsequent centroids chosen with probability proportional to the squared distance to the closest existing centroid. (default initialization in sklearn).

**Stopping Criteria:**
We can update for:
    1) A specified number of iterations (sklearn default : max_iter= 1000)
    2) Until the centroids don’t change at all
    3) Until the centroids don’t move by very much (sklearn default: tol= .0001)

**Numpy Practice:**
1) Use masking to grab the rows of features that correspond to the label of 1
2)  Find the column-wise mean of that subset of features
3) Compute the euclidean distance from each data point in features to this mean. Try to do this in one line with broadcasting!
```
features = np.array([[3,4], [2,2], [5,4], [6,9], [-1,0]])
labels = np.array([1,0,1,0,1])

features[labels==1]

features[labels==1].mean(axis=0)

np.linalg.norm(features-features[labels==1].mean(axis=0), axis=1)
```

**Evaluating K-Means:**
* How do we measure clustering performance / effectiveness?
    * Quantify how similar items are within a cluster
    * Minimize Intra-Cluster Variance or Within Cluster Variance (WCV)

**Choosing k:**
* Choosing k requires *a-priori* information:
    * Business logic (e.g., identify low, medium, and high priority customers)
    * Domain Knowledge (e.g., there are k equilibrium states resultant from the phenomena)
* Or a heuristic:
    * Elbow plot
    * Silhouette score
    * GAP statistic

**Elbow Plot:**
* Elbow method - look for value of k that drastically reduces the residual sum of squares within the clusters
* Look for inflection point or value of k where improvement diminishes
![Elbow Plot](http://www.learnbymarketing.com/wp-content/uploads/2015/01/method-k-means-elbow-plot.png)

**Silhouette Score:**
* The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample
    * `(b - a)/max(a,b)`
    * only defined for 2 <= k < n
* Values range from -1 to 1, with 1 being optimal and -1 being the worst
* Silhouette score accounts for inter and intra-cluster distance, while elbow plot only accounts for intra cluster
![Silhouette Score](http://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_silhouette_analysis_003.png)

**K-Means Assumptions:**
* Picked the “correct” k
* Clusters have equal variance
* Clusters are isotropic (variance spherical)
* Clusters do NOT have to contain the same number of observations
![K Means Assumptions](http://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_assumptions_001.png)

**Practical Considerations:**
* K-means is not deterministic - falls into local minima. Remedy by reinitializing multiple times and take the version with the lowest within- cluster variance (sklearn does multiple initializations by default)
* Susceptible to curse of dimensionality
* One hot encoded categorical can overwhelm - look into k-modes
* Try MiniBatchKMeans for large datasets (finds local minima, so be careful)

---
#### Day 3 Afternoon
### Hierarchical Clustering
**Review K-Means:**
1) Randomly assign a number, from 1 to K, to each of the observations.
2) Iterate until the cluster assignments stop changing:
    * For each of the K clusters, compute the cluster centroid: the vector of the p features means for the observations in the k-th cluster
    * Assign each observation to the cluster whose centroid is closest (defined using Euclidian distance)
* K-Means in a nutshell:
    * Computing distances
    * Computing means

**Hierarchical Clustering:**
* Type of ‘agglomerative clustering’ - we iteratively group observations together based on their distance from one another
* As we continue to group observations together we form a hierarchy of their similarities with one another
* This will answer different questions than KMeans - we no longer have to choose the number of clusters up front, instead we will have to define the nature of successive groups of observations (linkages!)
* Results don’t depend on initialization
* Not limited to euclidean distance as the similarity metric
* Easy visualized through dendrograms
    * “Height of fusion” on dendrogram quantifies the separation of clusters

**Hierarchical Clustering Visual:**
![Hierarchical Clustering](http://www.statisticshowto.com/wp-content/uploads/2016/11/clustergram.png)

**Hierarchical Clustering Steps:**
1) Begin with n observations and a measure of dissimilarity (Euclidean dist, cosine similarity, etc.) of all pairs of points, treating each observation as its own cluster.
2) Fuse the two “clusters” that are most similar. The similarity of these two indicates the height on the dendrogram where the fusion should be recorded
3) Compute the new pairwise similarities between the remaining clusters,
4) rinse and repeat

**Measures of (dis)similarity between groups:**
1) Single Linkage
    * Distance between two clusters is defined as the shortest distance between two points in each cluster.
    * “Nearest neighbor”
    * Drawback: Chaining -- several clusters may merge together due to just a few close cases.
    ![Single Linkage](http://www.saedsayad.com/images/Clustering_single.png)
2) Complete Linkage
    * Distance between two clusters is defined as the longest distance between two points in each cluster.
    * “Farthest neighbor”
    * Drawback: Cluster outliers prevent otherwise close clusters from merging.
    ![Farthest Neighbor](http://www.saedsayad.com/images/Clustering_complete.png)
3) Average Linkage
    * Distance between two clusters is defined as the average distance between each point in one cluster to another.
    * “Average neighbor”
    * Drawback: Computationally expensive.
    ![Average neighbors](http://www.saedsayad.com/images/Clustering_average.png)
* Average and complete are most common

**Linkage on Dendrograms:**
* Average linkage
    * Not too sensitive to outliers
    * Compromise between complete linkage and single
* Complete Linkage
    * More sensitive to outliers
    * May violate 'closeness'
* Single Linkage
    * Less sensitive to outliers
    * Handles irregular shapes fairly naturally

---
### Python Week 6
#### Day 4 Morning
#### Dimensionality Reduction (PCA)
**Why reduce the dimensions?**
* In reality dimensions are features or predictors
* Remove multicolinearity
* Deal with the curse of dimensionality
* Remove redundant features
* Interpretation & visualization
* Make computations of algorithms easier
* Identify structure for supervised learning

**The Curse of Dimensionality:**
* Definition #1: The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces (often with hundreds or thousands of dimensions) that do not occur in low-dimensional settings such as the three-dimensional physical space of everyday experience.
* Definition #2: The curse of dimensionality is a phenomenon that any model/technique which involves a distance metric suffers from. Due to the number of features, even the closest data points seem very far away.
* Heuristic: A rough heuristic for a model to be effective is that you need the distance between points to be less than some value `d`. For a single dimension (unit-length) this usually requires on average `1/d` points. If you have p dimensions the number of data points you need scales as `1/d^p`.

**What tools do we have already?:**
* AIC/BIC in the context of GLM/GLMMs
* LASSO regression
    * throw away unused features
* Neural Networks
    * For labeled data: feature extraction via upper-layer outputs
    * For labeled or unlabeled data: autoencoders
* Feature selection
    * `from sklearn.feature selection import VarianceThreshold`
    * `from sklearn.feature selection import SelectKBest`
        * For regression: `f_regression`, `mutual_info_regression`
        * For classification: `chi2`, `f_classif`, `mutual_info_classif`
        * recursive feature elimination

**Standardization:**
* Always start by standardizing the dataset
    1) Center the data for each feature at the mean (so we have mean 0)
    2) Divide by the standard deviation (so we have std 1)
* sklearn.preprocessing
    * The function scale provides a quick and easy way to perform this operation on a single array-like dataset
    * the class `StandardScaler` provides further functionality as a `Transformer` (use `fit`)
    ```
    from sklearn import preprocessing
    X = preprocessing.scale(X)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    ```

**Scaling with a train/test split:**
```
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

create some data
X,y = make_classification(n_samples=50, n_features=5)

make a train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

scale using sklearn
scaler = StandardScaler().fit(X_train)
X_train_1 = scaler.transform(X_train)
X_test_1 = scaler.transform(X_test)

scale without sklearn
X_train_2 = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test_2 = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
```

**Why do PCA in the first place?::**
* High dimensional data causes many problems. Here are a few:
    1) The Curse of Dimensionality
    2) It’s hard to visualize anything with more than 3 dimensions.
    3) Points are “far away” in high dimensions, and it’s easy to overfit small datasets.
    4) Often (especially with image/video data) the most relevant features are not explicitly present in the high dimensional (raw) data.
    5) Remove Correlation (e.g. neighboring pixels in an image)
* [PCA Visualized](http://setosa.io/ev/principal-component-analysis/)

**Correlation vs. Covariance:**
* How is it that correlation and covariance were related again?
    * They're the same thing except correlation is scaled (is always a value between -1 and 1).
[Correlation vs. Covariance ](https://stats.stackexchange.com/questions/18082/how-would-you-explain-the-difference-between-correlation-and-covariance)

**Scree Plot:**
* Displays the eigenvalues associated with a component or factor in descending order versus the number of the component or factor. You can use scree plots in principal components analysis and factor analysis to visually assess which components or factors explain most of the variability in the data.

**t-SNE:**
* t-distribution Stochastic Neighbor Embedding
* It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results.
* We may want to use another dimension reduction technique first...
    * If the data are very dense → PCA to 50 dimensions
    * If the data are very sparse → TruncatedSVD to 50 dimensions

---
#### Day 4 Afternoon
#### Singular Value Decomposition (SVD)
**Singular Value DEcomposition (SVD):**
* So we can use a technique called SVD for more efficient computation
* It is not always easy to directly compute eigenvalues and eigenvectors
* SVD is also useful for discovering hidden topics or latent features
* Under the hood, sklearn's PCA is actually SVD.

**SVD Code:**
```
import numpy as np
from numpy.linalg import svd
M = np.array([[1, 1, 1, 0, 0],
[3, 3, 3, 0, 0],
[4, 4, 4, 0, 0],
[5, 5, 5, 0, 0],
[0, 2, 0, 4, 4],
[0, 0, 0, 5, 5],
[0, 1, 0, 2, 2]])
u, e, v = svd(M)
print M
print "="
print(np.around(u, 2))
print(np.around(e, 2))
print(np.around(v, 2))
```

**Bayes Theorem:**
* Bayes’ Theorem allows us to switch around the events X and Y in a P(X|Y) situation, provided we know certain other probabilities.

P(A, B) = P(B, A)
P(A|B) ∗ P(B) = P(B|A) ∗ P(A)
P(A|B) ∗ P(B) = P(A, B)
P(A|B) = P(B|A) ∗ P(A)/P(B)

Posterior = Prior ∗ Likelihood/Evidence normalizing constant

**Naive Bayes Classifiers:**
* Naive Bayes classifiers are considered naive because we assume that all words in the string are assumed independent from one another
* While this clearly isn’t true, they still perform remarkably well and historically were deployed as spam classifiers in the 90’s. Naive Bayes handles cases where our number of features vastly outnumber our data points (i.e. we have more words than documents). These methods are also computationally efficient in that we just have to calculate sums.
* [Naive Bayes Video](https://www.youtube.com/watch?v=evtCdmjcZ4I)

**Naive Bayes Summary:**
* Unknown words → Laplace Smoothing
* Useful for Online learning
* Load of extensions and variants out there
* Pros:
    * Good with wide data (p >> n)
    * Good if n is small or n is quite big
    * Fast to train
    * Good at online learning, streaming data
    * Simple to implement, not necessarily memory-bound (DB implementations)
    * Multi-class
* Cons:
    * Naive assumption means correlated features are not actually treated right
    * Sometimes outperformed by other models

**Seaborn correlation matrix!**
* [sklearn doc](https://seaborn.pydata.org/examples/many_pairwise_correlations.html)
* Code:
```
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

Compute the correlation matrix
corr = d.corr()

Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```

---
### Python Week 6
#### Day 5 Morning
#### Map Reduce
**Morning Speaker (Tectonic) Reading List:**
* Think Like a Freak, Freakenomics
* Thinking Data - Max Shron
* Daniel Gilber - Stumbling Upon Happiness

**Big Data:**
* Data so large that it cannot be stored on one machine.
* Can be:
    * Structured: highly organized, searchable, fits into relational tables
    * Unstructured: no predefined format, multiple formats
* Often describes as 3 Vs: (high volume, velocity, and variety)
* Two possible solutions to Big Data:
    * Make bigger computers (scale up)
    * Distribute data and computation onto multiple computers (scale out)
![Big Data](http://bigdatadimension.com/wp-content/uploads/2017/03/whatisbd.png)
* ERP: Enterprise resource planning
* CRM: Customer relationship management

**Local vs. Distributed Computing:**
* Local: Uses the resources of 1 computer
* Uses the resources of many computers

| Size of Data  | Analysis Tools  | Data Storage  | Examples  |
|---|---|---|---|
| < 10 GB  | R/Python  | Local: can fit in one machine's RAM  | Thousands of sales figues  |
|10 GB - 1 TB   | R/Python with indexed files (key, record)  | Local: fits on one machine's hard drive | Millions of web pages   |
| > 1 TB  | Hadoop, Spark, Distributed Databases  | Distributed: Stored across multiple machines| Billions of web clicks, about 1 month of tweets |

**Tangent: On-premise vs. the Cloud:**
* On-premise: Software and/or hardware that are installed on the premises of the company that uses them.
* Cloud: Software and/or hardware installed at a remote facility and provided to companies for a fee.
![Visual](https://www.adeaca.com/blog/wp-content/uploads/2016/12/Cloud-vs-On-Premise-1.png)

**Apache Hadoop:**
* Hadoop (the full proper name is ApacheTM Hadoop®) is an open-source, distributed computing framework that was created to make it easier to work with big data.
* It provides a method to access and process data that are distributed among multiple clustered computers.
* “Hadoop” typically refers to four core components, though sometimes it refers to the ecosystem (next slide). The four components:
    * Hadoop Distributed File System (HDFS) - Manages and provides access to distributed data.
    * Hadoop YARN - Provides framework to schedule and manage jobs across the cluster.
    * Hadoop MapReduce - YARN-based parallel processing system for large datasets. MapReduce provides computation on distributed data.
    * Hadoop Common - A set of utilities that support the other three core modules.

**The Hadoop Ecosystem:**
* Headings:
    * Distributed File System (HDFS is one)
    * Distributed Computing (See MapReduce, Spark)
    * SQL-on-Hadoop (See Hive)
    * NoSQL databases
    * NewSQL databases (!?)
    * And many other headings

**HDFS:**
* Server farm (where Hadoop runs)
* A node (server)
* DataNode
    * DataNode installed on the nodes (servers) whose responsibility is to store and compute
    * It manages storing and locating blocks (chunks) of 64 MB of data on the node.
    * It communicates with and responds to requests from the NameNode, including the “heartbeat.”
    * Can communicate with other DataNodes (e.g. to copy data) and the Client.
* NameNode
    * Tracks where data blocks are stored in the cluster.
    * Interacts with client applications
    * Is the potential single point of failure for the entire HDFS. This is why there is a backup NameNode and its construction and components are “enterprise class”.
    * Manages backing up of data blocks (generally stored in 3 different nodes in different racks).

**MapReduce:**
* TaskTracker
    * Installed on nodes with the DataNode software.
    * Performs the map, shuffle and sort, and reduce operations.
    * Monitors status of these operations and reports progress to JobTracker. Also sends a “heartbeat” to JobTracker to indicate that it’s functioning properly.
    * Can communicate with other TaskTrackers.
* JobTracker
    * Coordinates data processing.
    * Interacts with NameNode to determine where data is stored.
    * Will schedule a different TaskTracker if a TaskTracker doesn’t submit a “heartbeat” or has corrupt data.
    * Communicates with the Client.
    * Just like the NameNode, the JobTracker is given “enterprise” hardware.

**MapReduce Steps:**
* Send the computation to the data rather than trying to bring the data to the computation.
* Computation and communication are handled as (key, value) pairs.
* In the “map” step, the mapper maps a function on the data that transforms it into (key, value) pairs. A local combiner may be run after the map to aggregate the results in (key, local aggregated values).
* After the mapping phase is complete, the (key, value) or (key, local aggregated value) results need to be brought together, sorted by key. This is called the “shuffle and sort” step.
* Results with the same key are all sent to the same MapReduce TaskTracker for aggregation in the reducer. This is the “reduce” step.
* The final reduced results are communicated to the Client.
* It’s really Divide and Conquer:
    1) Split one large tasks into many smaller sub-tasks that can be solved in parallel on different nodes (servers)
    2) Solve these tasks independently
    3) Recombine the results of the sub-tasks for the final results
* The types of problems MapReduce is especially suited for:
    * Count (morning assignment), sum, avg, sort, graph traversal and analysis (optional afternoon assignment)
    * Some machine learning algorithms
* Classic example of MapReduce: Word count of a body of text.
![MapReduce Visualized](https://miro.medium.com/max/1846/0*it9fFvZ5h2eFL2e3.jpg)

**Takeaways:**
* Hadoop - open-source framework made to handle big data through distributed computing.
* HDFS - data management component of Hadoop
    * NameNode - keeps track of where data is, makes sure it’s backed up
    * DataNode - stores the data
* MapReduce - computation component of Hadoop
    * JobTracker - coordinates jobs, communicates with client
    * TaskTracker - performs computations on local data
        * A local mapper maps a function on data, perhaps using local combiner, then sends the results somewhere to be reduced
        * Data is handled as (key, value) pairs
        * All computations written to hard disk (for redundancy, but slow)
* Many other components in Hadoop Ecosystem
* Good practice with cloud computing in general: make sure it works on a small-scale data set locally first, then see if analysis in the cloud gives you the same results. If it does, you can extrapolate to larger datasets that couldn't be processed locally with some confidence.

---
#### Day 5 Afternoon
#### Profit Curves and Imbalanced Classes
**Problem Motivation:**
* Classification datasets can be “imbalanced”.
    * i.e. many observations of one class, few of another
    * Will give concrete examples later, but even a minority class of comprising 33% of the data can be considered imbalanced.
* Costs (in time, money, or life!) of a false positive is often different from cost of a false negative. Need to consider external (e.g. business) costs.
    * e.g. missing fraud can be more costly than screening legitimate activity
    * False negative in disease screening vs False negative in email spam classification
* Accuracy-driven models will over-predict the majority class.
    * In the case of imbalanced classes, accuracy is a pretty bad metric.

**Solutions:**
* Practical steps (help your model fit better):
    * Stratifying train_test_split
    * Change weighting of training data for poorly represented class
* Cost-sensitive learning (use outside costs & benefits to set prob. thresh):
    * thresholding (aka “profit curves”)
* Sampling (reduce imbalance with more/less data):
    * Oversampling
    * Undersampling
    * SMOTE - Synthetic Minority Oversampling Technique

**Dealing w/ Imbalanced Classes: Practical Steps:**
* Stratifying train_test_split
    ```
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)
    ```
* Change weighting of training data for poorly represented class
    * In objective function minimization, all classes are weighted equally by default.
    Option 1) `class_weight` argument in sklearn models
        * `class_weight=balanced`
    Option 2) `.fit(X, y[,sample_weight])` fit the model according to the given training data

**Dealing with Imbalanced Classes: Cost Sensitive Learning:**
* Quantify relative costs of TP, FP, TN, FN
* Construct a confusion matrix for each probability threshold, and use a cost-benefit matrix to calculate a “profit” for each threshold. Pick the threshold that give the highest profit.
* ROC Curve:
    * Shows FPR = (1-TNR) vs TPR (aka Recall)
    * Doesn't give preference to one over the other

**Sampling Techniques: Undersampling:**
* Undersampling randomly discards majority class observations to balance training sample.
* Pro: Reduces runtime on very large datasets.
* Con: Discards potentially important observations.

**Sampling Techniques: Oversampling:**
* Oversampling replicates observations from minority class to balance training sample.
* Pro: Doesn't discard information.
* Con: Likely to overfit

![Under/Oversampling](https://storage.ning.com/topology/rest/1.0/file/get/2808331754?profile=original)

**SMOTE:**
* SMOTE - Synthetic Minority Oversampling TechniquE
* Generates new observations from minority class.
* For each minority class observation and for each feature, randomly generate between it and one of its k-nearest neighbors.
* SMOTE pseudocode:
```
synthetic_observations = []
while len(synthetic_observations) + len(minority_observations) < target:
    obs = random.choice(minority_observations):
    neighbor = random.choice(kNN(obs, k)) # randomly selected neighbor new_observation = {}
        for feature in obs:
            weight = random() # random float between 0 and 1 new_feature_value = weight*obs[feature] \
                    + (1-weight)*neighbor[feature]
            new_observation[feature] = new_feature_value

synthetic_observations.append(new_observation)
```

**Sampling Techniques - Distributions:**
* What's the right amount of over-/under-sampling
* If you know the cost-benefit matrix:
    * Maximize profit curve over target proportion
* If you don't know the cost-benefit matrix:
    * No clear answer..
    * ROC's AUC might be more Useful

**Cost Sensitivity vs. Sampling:**
* Neither is strictly superior.
* Oversampling tends to work better than undersampling on small datasets.
* Some algorithms don’t have an obvious cost-sensitive adaptation,
requiring sampling.

---
### Python Week 7
#### Day 1 Morning
#### Non-Negative Matrix Factorization (NMF)
**Motivation:**
* With PCA and SVD you can decompose a matrix (in this running example, of users, movies, and their ratings of the movies) into latent topics that help relate groups of movies (or words, or books, or whatever your features are in the matrix).
* But there are problems with both, first SVD:
    * **Recall:** $M = U S V^T$

    1. The number of columns in $U$ differs from the number of rows in $V^T$. I.e. The number of latent features differs in $U$ and $V^T$, which is weird.

    2. Values in $U$ and $V^T$ can be negative, which is weird and hard to interpret. For example, suppose a latent feature is the genre 'Sci-fi'. This feature can be positive (makes sense), zero (makes sense), or negative (what does that mean?).

**NMF Overview:**
* Take a large matrix (denoted **V**) and factor into two smaller dimensional matrices **W** and **H**.
* Force solutions to have all non-negative solutions
* Creates a 'parts based representation'
* Generally speaking, NMF is a method for modeling the generation of directly observable visible variables V from hidden variables H
* Think of NMF like 'fuzzy clustering'
    * The concepts are clusters
    * Each row (document, user, etc...) can belong to more than one concept

**Parts Based Representation:**
*  The non-negativity constraints lead to parts-based representation due to allowing for additive, not subtractive, combinations.
* Non-negativity constraint is intuitively compatible with combining parts to make a whole
* For example, we could apply NMF on a dataset of human faces to identify latent representations of a face. Can use those representations to additively build up a face and create in piecemeal fashion.
* This differs from PCA encoding with negative components which detract from components already in place

**Applications: Linguistic:**
* Motivations: words can mean lots of different things dependent on pronunciation and context (homonyms)
* Examples:
    * 'ruler'
    * 'unionized'
* NMF can tease out the meanings of a word based on other words that it frequently appears with
    * [“ruler”, “monarch”, “king”, “president”] vs. [“ruler”,  “protractor”, “calculator”, “slide rule”, “abacus”]
    * [“unionized”, “workers”, “labor”, “rights”] vs. [“unionized”, “ion”, “atom”, “reaction”]

**Other Use Cases:**
* Topic modeling:
    * Text from news articles
    * Text from books
    * Text from webpages
* Content based recommendation engines

**Sparsity:**
* NMF representations are naturally sparse, in that many of the components are exactly equal to zero. In contrast, PCA often incorporates some aspect of every feature in a component

**NMF Algorithm - Optimization:**
* Lee & Seung's multiplicative update rules
    * What you should do today
    * Provide a good compromise between speed and ease of implementation
* Additive update rules (e.g., gradient descent)
    * Convergence can be slow and is sensitive to step size

**Choosing k:**
1) Compare reconstruction error (will always decrease, look for elbow)
2) Compare cosine similarity within topics and between topics (as always...point of diminishing returns with increasing k)
3) Domain knowledge
4) Manual inspection for small values/ranges of k

**Sklearn Implementation:**
* Code:
```
import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_
```

---
### Python Week 7
#### Day 3 Morning
#### Speedy Computing
**Timing units:**
* 1 *s* = 1000 *ms*
* 1 *ms* = 1000 *μs*
* 1 *μs* = 1000 *ns*

**Terminology:**
* Multicore computing - using multithreading and multiple cores
* Symmetric multiprocessing - two or more identical processors connected to a single unit of memory
* Distributed computing - processing elements are connected by a network
* Cluster computing - group of loosely coupled computers that work together (Beowulf)
* Massive parallel processing - many networked processors usually > 100.
* Grid computing - distributed computing but makes use of a middle layer to create a super virtual computer.

#### Amazon Web Services (AWS)
**Introduction:**
* AWS is a cloud service platform
* Cloud services are the delivery of computing services - servers, storage, databases, networking, software, analytics, and more - over the internet.
* Advantages: accessibility, dynamic scaling, no need for hardware or support staff, small upfront cost
* Disadvantages: security (someone else owns the server), recurring cost for services

**History:**
* In 2000 Amazon was struggling with scaling problems
* Started building internal systems to address their problems
* Realized that their solutions were useful to others, too
* Amazon Web Services launched in 2002, Elastic Compute Cloud in 2006
* Was the first widely accessible cloud computing infrastructure service
* Public cloud revenue market share: 47% AWS, 10% Azure, 4% Google Cloud

**AWS Services We Will Use:**
* Simple Storage Service (S3)
    * Long-term big data storage for the internet.
    * Where we will store data and final model results
* Elastic Compute Cloud (EC2)
    * Provides secure, resizable compute capacity in the cloud
    * Where we will train models and deploy applications (Flask web apps, databases)
    * If you are training a neural net, you’ll want to use GPU compute instances
    * Use a Spark cluster for distributed computing

**AWS Storage + Execution:**
* Primary services Amazon AWS offers:

| Name | Full Name              | Service                           |
|------|------------------------|-----------------------------------|
| S3   | Simple Storage Service | Storage                           |
| EC2  | Elastic Compute Cloud  | Execution                         |
| EBS  | Elastic Block Store    | Storage attached to EC2 instances |

* If want to store some video files on the web. Which Amazon service should I use?
    * AnswerL S3
* If just created an iPhone app which needs to store user profiles on the web somewhere. Which Amazon service should I use?
    * Answer: S3
* If want to create a web application that uses Javascript in the backend along with a MongoDB database. Which Amazon service should I use?
    * S3 + EC2 + EBS

**Differences between S3 and EBS:**

| Feature  | S3  | EBS  |
|---|---|---|
| Can be accessed from  | Anywhere on the web any EC2 instance  | Specific availability zone EC2 instance attached to it |
| Pricing  | Less expensive Storage (3¢/GB); Use (1¢/10,000 requests)  | More expensive; Storage (3¢/GB); [+ IOPS]  |
| Latency  | Higher  |  Lower |
|  Throughput |  Usually more | Usually less  |
| Performance  | Slightly worse  | Slightly better  |
| Max volume size  |  Unlimited | 16 TB  |
|  Max file size | 5 TB  | 16 TB  |

* What is latency?
    * Answer: Latency is the time it takes between making a request and the start of a response.
* Which is better? Higher latency or lower?
    * Answer: Lower is better.
* Why is S3 latency higher than EBS?
    * One reason is that EBS is in the same availability zone.

**DSI Workflow:**
1) Upload/access data on S3
2) Use pandas or boto to pull a subset of the data down to your local machine
    * Note in pandas the option for an S3 bucket URL, and an option for chunksize
3) Develop a script to train your model locally on a subset of the data
4) Upload your script to EC2
5) Run script on EC2 on the full data set
    * Ideally data is not too big for the Elastic Block Storage (EBS) associated with the EC2 instance, but if it is you can still just read in data from S3
6) Write results to S3
7) Terminate your EC2 instance and EBS

**Buckets and Files:**
* What is a bucket?
    * A bucket is a container for files.
    * Think of a bucket as a logical grouping of files like a sub-domain
    * A bucket can contain an arbitrary number of files
* A file in a bucket can be 5 TB
* bucket Names:
    * Bucket names must be unique across all of s3.
    * Bucket names must be at least 3 and no more than 63 characters long.
    * Bucket names must be a series of one or more labels, separated by a single period.
    * Bucket names can contain lowercase letters, numbers, and hyphens.
    * Each label must start and end with a lowercase letter or a number.
    * Bucket names must not be formatted as an IP address (e.g., 192.168.5.4).

**Python - AWS Integration with bobo3:**
* Boto is the Amazon Web Services (AWS) SDK for Python, which allows Python developers to write software that makes use of Amazon services like S3 and EC2. Boto provides an easy to use, object-oriented API as well as low-level direct service access.
* Step 1: Create a connection to S3
```
import boto3
boto3_connection = boto3.resource('s3')

def print_s3_contents_boto3(connection):
    for bucket in connection.buckets.all():
        for key in bucket.objects.all():
            print(key.key)

print_s3_contents_boto3(boto3_connection)
```
* Step 2: Create a Bucket:
```
import os
username = os.environ['USER']
bucket_name = username + "-new-bucket"
boto3_connection.create_bucket(Bucket=bucket_name)
```
* Step 3: Make a file:
```
!echo 'Hello world from boto3!' > hello-boto.txt
```
* Step 4: Upload the file to S3:
```
s3_client = boto3.client('s3')
s3_client.upload_file('hello-boto.txt', bucket_name, 'hello-remote.txt')
print_s3_contents_boto3(boto3_connection)
```
* Step 5: Download a file from S3:
```
s3_client.download_file(bucket_name, 'hello-remote.txt', 'hello-back-again.txt')
print(open('hello-back-again.txt').read())
```

**Miscellaneous:**
* Access AWS services as an IAM user, not as root user
* Export your AWS access keys as environment variables
* Never, ever put your AWS keys in a file that will be shared publically (e.g. Github)
* Connect to an EC2 instance using ssh, and transfer files to/from EC2 using scp.
* Recommend using the DSI-Template3 AMI (Amazon Machine Image) in N. Virginia for general data science, DSI-DeepLearning4 if you get a gpu (type p2 or p3) instance for neural net training. However, Amazon makes and maintains public AMIs that you could use, or you could make your own.
* Some people prefer the AWS Command Line Interface
* Your EC2 instance will stop whatever it’s doing if the ssh connection between your local machine and the server is interrupted, unless you use a terminal multiplexer like screen or tmux to detach from your server session before the ssh connection is interrupted. You can later re-attach to your session on the server to get results. Minimal screen commands to run on the server:
    1) `$ screen -S my-session` # creates a session called my-session
    2) Start your web app, or model training, or whatever process you want to continue.
    3) Typing `Ctrl - a`, then `d` to detach from the session
    4) You can now safely interrupt the ssh session; your EC2 instance will keep working
    5) ssh back into your instance
    6) `$ screen -ls` # lists available screen sessions
    7) `$ screen -R my-session` # re-attaches to my-session

---
### Python Week 7
#### Day 4 Morning
#### NoSQL and MongoDB
**RDBMS Data Mode:**
* An RDBMS is composed of a number of user-defined tables, each with columns (fields) and rows (records)
    * each column is of a certain data type (integer, string, date)
    * each row is an entry in the table (an observation) that holds values for each one of the columns
    * tables are specified by a schema that defines the structure of the data
    * we specify the table structure ahead of time

**RDBMS Tradeoffs:**
* Advantages:
    * Pre-defined schema permits efficient, complex queries
    * Scales vertically so making queries and joins can be done quickly
* Disadvantages:
    * Often requires that all data live on the same server
    * Doesn’t shard (horizontal partitioning across nodes) easily

**NoSQL:**
* NoSQL refers to non-SQL / non-relational / “not only” SQL databases
* Many NoSQL databases (but not all) are document-oriented
    * As opposed to row/record-oriented
* Each object/document can be completely different from the others
* No schema - each document can have or not have whatever fields are appropriate for that particular document
* MongoDB falls under the NoSQL umbrella
    * Just like PostgreSQL, MySQL, Spark SQL, etc. fall under the SQL umbrella

**SQL vs. NoSQL:**
* SQL:
    * e.g., Postgres
    * has databases...
    * which contain tables...
    * tables have rows and columns
* NoSQL:
    * e.g., MongoDB
    * has databases...
    * which contain collections...
    * collections have documents composed of fields

**MongoDB: Features:**
* Document-based storage system for semi-structured data, can have data redundancies
* Documents represented as JSON-like objects
* A change to database generally results in needing to change many
documents
* Since there is redundancy, simple queries are often faster, but more complex queries are slower
* No schema or joins
* Doesn’t need to know structure of data in advance
* Auto-sharding, easily replicated across servers
* Cursor: when you ask MongoDB for data, it returns a pointer to the result set called a cursor
* Actual execution is delayed until necessary

**MongoDB: Document Fields:**
* Field Names:
    * Must be strings
    * Can’t start with dollar sign ($) character
    * Can’t contain a dot (.) character
    * Can’t contain the null character
    * \_id is reserved for use as primary key
* Field Values:
    * Can be mixture of other documents, arrays, and arrays of documents

**MongoDB: Queries:**

| SQL  | Mongo  |
|---|---|
|  `SELECT * FROM users;` | `db.users.find()`  |
| `SELECT * FROM users WHERE age = 33 ORDER BY name ASC;`  | `db.users.find({age:33}).sort({name: 1})`   |
| `SELECT COUNT(*) FROM users WHERE age > 30;`  | `db.users.find({age: {$gt: 30}}).count()`  |

**MongoDB: Aggregations:**

| SQL  | Mongo  |
|---|---|
| `SELECT age, SUM(1) AS counter FROM users WHERE country=”US” GROUP BY age;`  | `db.users.aggregate([{$match: {country: ’US’}},{$group: {’_id’: ’$age’, counter: {$sum:1}} }])`  |

**Exercise:**
* write out the SQL and Mongo queries for the following:
    * From a collection called “log”, return the total number of records where the “country” is “India”, grouped by “city”, and name it “counter”.
        SQL:
        ```
        SELECT city, COUNT(country) as counter
        FROM log
        WHERE country = 'India'
        GROUP BY city;
        ```
        Mongo:
        ```
        db.log.aggregate([{$match: {country: 'India'}}, {$group: {'_id': '$city', counter: {$sum: 1}}}])
        ```

**Mongo Clients:**
* There are two main ways to connect to a Mongo database:
    1) Via the Mongo shell (JavaScript client)
    2) Via PyMongo (Python client)

**MongoDB Installation:**
* First, install MongoDB using your operating system's package manager:
    * Mac: `$ brew install mongodb`

**Mongo Server:**
* Before we connect to mongo, we have to start a mongo server. We do that by typing the following in the terminal:
    * `$ mongod`
* Note: The mongo daemon needs to continue to run during your database session. So, you should run the above command in a separate terminal window/tab or use something like tmux.

**Connecting to MongoDB using Mongo Shell:**
* After starting the mongo server, we can connect from the Mongo shell in one of two ways:
```
$ mongo
```
* The above connects to a default database (usually your username)
```
$ mongo <database_name>
```
* The above connects to named database

**Basic Mongo Shell Commands:**

| Command  | Purpose  |
|---|---|
| `help`  | List top level mongo commands  |
| `db.help()`  | List database level mongo commands  |
| `db.collection.help()`  | List collection level mongo commands |
| `show dbs`  | Obtain a list of databases   |
|  `use database` | Change the current database to *database*  |
| `show collections`  | Obtain list of collections in current database  |

**PyMongo Installation:**
* We can interact with Mongo databases from Python using PyMongo
* To install:
```
$ conda install pymongo
```
* Don't foret to start the mongo daemon:
```
$ mongod
```

**PyMongo Client:**
```
from pymongo import MongoClient
client = MongoClient()
db = client['your db']
coll = db['your_collection']
```

**PyMongo Queries:**
* Queries look similar to those inside of Mongo shell:
```
res = coll.fin({'name': {$ne: 'Jeremy Renner'}})
```
* This returns a query cursor object as a generator that we can use in Python.

---

#### Day 4 Afternoon
### Web Scraping
**Motivation: Why do we want to do web scraping?:**
* Any time that you want data from the web and it doesn’t have a clickable link, you will have to pull down that data via the command line (e.g., using curl) or via a program (e.g, using Python)
* The web is an enormous database of text/image/video training data

**WWW vs. the Internet:**
* The world wide web is a global collection of interconnected hypermedia documents hosted on web servers
* The Internet is the global network that connects them (using TCP/IP)
* Think of web servers as islands existing all over the globe, while the Internet provides bridges connecting those islands

**URLs:**
* URL - Uniform Resource Locator. Used to specify the location of a document within the world-wide web.
* Each url has a few different pars:
    * Protocol - specifies the means for communicating with the web server, typically http or https. Often gets automatically filled in in your web browser, but this is usually not the case when web scraping. See http://www.realifewebdesigns.com/web-resources/web-protocols.html for examples.
    * Host - points to the name of the web server you want to communicate with. A host name is associated with a specific IP address.
    * Port - holds additional information used to connect with the host. (Think: host is the city name, port is street address)
    * Path - Indicates where on the server the file you are requesting lives

**Client-Server Relationships:**
* At any point in time, any person or computer connected to the internet can be either a server or a client
* The client is the requesting party (requesting some info, like a webpage)
* The server is the party providing that information and responding to requests from a client
* If we visit www.example.com in our browser, then we are the client and www.example.com is the server.
* The interaction starts with the client issuing a GET request to the server, indicating that it would like some specific piece of information
* Once the server gets this request, it will send back a response with the information requested in the body (and a header with a status code in it)

**Http Status Codes:**
* In general...
    * 2xx successful (usually 200)
    * 3xx redirect - ultimately successful
    * 4xx client side error aka the user’s fault (common: 404 - you are looking for a file that doesn’t live where you think it does, or wrong permissions)
    * 5xx server side error aka the service’s fault

**HTML: Hypertext Markup Language:**
* The majority of web pages are formatted using Hypertext Markup Language (HTML)
    * Transmitted via Hypertext Transport Protocol (HTTP) over TCP/IP
* HTML uses tags, where each tag describes different document elements

**Basic HTML Tags:**
![Basic HTML Tags](http://www.computing.dcu.ie/~humphrys/Notes/Networks/tanenbaum/7-27.jpg)

**HTML and CSS:**
* Permits clean separation of content from presentation style
* Radical site redesigns are possible by modifying just the style and not rewriting any of the content

**CSS Classes, IDs, and Selectors:**
* Web developers may define CSS classes (starting with .) to identify HTML elements for styling
    * The same CSS class can be assigned to multiple page elements
    * Applying a new style to that class in the CSS changes the appearance of all elements using that class
* CSS IDs (starting with #) identify a single, unique element within a page.
* CSS selectors are patterns used to select HTML element(s) (including those matching CSS classes and IDs) for formatting.
* [CSS Reference](https://www.w3schools.com/cssref/trysel.asp)

**HTML and Web Scraping:**
* All those HTML tags, style classes and ids make great hooks for finding exactly the content you want in a web page.
* In practice, when web scraping, we will issue a get request for an entire page’s html, and then pull specific elements either by their HTML tags or CSS selectors.

**Viewing Page Source:**
* To see all of the HTML on a web page, right click and use ‘View Page Source’
* Better: right click and choose ‘Inspect’ (or ‘Inspect Element’)  (Chrome, Firefox and Safari)
    * Shows you the html that corresponds to the individual element you chose to look at.

**Scraping Individual Elements:**
* Examine HTML corresponding to individual element we want to scrape, and then use either its HTML tag or CSS selectors to parse it out in Python
* Workflow:
    1) Find the page you want to scrape information from
    2) Find the element(s) that you want to grab
    3) Use inspect element to figure out what HTML tag or CSS selectors to use to grab the elements
    4) Use Python to fetch those elements
* This whole process can be made easy using BeautifulSoup in Python

**Using Python and BeautifulSoup:**
* Use requests library to issue a ‘GET’ request on the url, then use BeautifulSoup to parse the html
```
import requests
from bs4 import BeautifulSoup

req = requests.get('www.example.com')
html = BeautifulSoup(req.content, 'html.parser')
```
* Next, use methods on the html object returned from BeautifulSoup to get content from the page:
```
html.find('a') # Returns the first 'a' tag
html.find_all('a') # Returns all 'a' tags
html.find('a').text # Returns teh test of the first 'a' tag
```

**Beautiful Soup: Selecting Tags:**
* `.select()` - This method allows us to select tags using CSS selectors
* `.find_all()` - This method allows us to select all tags matching certain parameters and returns them as a list. For example:
    * `soup.find_all('div')`: returns a list of all div tags in the soup object
    * `soup.find_all('div', attrs={'class': 'content'})`: returns only the div tags that also have class = content
* `.find()` - This method is the exact same as calling `soup.find_all(limit=1)`. Rather than returning a list, it only returns the first match that it finds.

**Beautiful Soup: Navigating the HTML Tree:**
* Tags may contain strings and other tags. These elements are the tag’s children. Beautiful Soup provides a lot of different attributes for navigating and iterating over a tag’s children. For example:
```
head_tag = soup.head
title_tag = head_tag.contents[0]
for child in title_tag.children:
    print(child)
```
* You can do `.find()` and `.select()` on tag's contents too.

**Pandas and Web Scraping:**
* Pandas can read and parse HTML. So, if you only need data from tables from a single page this is probably the easiest approach.
```
tables = pd.read_html(“http://www.website.com/page.html”)
```
* This returns a list of DataFrames where each DF is made from a table in the source page. But, depending on formatting quirks, this might be significantly worse than using Beautiful Soup.

**Application Programming Interfaces (APIs):**
* Webmasters don’t want you pinging their websites every 10ms to scrape some information.
* Often times they build APIs to help you collect just the info you need (and so that you don’t crush their servers).
* It’s win-win because you use less bandwidth on their server, and you get the data you need in a consistently formatted fashion.
* [API Python Tutorial](https://www.dataquest.io/blog/python-api-tutorial/)

**API Keys:**
* You will often need to register to gain access to a site’s API.
* Sometimes to to track usage, often to limit the number of calls, and occasionally as a business model where you pay per call or by amount of transferred data.
* These are essentially your login credentials, so never publish your keys! Don’t push up any script to GitHub that has your keys in plain text.

**API Key Security:**
* One option: Store your key as an environment variable on your local machine.
* In your bash profile (`~/.bash-profile`, `~/.bashrc`, or `~/.profile`) include the following:
    ```
    export API_KEY=”my api key”
    ```
* Now we can use the variable API_KEY in our Python scripts using the os package without fear of publishing our private key when we push our repos.
```
import os

my_key = os.environ[‘API_KEY’]
```

---
### Python Week 7
#### Day 5 Morning
#### Spark
**Motivation:**
* Sometimes a single machine simple cannot perform a given task fast enough
* Sometimes there are too many tasks
* And yet other times there is so much data that the data needs to be distributed
* Spark is a tool for managing and coordinating the execution of tasks on data across a cluster of computers

**High-Performance Computing Terminology:**
* Multicore computing - Using multithreading and multiple cores
* Symmetric multiprocessing - Identical processors connected to a single unit of memory
* Distributed computing - Processing elements are connected by a network
* Cluster computing - Group of loosely coupled computers that work together (Beowulf)
* Massive parallel Processing - many networked processors usually > 100.
* Grid computing - Distributed computing but makes use of a middle layer to create a super virtual computer

**Spark vs. Hadoop:**
* Spark is a cluster-computing framework. When you compare it to hadoop it essentially competes with which the MapReduce component of the Hadoop ecosystem.
* Spark does not have its own distributed filesystem, but can use HDFS.
* Spark uses memory and can use disk for processing, whereas MapReduce is strictly disk-based.
* Hadoop is considered more stable
* Hadoop has two main limitations:
    1) Repeated disk I/O can be slow, and makes interactive development very challenging
    2) Restriction to only map and reduce constructs results in increased code complexity, since every problem must be tailored to the map-reduce format

**The Spark Ecosystem:**
* Spark in addition to a cluster mode has a local mode, where jobs are run as threads instead of processes
* Spark applications have two components
    1) driver process - bookeeeping, responding, scheduling, and distributing work
    2) executor processes - execute code, and reporting the state of computation
    * Spark applications are managed by a SparkSession
    * The Java virtual machine (JVM) controls the following flow:
        * user-code -> SparkSession -> executors
    * Now while our executors, for the most part, will always be running Spark code. Our driver can be driven from a number of different languages through Spark's Language APIs.
        * Scala
        * Python
        * SQL
        * Java
        * R
    * When we start Spark in this interactive mode, we implicitly create a SparkSession which manages the Spark Application.

**Spark RDDs Core Abstractions:**
* We control our Spark Application through a driver process.
* The SparkSession instance is the way Spark executes user-defined manipulations across the cluster.
* In Scala and Python the variable is available as spark when you start up the console.
```
import pyspark as ps
import random

spark = ps.sql.SparkSession.builder \
        .appName("rdd test") \
        .getOrCreate()

my_range = spark.range(1000).toDF('number')
```
* We created a DataFrame with one column containing 1000 rows with values from 0 to 999. This range of number represents a distributed collection.
* When run on a cluster, each part of this range of numbers exists on a different executor. This range is what Spark defines as a DataFrame.
* The reason for putting the data on more than one computer should be intuitive: either the data is too large to fit on one machine or it would simply take too long to perform that computation on one machine.
* Spark has several core abstractions:
    * Datasets
    * DataFrames
    * SQL Tables
    * Resilient Distributed Datasets (RDDs)
* These abstractions all represent distributed collections of data however they have different interfaces for working with that data. The easiest and most efficient are DataFrames, which are available in all languages.

**Partitions:**
* In order to allow every executor to perform work in parallel, Spark breaks up the data into chunks, called partitions.
* A partition is a collection of rows that sit on one physical machine in our cluster.
* A DataFrame’s partitions represent how the data is physically distributed across your cluster of machines during execution.
* If you have one partition, Spark will only have a parallelism of one even if you have thousands of executors. If you have many partitions, but only one executor Spark will still only have a parallelism of one because there is only one computation resource.
* An important thing to note, is that with DataFrames, we do not (for the most part) manipulate partitions individually. We simply specify high level transformations of data in the physical partitions and Spark determines how this work will actually execute on the cluster.

**Transformations:**
* In Spark, the core data structures are immutable meaning they cannot be changed once created.
* In order to change a DataFrame you will have to instruct Spark how you would like to modify the DataFrame you have into the one that you want.
    * These instructions are called transformations.
    * Transformations are the main way of how you will be expressing your business logic using Spark.
    * Example:
        ```
        divis_by_2 = my_range.where("number % 2 = 0")
        ```
    * We only specified an abstract transformation and Spark will not do anything until we call an action.
    * A transform maps an RDD to another RDD - it is a lazy operation that only changes the direct acyclic graph representation. To actually perform any work, we need to apply an action.
        * Examples of action methods are `collect()` and `saveAsObjectFile()`

**RDDs:**
* The RDD (Resilient Distributed Dataset) API has been in Spark since the 1.0 release.
* The RDD API provides many transformation methods, such as map(), filter(), and reduce() for performing computations on the data.
* Each of these methods results in a new RDD representing the transformed data.
* However, these methods are just defining the operations to be performed and the transformations are not performed until an action method is called.
* At the core, an RDD is an immutable distributed collection of elements of your data, partitioned across nodes in your cluster that can be operated in parallel with a low-level API that offers transformations and actions.
* Consider these scenarios or common use cases for using RDDs when:
    * you want low-level transformation and actions and control on your dataset
    * your data is unstructured, such as media streams or streams of text
    * you want to manipulate your data with functional programming constructs than domain specific expressions
    * you don’t care about imposing a schema, such as columnar format, while processing or accessing data attributes by name or column; and
    * you can forgo some optimization and performance benefits available with DataFrames and Datasets for structured and semi-structured data

---
### Python Break Week
#### Optional
#### Time Series
**Time Series Fundamental Concepts:**
* A time series is a specific type of data, where measurements of a single quantity are taken over time.
* Examples: Google Searches
* A trend in a time series is a gradual change in average level as time moves on.
    * A trend an be increasing, decreasing, or neither (if, for example, a trend changes direction at some point in time).
* You can often use a regression model to capture a general trend in the time series.
    * If we subtract out the fit trend from the original time series, we get the detrended time series
        * Detrending a time series is often times a first step in analysing time series.
    * When linear detrending is inappropriate you can detrend by computing a moving average.
        * We essentially slide a window of a fixed size across our data, and average the values of the time series within the window.

**Moving Averages:**
* Moving averages: slide a window (*w*) of a fixed size across our data, and average the values of the time series within the window.
* Equation:
$$\hat{y}_i = \frac{1}{2w + 1} \sum_{j = -w}^{w} y_{i + j}$$

* Smaller values of $w$ tend to be influenced by noise of other non-trend patterns in the time series
* Large values of $w$ produce smoother estimates of the general trend in the data

**Seasonality:**
* A seasonal pattern in a time series is one that tends to appear regularly and that aligns with features of the calendar.
* When we have data that aligns with calendar regularities (quarterly, weekly, yearly), it is a good idea to chose the window so that an entire annual cycle is used in the smoothing. This will average out any seasonal patterns in the data
* Just like we can detrend a time series, we can also deseasonalize a time series.
* The simplest method is to create dummy variables at regular intervals of the calendar:
    * A dummy for every month
    * Or, a dummy for every season
    * Or, be even more granular than that
* and then fit a linear regression to the time series using these dummy variables.

**Trend-Seasonal-Residual Decomposition:**
* The Classical Trend-Seasonal-Residual Decomposition expresses a time series as the sum of three components:
$$y_t = T_t + S_t + R_t$$

* and is accomplished as follows:
    * Suppose, for definiteness, that we are working with weekly data, so that each $52$ observations forms a calendar year. Then, the time series is decomposed as follows:
        1) Compute the trend component $T_t$ using a moving average with window width $52$ (or $12$ for monthly data. Then detrend the time series
        2) Compute the seasonal component $S_t$ of the detrended time series $y_t - T_t$ by averaging together the observations that fall in the same week (or month, if monthly data). Note, this is equivalent to fitting a linear regression to the detrended data with an indicator for each week, and then making predictions for each week of the original time series
        3) The remainder, or error, or residual time series $E_t$ is $y_t - T_t - S_t$

**In Python:**
* Statsmodel.tas.seasonal_decompose
* pd.rolling_mean (moving average)

**Statistical Concepts:**
* Random Processes:

---
### Python Week 8
#### Day 1 Morning
#### Spark SQL
**Overview:**
* What is Spark SQL?
    * Spark SQL takes basic RDDs and puts a schema on them.
* What is a DataFrame?
    * DataFrames are the primary abstraction in SparkSQL
    * DataFrames are the norm in Spark (not RDD's)
* What are schemas?
    * Schemas are metadata about your data
    * Schema = Table Names + Column Names + Column Types
* What are the pros of schemas?
    * Schemas enable using column names instead of column positions
    * Schemas enable queries using SQL and DataFrame syntax
    * Schemas make your data more structured

**Terminology:**
* Apache Spark - Distributed computing system (i.e., implementing computations on a distributed file system) commonly used when working with big data & for machine learning; an alternative to MapReduce
* Cluster - A group of loosely coupled computers that work together (i.e. distributed computing)
* Resilient Distributed Dataset (RDD) - An immutable distributed collection of your data
* Driver processes - bookkeeping, responding, scheduling, and distributing work; run on the master node
* Executor processes - executing code and reporting the state of computation; run on the worker/slave nodes
* We need a cluster manager to run the machines in our cluster.
* Spark applications are managed by a SparkSession
* User code is used to interface with the SparkSession, which then delegates tasks to the executors

**Schemas:**
* Metadata about your data
* Schema = Table Names + Column Names + Column Types
* Schemas enable using column names instead of column positions
* Schemas enable queries using SQL and DataFrame syntax
* Schemas make your data more structured

**DataFrames:**
* DataFrames are just RDDs with a schema.
* These Spark DataFrames are very similar to pandas dataframes (and we can convert them to an actual pandas dataframe too).
* A DataFrame contains an RDD of row objects, each representing a record
* Technically a DataFrame is separate from an RDD, but we can do many of the same things with it
* Having a schema allows them to store & process data more efficiently
* Note that before Spark 1.3, DataFrames were called SchemaRDD (and
were a bit different)

**Advantages of DataFrames:**
* They provide an abstraction that simplifies working with structured datasets
* They can read and write data in a variety of structured formats
* They let you query the data using SQL
* They are much faster than traditional RDDs

**RDD Syntax vs. DataFrame Syntax:**
* Spark default RDDs use key-value pairs
* What if our data is not key-value pairs similar to the following:
```
{'name': 'Amy', age: 18, hobby: 'drinking'} {'name': 'Greg', age: 60, hobby: 'fishing'} {'name': 'Susan', age: 30}
```
* To return a query that returns people who are older than 18 with hobbies in RDD:
```
rdd.filter(lambda d: d['age'] > 18) \ .filter(lambda d: 'hobby' in d.keys()) \ .map(lambda d: d['name'])
```
* To do the same with in DataFrames:
```
spark.sql(
'SELECT name
FROM table
WHERE age > 18 AND hobby IS NOT NULL')
```

**DataFrame Methods for EDA:**
* `.show(n)` or `.head(n)` to get the first n rows
* `.describe()` computes statistics for numeric and string columns
* `.printSchema()` gives you the schema of the table (columns and datatypes, like `df.info()` in pandas)
* `.sample()` and `.sampleBy()` give you subsets of the data for easier development

**Usual Spark Workflow:**
1) Create the environment to run Spark SQL from python
2) Create DataFrames from RDDs or from files
3) Run some transformations
4) Execute actions to obtain values (local objects in python)

OR

1) Source activate python 3 (if you installed spark in python 3)
2) Start the spark session
3) Load the data (locally, from an S3 bucket, from HDFS, etc.)
4) If there is not already a schema...
    * Define a schema
    * Define the dataframe object (using the data + the schema)
5) Register the table for SQL
6) Run queries on the registered tables
7) View the query results using show() or collect()

**Initializing a SparkSession in Python:**
```
import pyspark as ps

Use the command spark for your sqlcontext commands
Use the command sc for your master sparkcontext commands
spark = ps.sql.SparkSession.builder \
            .master("local[4]") \
            .appName("df lecture") \
            .getOrCreate()
```

**Creating a DataFrame from an RDD:**
* To create a DataFrame from an existing RDD, you need to add a schema.
* To build a schema, you will use existing data types provided in the `pyspark.sql.types`:

| Types | Python-like type |
| - | - |
| StringType | string |
| IntegerType | int |
| FloatType | float |
| ArrayType\* | array or list |
| MapType | dict |

* To specify a schema:
```
from pyspark.sql.types import *

schema = StructType( [
    StructField('id',IntegerType(),True),
    StructField('date',StringType(),True),
    StructField('store',IntegerType(),True),
    StructField('state',StringType(),True),
    StructField('product',IntegerType(),True),
    StructField('amount',FloatType(),True) ] )

df = spark.createDataFrame(rdd_sales,schema)

df.show()
```

**Creating a DataFrame from a CSV File (Inferring Schema):**
* Use `sqlContext.read.csv` to load a CSV into a DataFrame. It can infer the schema and allows for useful parameter specification.
```
df = spark.read.csv('data/aapl.csv',
                         header=True,       # use headers or not
                         quote='"',         # char for quotes
                         sep=",",           # char for separation
                         inferSchema=True)  # do we infer schema or not ?

df.show()
```

**Creating a DataFrame from a JSON File (Inferring Schema):**
* Use `sqlContext.read.json` to load a JSON file into a DataFrame. It can infer the schema and allows for useful parameter specification.
```
df = spark.read.json('data/sales.json')

df.show()
```

**Viewing DataFrame Schema:**
* To view the schema of an existing DataFrame:
```
df.printSchema()
```

**Actions: Turning DataFrame into a Local Object:**
* Some DataFrame actions use the same syntax as RDDs, others are new or different.


| Method | DF vs RDD? | Description |
| - | - | - |
| [`.collect()`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.collect) | identical | Return a list that contains all of the elements as Rows. |
| [`.count()`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.count) | identical | Return the number of elements. |
| [`.take(n)`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.take) | identical | Take the first `n` elements. |
| [`.top(n)`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.top) | identical | Get the top `n` elements. |
| [`.first()`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.first) | identical | Return the first element. |
| [`.show(n)`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.show) | <span style="color:green">new</span> | Show the DataFrame in table format (`n=20` by default) |
| [`.toPandas()`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.toPandas) | <span style="color:green">new</span> | Convert the DF into a Pandas DF. |
| [`.printSchema(*cols)`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.printSchema)\* | <span style="color:green">new</span> | Display the schema. This is not an action, it doesn't launch the DAG, but it fits better in this category. |
| [`.describe(*cols)`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.describe) | <span style="color:green">new</span> | Compute statistics for this column. |
| [`.sum(*cols)`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.sum) | <span style="color:red">different</span> | Applies on GroupedData only (see transformations). |
| [`.mean(*cols)`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.mean) | <span style="color:red">different</span> | Applies on GroupedData only (see transformations). |
| [`.min(*cols)`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.min) | <span style="color:red">different</span> | Applies on GroupedData only (see transformations). |
| [`.max(*cols)`](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.max) | <span style="color:red">different</span> | Applies on GroupedData only (see transformations). |

**Transformations on DataFrames:**
* Both actions and transformations are lazy, Spark doesn't apply the transformations right away.
* Transformations transform a DataFrame into another because DataFrames are also immutable.
    * DataFrames are just RDDs with a schema

| Method | Type | Category | Description |
| - | - | - | -
| [`.withColumn(label,func)`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.withColumn) | transformation | mapping | Returns a new DataFrame by adding a column or replacing the existing column that has the same name. |
| [`.filter(condition)`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.filter) | transformation | reduction |  Filters rows using the given condition. |
| [`.sample()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sample) | transformation | reduction | Return a sampled subset of this DataFrame. |
| [`.sampleBy(col)`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sampleBy) | transformation | reduction | Returns a stratified sample without replacement based on the fraction given on each stratum. |
| [`.select(cols)`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.select) | transformation | reduction | Projects a set of expressions and returns a new DataFrame. |
| [`.join(dfB)`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.join) | transformation | operations | Joins with another DataFrame, using the given join expression. |
| [`.groupBy(col)`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.groupBy) | transformation | operations | Groups the DataFrame using the specified columns, so we can run aggregation on them.  |
| [`.sort(cols)`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sort) | transformation | sorting |  Returns a new DataFrame sorted by the specified column(s). |

**Referencing Columns/Column Operations:**
* In pandas DataFrames we would create a new column based on the value of two other columns via:
```
df['Range'] = df['High'] - df['Low']
```
* In Spark DataFrame we would do the same via:
```
from pyspark.sql.functions import col

df = df.withColumn('Range', col('High') - col('Low'))
```

**Creating Functions in Spark:**
* In python we define functions via:
```
def my_specialfunc(h,l,o,c):
    return ((h-l)*(o-c))
```
* In Spark you would do the same via:
```
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType

my_specialfunc_udf = udf(lambda h,l,o,c : ((h-l)*(o-c)), DoubleType())
```

**Selecting Specific Columns in Spark DataFrame:**
```
new_df = df.select(['col1', 'col2'])
```

**GroupBy in Spark DataFrame:**
```
from pyspark.sql import functions as F

df_out = df_sales.groupBy(col("State")).agg(F.sum(col("Amount")),F.mean("Amount"))
```

**Sorting in Spark DataFrame:**
```
from pyspark.sql import functions as F

df_out = df_sales.groupBy(col("State"))\
            .agg(F.sum(col("Amount")).alias("sumAmount"))\
            .orderBy(col("sumAmount"), ascending=False)
```

---

#### Day 1 Afternoon
### Machine Learning and Spark
**Machine Learning on Spark:**
* Algorithms: common learning algorithms such as classification, regression, clustering, and collaborative filtering
* Featurization: feature extraction, transformation, dimensionality reduction, and selection
* Pipelines: tools for constructing, evaluating, and tuning ML Pipelines
* Persistence: saving and load algorithms, models, and Pipelines
* Utilities: linear algebra, statistics, data handling, etc.

**Spark Machine Learning Pipeline Terms:**
* Pipeline:
    * Running a sequence of algorithms in a set order to process & learn from data
    * Many Data Science workflows can be described as a pipeline, i.e. just a sequential application of various Transforms and Estimators
* Transformers
    * They implement a transform() method
    * They convert one DataFrame into another, usually by adding columns
    * For example, this is how you get predictions, through using a transform method and adding a column of predictions to your DataFrame
    * Examples of transformers: VectorAssembler, Tokenizer, StopWordsRemover, and many more
* Estimators
    * Any algorithm that fits or trains on data
    * They implement a fit() method whose argument is a DataFrame
    * The output of fit() is another type called a Model, which is actusally a Transformer
    * Examples of estimators: LogisticRegression, DecisionTreeRegressor, and many more

**Machine Learning Libraries:**
* In the past, there was a trade-off between using the two different machine learning libraries available - Spark MLlib and Spark ML
* The terms ‘MLlib’ & ‘ML’ can be used in a few different ways depending on what you’re reading and the Spark version you’re looking at
* Spark is now using the dataframe-based API as the default API
* The RDD-based API is expected to be removed in Spark 3.0

**Transformers:**
* They implement a transform method
* They convert one DataFrame into another, usually by adding columns
* Examples: VectorAssembler, Tokenizer, and StopWordsRemover

**Estimators:**
* According to the documentation: "An Estimator abstracts the concept of a learning algorithm or any algorithm that fits or trains on data"
* They implement a fit method whose argument is a DataFrame
* The output of fit is another type called Model, which is a Transformer.

**Pipelines:**
* Many data science workflows can be described as sequential application of various Transforms and Estimators.

---
### Python Week 8
#### Day 3 Morning
#### Recommender Systems
**Recommenders:**
* A recommender is an information filtering system that seeks to predict a user’s rating/preference for something.
* Generally, recommenders make recommendations one of three ways:
    1) Popularity - recommend what’s most popular (trending on Twitter)
    2) Content-based - use similarities based on attributes of items (text descriptions,features) to group/cluster them and provide recommendations for things that are similar based on user preference.
        * Example: [SkiRunRecommender](https://github.com/kfarbman/ski-recommender)
    3) Collaborative filtering (CF) - for a given user, use ratings/preferences of similar users to predict which unrated items the user would like.
        * Example: [Board Game Geek Recommender](https://github.com/Jomonsugi/bgg-exploration)
    * Hybrid approaches are possible.

**Ratings - Explicit vs. Implicit:**
* Explicit Ratings
    * Allow user to unequivocally quantify how much he/she liked something.
    * Examples: Did you like this song? How many stars would you give this movie?

* Implicit Ratings
    * Makes inference from user behavior.
    * Examples: How many times have you played that track? Which songs are you playing? What movies are you watching?

**Collaborative Filtering:**
* Conceptually: use ratings/preferences of similar users or items to predict which unrated items the user would like. Utilizes user behavior, not item content.
* Types of collaborative filtering:
    * Memory, neighborhood based: Look for k-NN most similar items/users to provide the rating/preference (this morning).
    * Model, matrix factorization based: Usually use dimensionality reduction and latent factors to make a model to predict ratings (this afternoon).
    * Hybrid: mixes both and more. Recommenders are an active area of research.

**Collaborative Filtering, Neighborhood-Based:**
The Process:
    1) Gather ratings (explicit/implicit)
    2) Put ratings in a matrix (usually rows are users, columns are items)
    3) Determine where you need a rating (usually items for a given user)
    4) Realize that your rows and columns are both vectors
    5) Can find similar vectors (user-user or item-item)
    6) Use the similarity of k users/rows to make a new rating
* Usually Item-Item is preferred because:
    * Less computationally expensive
    * Item ratings more heavily populated than user ratings - more robust results

**Calculating Similarity:**
* Euclidean distance
    * The “straight line” distance between two vectors
* Euclidean similarity
    * Ranges from 0 (not similar) to 1 (very similar)
* Cosine Similarity
    * How much do the vectors point the same way? (-1 opposite, 0 orthogonal, +1 same)
    * Cosine similarity is the most commonly used distance metric in recommender systems.
* Standardized cosine similarity
    * Ranges from 0 (not similar) to 1 (very similar)
* Pearson's correlation coefficient (R)
    * Quantifies the strength of a linear relationship between two vectors. -1 inversely related, 0 no relationship, 1 positively correlated
* Pearson Similarity
    * Ranges from 0 (not similar) to 1 (very similar)
* Jaccard Index (or Similarity Coefficient)
    * It’s a statistic for comparing the similarity of sample sets. Ranges from 0 (no overlap or similarity) to 1 (complete overlap or similarity)
    * Good for implicit rankings

**Similarity Matrix:**
* Use the similarity metric to calculate the similarity of all the items to each other, resulting in a similarity matrix.

**Collaborative Filtering Difficulties:**
* Cold start: For a new user that has no ratings, what do you recommend to him/her? For a new item with no ratings, how do you recommend it?
* Scalability: For neighborhood based item-item recommenders, mn2 similarity metrics must be calculated (hopefully the night before) before ratings can be calculated. Requires a lot of computation power.
* Sparsity: There are often many items but the ratings are sparse. High dimensionality without data. Overfitting!
    * Enter CF dimensionality reduction technique, matrix factorization

**Test/Train vs. Cross-Val:**
* Don't test/train split with recommender systems, instead use cross-val.

---

#### Day 3 Afternoon
#### Matrix Factorization for Recommender Systems
**Downfall of Collaborative Filtering:**
* Sparse Ratings Matrix, could be very, very sparse (99% of entries unknown)
* Item-Item I like action movies → rate Top Gun and Mission Impossible 5s.→ I’m recommended Jerry Maguire even though I won’t like it.
* User-User I like Tom Cruise → rate Top Gun and Mission Impossible 5s. → I’m recommended Transformers even though I won’t like it.

**Factorization:**
* Factorization could account for something along the lines of these attributes as was our hope in Linear Regression.
* All of the factorization models that we know can be interpreted as a linear combination of bases.
* There’s a chance, especially with NMF, that those bases, latent features, could correspond with some of these “attributes” that we’re looking to describe the movies with.

**Factorization Problem:**
* Problem: PCA, SVD and NMF all must be computed on a dense data matrix, X.
* Potential Solution: inpute missing values, naively, with something like the mean of the known values. Note: This is what sklearn does when it says it factorizes sparse matrices.

**Factorization Goal:**
* Create a factorization for a sparse data matrix, X, into
U · V , such that the reconstruction to Xˆ serves as a model.
* More formally, for a previously unknown entry in X, Xi,j the corresponding entry in Xˆ, Xˆi,j serves as a prediction.
* Note: Since we could easily overfit the known values in X we want to regularize, one way to do this is by reducing the inner dimension in U and V, k.

**Difference between CF and MF:**
* Collaborative Filtering (neighborhood models) → Memory Based. Just store data so we can query what/whom is most similar when asked to recommend.
* Factorization Techniques → Model Based. Creates predictions, from which the most suitable can be recommended.

**Computing the Factorization:**
* Similar to what we did to find the factorization in NMF, we’re going to minimize a cost function.
* Now, though we can’t do it at the level of the entirety of X, since it is sparse.
* However, we can optimize with respect to the data in X that we do have.

**Baseline Predictor (Biases):**
* Much of the observed ratings are associated with a specific user’s personality or an item’s intrinsic value, not an interaction between the two which is what get captured in the factorization.
* To encapsulate these effects, which do not involve user-item interactions, we introduce baseline predictors.
    * μ: Baseline average value in X.
    * bi: Baseline rating for user i.
    * bj: Baseline rating for item j.
* From this we can describe our predictions with:
    Xˆi,j = μ + b i + b j + U i · V j

**Regularization:**
* Another way to regularize our decomposition to help prevent from overfitting to our sparse data is via a penalty, λ, placed on the magnitude of: bi , bj , Ui and Vj . The most common is the L2 norm.

**Validation:**
* Validating any recommender is difficult, but it is necessary as we’re going to want to tune the hyperparameters that we introduced into our model, ν and λ.
* The most frequently used metric is Root Mean Squared Error (RMSE) on the known data

**MF - Pros/Cons:**
* Pros
    * Decent with sparsity, so long as we regularize.
    * Prediction is fast, only need to do an inner product
    * Can inspect latent features for topical meaning
    * Can be extended to include side information.
* Cons
    * Need to re-factorize with new data. Very slow
    * Fails in the cold start case
    * Not great open source tools for huge matrices
    * Difficult to tune directly to the type of recommendationn you want to make. Tied to the difficulty of measuring success.

**Advanced Factorization Methods:**
* Non-negativity constraint → More interpretable latent features.
* SVD++ → uses implicit feedback (clicks, likes, etc.) to enhance model.
* Time-aware factor model → accounts for temporal information about data.

---
### Python Week 9
#### Day 1 Morning
#### Flask
* To open any application on your computer via the command line:
```
open -a 'Spotify'
```

**First Flask Example:**
```
%%writefile src/flask_ex1.py

"""import Flask"""
from flask import Flask

"""initialize the Flask app, note that all routing blocks use @app"""
app = Flask(__name__)  # instantiate a flask app object

"""routing blocks - note there is only one in this case - @app.route('/')

home page - the first place your app will go"""
@app.route('/', methods = ['GET', 'POST'])  # GET is the default, more about GET and POST below
"""the function below will be executed at the host and port followed by '/'
the name of the function that will be executed at '/'. Its name is arbitrary."""
def index():
    return 'Hello!'

"""no more routing blocks"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
    """the combination of host and port are where we look for it in our browser
    e.g. http://0.0.0.0:8080/
    setting debug=True to see error messages - wouldn't do this on a final deployed app"""
```
* Now go to the terminal and type:
```
$ python src/flask_ex1.py
```
* Now navigate to the specified address in the browser. When you are done, hit Ctrl-C to stop the app.

**GET vs. POST:**
* Note: the route - `app.route('/')` - has two methods: GET and POST
* Understanding these two terms requires putting in context how the browser (the client) is interacting with information stored on your local computer (the server)
* GET is used by the browser to get information from the server. In the example above, the browser asked the server to get the information returned by the `index` function. Every routing block has a GET method (it's the default).
* POST tells the server that the browser wants to send information to it. HTML forms usually transmit data this way.
* GET requests will show information in the browser address bar while POST won't. Confidential information should be sent with POST.

---
### Python Week 9
#### Day 2 Morning
#### Graph Theory
**Introduction:**
* Two parts of a graph: nodes and edges
    * Examples: Roads and cities
    * A single node with no edge is still a graph; however a single edge without a node cannot be a graph.
* Two kinds of graphs:
    1) Directed graphs
    2) Undirected graphs
    * Those kinds of graphs can be: cyclic or acyclic
* Directed Acyclic Graph (DAG): A special type of graph we use for computation
* Complete Graph: When all nodes in a graph are connected by edges
* Clique: Fully connected sub-graph (minimum of three nodes)

**Terms:**
* Network and graph are used interchangeably
* A graph is a set of nodes or vertices joined by lines or edges.
* Networks are all around us (roads, friendships, collaborations)
* Provide a data structure that is intuitive

**NetworkX:**
* Python language data structures for graphs, digraphs, and multigraphs
* Many standard graph algorithms
* Network structure and analysis measures
* Generators for classic graphs, random graphs, and synthetic networks
* Nodes can be  ̈anything”(e.g. text, images, XML records)
* Edges can hold arbitrary data (e.g. weights, time-series)
* Originally developed by Aric Hagberg, Dan Schult, and Pieter Swart at Los Alamos National Laboratory, but now it is the main package to work with Network in Python and is developed by many others including Google.

**Graph or Network Data Examples:**
* Facebook (friends)
* Internet
* Roads (traveling salesman)
* Movies (common actors, Kevin Bacon)
* Scientific collaboration (Paul Erdos)
* Customer Journey (ClickFox)
* Cybersecurity

**Is NetworkX the right library?:**
* When to USE NetworkX to perform network analysis?
    * Unlike many other tools, it is designed to handle data on a scale relevant to modern problems.
    * Most of the core algorithms rely on extremely fast legacy code
    * Highly flexible graph implementations (a graph/node can be anything!)
    * Extensive set of native readable and writable formats
    * Takes advantage of Python’s ability to pull data from the Internet or databases
* When to AVOID NetworkX to perform network analysis?
    * Large-scale problems that require faster approaches (i.e. Facebook/Twitter whole social network...)
    * Better use of resources/threads than Python
*When NetworkX fails try Apache Spark’s GraphX*

**Types of Networks:**

|  Graph Type | NetworkX Class  |
|---|---|
| Undirected Simple  | Graph  |
| Directed Simple  | DiGraph  |
| With Self-Loops  | Graph, DiGraph  |
| With Parallel Edges  | MultiGraph, MultiDiGraph  |

**Drawing and Plotting:**
* Graphs are intuitive in part due to ease of visualization
* IT is possible to draw graphs in:
    * NetworkX
    * GraphViz
    * matplotlib

**More Terminology:**
* Neighbors: The neighbors of a node are the nodes that it is connected to
* Degree: The degree of a node is the number of neighbors for a given node
* Directed graphs can be split into indegree and outdegree.
* Walk: A walk is a sequence of nodes and edges that connect
* Path: A path is a walk where no node is crossed twice
* A closed path is known as a cycle
* Connected: A graph is connected if every pair of vertices is connected by some path
* Subgraph: A subgraph is a subset of the nodes of a graph and all the edges between them
* Graph Diameter: The diameter of a graph is the largest number of vertices that much be traversed in order to get from one vertex to any other vertex

---
### Python Week 9
#### Day 3 Morning
#### Agile/Scrum
**Introduction:**
* Scrum is a way to implement agile
* Other frameworks exist for implementing agile like kanban
* Communication is central to flexibility
* The reality is that many (most?) teams have toward an agile organizational structure
* Agile: A structured and iterative approach to project management and product development

**Agile vs. Waterfall:**
* Waterfall or the traditional approach
    * linear approach to software development
    * e.g. design → code → test → fix → deploy
    * developers and customers agree on a product early
    * progress is easily measured
    * process is not easily modified
* Agile
    * Iterative team based approach to development
    * Time is boxed into phases called sprints
    * Gives you the ability to respond to change
    * Agile requires very frequent meetings to keep things realistic and on track

**Scrum:**
* Commitment to short iterations of work. It uses four types ceremonies:
    1) Sprint planning: A team planning meeting that determines what to complete in the coming sprint.
    2) Daily stand-up: Also known as a daily scrum, a 15-minute mini-meeting for the software team to sync
    3) Sprint demo: A sharing meeting where the team shows what they’ve shipped in that sprint.
    4) Sprint retrospective: A review of what did and didn’t go well with actions to make the next sprint better.

**Scrum Roles:**
* The scrum master - Scrum masters also look to resolve impediments and distractions for the development team
* The product owner - The product frequently converses with the team about each user story. Some are informal one-on-one conversations. Some are formal conversations with the whole team.
    * Your product owner has only a vague understanding of models

**Scrum Master (AKA The Agile Coach):**
* Stand-up will happen nearly down with you almost every day
* Sprint planning is part of the stand-up
* The sprint duration will be at the coach’s discretion
* Sprints can (for example) be planned to end Wednesdays and Fridays
* Scrum master’s document progress as well as do review code
* If you need to chat or talk about something at length, setup a time outside of stand-up

---
### Next Cohort
#### Docker
**What is Docker?:**
* Open-source project based on Linux containers.
* Docker is a software containerization platform.
* A docker container is a stand-alone piece of software that includes everything needed to run it.
* Containers are isolated from each other, but can share libraries where possible.
* Containers are created with Linux, but share a kernel with almost any type of OS.
    * A kernel is the central part of the OS.

**Docker Motivation:**
* Shipping code to the server should not be hard.
* The developer can specify what's inside the container, and IT just needs to handle the container.
* No matter where the container goes, what's inside will work the same way.

**Docker Timeline and Popularity:**
* Docker was made open source in 2013 by Soloman Hykes at dotCloud.
* Main contributors in 2016: Docker team, Cisco, Google, Huawei, IBM, Microsoft.
* Has been downloaded more than 13 billion times as of 2017.
* Main reasons  for popularity:
    1) Ease of use: Build and test it on your laptop and it will run anywhere.
    2) Speed: Containers are sandboxed environments running on the kernel that take up fewer resources and are faster to start-up than virtual machines.
    3) DockerHub: The free app-store for Docker images.
    4) Modularity and scalability: An application can be divided into multiple containers but those containers can communicate.

**Docker Volumes:**
* Docker volumes are the preferred way of persisting data needed and created by containers.
* Containers by themselves have a writable layer (you can store data in a container without using a volume), but a volume is better because:
    * it doesn’t increase the size of the container using it
    * the volume exists outside of the lifecycle of the container
    * the volume can be shared with other containers
* You can create a volume when a container is made.
* You can move data back and forth between the host file system and the docker volume.

**Docker Command Reference:**
| Command  | Description  |
|---|---|
| `docker run <image>`  | Make and start a container from an image  |
| `docker start <name | id>`  | Start a pre-existing container  |
| `docker stop <name | id>`  | Stop a container  |
| `docker ps`  | Show running containers  |
| `docker ps -a`  | Show running and stopped containers  |
| `docker rm <name | id>`  | Removes a container  |
| `docker`  | Show available docker commands  |
| `docker COMMAND --help`  | Gets help on COMMAND  |
| `docker logs <id>`  | Return log of container (including browser link and token)  |
