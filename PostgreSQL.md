### PostgreSQL

#### Opening/Closing PostgreSQL
* To open PostgreSQL in terminal:
~~~
psql
~~~
* To close PostgreSQL in terminal:
~~~
\q
~~~

#### Create a Database
* To create a database:
~~~
CREATE DATABASE animalshp;
~~~

#### Connecting to Databases
* To connect to a database after opening psql:
~~~
\c animalshp
~~~
* To connect to a database before opening psql:
~~~
psql animalshp
~~~

#### Create Database From CSV
~~~
CREATE TABLE animals (
    species VARCHAR (20),
    vertebrate_class VARCHAR (20),
    appearance VARCHAR (20),
    num_leg INT);

COPY animals FROM '/Users/chrisfeller/Downloads/animals.csv' DELIMITER ',' CSV HEADER;
~~~

#### Tables
* Animals:

| species   | vertebrate_class | appearance | num_legs |
|-----------|------------------|------------|----------|
| cat       | mammal           | fur        | 4        |
| rat       | mammal           | fur        | 4        |
| owl       | bird             | feathers   | 2        |
| snake     | reptile          | dryscales  | 0        |
| toad      | amphibian        | smoothskin | 4        |
| alligator | reptile          | dry scales | 4        |

* Pets:

| name        | species | owner              | gender | color         |
|-------------|---------|--------------------|--------|---------------|
| Nagini      | snake   | Lord Voldemort     | female | green         |
| Hedwig      | owl     | Harry Potter       | female | snow white    |
| Scabbers    | rat     | Ron Weasley        | male   | unspecified   |
| Pigwidgeon  | owl     | Ron Weasley        | male   | grey          |
| Crookshanks | cat     | Herminone Granger  | male   | ginger        |
| Mrs Norris  | cat     | Argus Filch        | female | dust-coloured |
| Trevor      | toad    | Neville Longbottom | male   | brown         |

#### DESCRIBE TABLE
* To view table schema:
~~~
\d+ animals
~~~
#### SELECT
* Select everything from a table:
~~~
SELECT * FROM animals;
~~~

#### WHERE
* Use the `WHERE` clause to show the appearance of 'rats':
~~~
SELECT appearance
FROM pets
WHERE species = 'rat';
~~~
* Use `WHERE` to display the owners of male pets, the name of the pets and their vertebrate class.
~~~
SELECT pets.owner, pets.name, animals.vertebrate_class
FROM pets
JOIN animals on pets.species = animals.species
WHERE pets.gender = 'male';
~~~
#### IN
* Use `IN` to show the species for animals with vertebrate_class 'mammal' and 'amphibian.'
~~~
SELECT species
FROM animals
WHERE vertebrate_class IN ('mammal', 'amphibian');
~~~

#### BETWEEN
* Use `BETWEEN xxx AND xxx` to show species that have at least one leg, but no more than 3 legs:
~~~
SELECT species
FROM animals
WHERE num_leg BETWEEN 1 AND 3;
~~~

#### LIKE
* Use `LIKE` to show species that have an appearance that starts with 'f':
~~~
SELECT species
FROM animals
WHERE appearance LIKE 'f%';
~~~

#### CASE
* Use `CASE` to show pet names and a column to indicate whether the pet's name is long or short (a long name is strictly more than 6 characters long). Filter to select only female pets.
~~~
SELECT name, CASE
    WHEN LENGTH(name) > 6 THEN 'long'
    ELSE 'short'
    END
FROM pets
WHERE gender = 'female';
~~~

#### DISTINCT
* Use `DISTINCT` to list the species that can be pets - each species should appear only once:
~~~
SELECT DISTINCT(species)
FROM pets;
~~~

#### COUNT
* Use `COUNT` to see how many pets Ron Weasley owns:
~~~
SELECT COUNT(*)
FROM pets
WHERE owner = 'Ron Weasley';
~~~

#### STRING CONCATENATION
* Use string concatenation to output the string 'Ron Weasley has X pets',
where 'X' is the number of pets he has.
~~~
SELECT 'Ron Weasley has ' || COUNT(name) || ' pets.' AS answer
FROM pets
WHERE owner = 'Ron Weasley';
~~~

#### GROUP BY
* Use `GROUP BY` to count how many pets each owner has. Give the output as 'NAME has X pets', with names alphabetically ordered (use `ORDER BY`).
~~~
SELECT owner || ' has ' || COUNT(name) || ' pets'
FROM pets
GROUP BY owner
ORDER BY owner;
~~~

#### HAVING
* Use `HAVING` to select all owner who have exactly one pet.
~~~
SELECT owner
FROM pets
GROUP BY owner
HAVING COUNT(name) = 1;
~~~
* Use `HAVING` to select owners who have pets of more than one vertebrate class.
~~~
SELECT pets.owner
FROM pets
JOIN animals on pets.species = animals.species
GROUP BY owner
HAVING COUNT(animals.vertebrate_class) > 1;
~~~
* Use `HAVING` to select owners who have exactly one pet.
~~~
SELECT owner
FROM pets
GROUP BY owner
HAVING COUNT(name) = 1;
~~~

#### JOIN
* Use `JOIN` to display the names of the pets and their vertebrate class.
~~~
SELECT pets.name, animals.vertebrate_class
FROM pets
JOIN animals on pets.species = animals.species;
~~~
* Now let's find out what our pets look like: list all the pets, their appearance, and their color.
~~~
SELECT pets.name, animals.appearance, pets.color
FROM pets
JOIN animals on pets.species = animals.species;
~~~
#### ALTER TABLE
* To change a field type to date:
~~~
ALTER TABLE tablename ALTER COLUMN columnname TYPE DATE using to_date (columnname, 'MM-DD-YYYY');
~~~
* To change a field type to integer:
~~~
ALTER TABLE beds ALTER COLUMN available_residential_beds TYPE integer USING (available_residential_beds::integer);
~~~
