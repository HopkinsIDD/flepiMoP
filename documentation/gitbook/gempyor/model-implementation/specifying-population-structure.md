---
description: >-
  This page describes how users specify the names, sizes, and connectivities of
  the different subpopulations comprising the total population to be modeled
---

# Specifying population structure

## Overview

The `subpop_setup` section of the configuration file is where users can input the information required to define a population structure on which to simulate the model. The options allow the user to determine the population size of each subpopulation that makes up the overall population, and to specify the amount of mixing that occurs between each pair of subpopulations.

An example configuration file with the global header and the spatial\_setup section is below:

```
name: test_simulation
data_path: data
model_output_dirname: model_output
start_date: 2020-01-01
end_date: 2020-12-31
nslots: 100

subpop_setup:
  geodata: model_input/geodata.csv
  mobility: model_input/mobility.csv
```

## Items and options

| Config Item | Required?    | Type/Format  | Description                           |
| ----------- | ------------ | ------------ | ------------------------------------- |
| geodata     | **required** | path to file | path to file relative to `data_path`  |
| mobility    | **required** | path to file | path to file relative to `data_path`  |
| selected    | **optional** | string       | name of selected location in`geodata` |

### `geodata` file

* `geodata` is a .csv with column headers, with at least two columns: `subpop` and `population`.
* `nodenames` is the name of a column in `geodata` that specifies unique geographical identification strings for each subpopulation.
* `selected` is the list of selected locations in geodata to be modeled

#### Example geodata file format

```
subpop,population
10001,1000
20002,2000
```

### `mobility` file

The `mobility` file is a .csv file (it has to contain .csv as extension) with long form comma separated values. Columns have to be named `ori`, `dest`, `amount,` with amount being the average number individuals moving from the origin subpopulation `ori` to destination subpopulation `dest` on any given day. Details on the mathematics of this model of contact are explained in the [Model Description section](../model-description.md#mixing-between-subpopulations). Unassigned relations are assumed to be zero. The location entries in the `ori` and `dest` columns should match exactly the `subpop` column in `geodata.csv`

#### Example mobility file format

```
ori, dest, amount
10001, 20002, 3
20002, 10001, 3
```

It is also possible, but **not recommended** to specify the `mobility` file as a .txt with space-separated values in the shape of a matrix. This matrix is symmetric and of size K x K, with K being the number of rows in `geodata`. The above example corresponds to

```
0 3
3 0
```

## Examples

#### Example 1

To simulate a simple population structure with two subpopulations, a large province with 10,000 individuals and a small province with only 1,000 individuals, where every day 100 residents of the large province travel to the small province and interact with residents there, and 50 residents of the small province visit the large province

```
subpop_setup:
  geodata: model_input/geodata.csv
  mobility: model_input/mobility.csv
```

`geodata.csv` contains the population structure (with columns `subpop` and `population`)

```
subpop,          population
large_province, 10000
small_province, 1000
```

`mobility.csv` contains

```
ori,            dest,           amount
large_province, small_province, 100
small_province, large_province, 50
```

