# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:13:31 2020

@author: Caleb
"""


"""0) Setting up the environment
====================================
"""

"""
- A very important part that is never mentionned is setting up python to Run

- All Python files point to a particular folder where all the inputs and outputs should
be placed.

- To find where this folder is, use:

"""
import os
os.getcwd()

"""1) First Steps
==============================================="""
# Importing packages
import pandas as pd

# Read an Excel file called Test.xlsx
pd.read_excel("Test.xlsx")


# Reading a csv called Test.csv
pd.read_csv("Test.csv")
"""
-We use the 'as pd' to be a short form for pandas throughout our code
-The most important packages for use are numpy(np), pandas(pd) and os

- Pandas is used to manipulate data. (filtering etc)

- Numpy creates vectors that should only contain numbers and therefore is a very fast way
of performing numerical calculations.

"""


"""2) LOOPS
======================="""
for i in range(10):
    print(i)

"""
- For loops are some of the most important tools in Python.

- Typically if you want to look at all the values in a vector, you use a for loop

- This means that Python goes through a list and performs operations

"""
