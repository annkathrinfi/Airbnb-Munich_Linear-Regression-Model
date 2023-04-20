# Airbnb Munich: Linear-Regression-Model

This project is part of the Udacity Nanodegee "Data Science". The goal is to apply the newly developed skills of "Lesson 2: The Data Science Process" to answer 3-5 questions, related to real-world data.

## Context
The database is the Airbnb listings-data for Munich from 24th December 2021.
Key questions that will be answered are the following:

1. Which are the popular neighbourhoods of Munich?
2. In which neighbourhood are most listings located?
3. Do listings in a popular neighbourhood get more reviews per month?
4. Which features influence how many reviews a listing will receive in a month?
5. Can we predict the number of reviews per month with linear regression?

## Libraries used
The following libraries were used to prepare the data, build a regression model and to visualise outcome:
```
# Import Packages
import numpy as np
import pandas as pd
import seaborn as sb

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import r2_score

from skew_autotransform import skew_autotransform

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline
```

## Explanation of the Files in the Repository
The following files are uploaded in the repository:
- ``Airbnb Munich_Prediction of Reviews per Month.ipynb``: The Jupyter notebook contains the complete Python code used for the data preparation, exploratory analysis and model building.
- ``listings.csv.gz``: compressed csv-file, which holds the data used for this analysis.
- ``skew_autotransform.py``: Python function to automatically transform skewed data in Pandas DataFrame (https://github.com/datamadness/Automatic-skewness-transformation-for-Pandas-DataFrame)

## Summary of the Results of the Analysis
In this project, I analysed the Airbnb Dataset for listings in Munich. 
Exploratory analyses discovered the most popular neighbourhoods of Munich as well as the ones with the most listings. The results show that listings in popular neighbourhoods do not necessarily get more ratings per month. 
The linear regression model showed the most important features for the prediction of review per month, where the listing's price, its overall rating and its availability are among the top 5. 
However, the features in the model only explained 31% of the variance in the number of reviews a listing gets per month. Features such as the number of actual bookings per month and factors of human behaviour and the guests' personalities are not included in the model, but might be helpful for a more accurate prediction.
Find a blog post based on the results of this analysis at https://medium.com/@annkafi/how-can-you-improve-the-number-of-reviews-your-airbnb-listing-receives-3db3c4d92cdc . 
