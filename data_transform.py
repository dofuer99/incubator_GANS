#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 03:11:10 2018

@author: zeng
"""

import numpy as np
#import pandas as pd
#import tensorflow as tf


#encode the data from categorical value into numerial values
class encode():
    def __init__(self, df):
        self.relevant = ['age', 'job', 'marital', 'education', 'default', 
                         'balance', 'housing', 'loan']
        self.raw = []
        self.df = df
        
    #job column
    def encode_job(self, job):
        joblist = ['entrepreneur', 'blue-collar',
       'unknown', 'retired', 'admin.', 'services', 'self-employed',
       'unemployed', 'housemaid', 'student', 'management', 'technician']
        return joblist.index(job) 

    #marital_status
    def encode_marital(self, status):
        marital_status = ['married', 'single', 'divorced']
        return marital_status.index(status) 

    #education
    def encode_education(self, edu):
        education = ['tertiary', 'secondary', 'primary', 'unknown']
        return education.index(edu)

    #default
    def encode_default(self, default):
        if default == 'no':
            return -1
        elif default == 'yes':
            return 1

    #housing_loan
    def encode_housing(self, housing_loan):
        if housing_loan == 'no':
            return -1
        elif housing_loan == 'yes':
            return 1
        
    #loan        
    def encode_loan(self, loan):
        if loan == 'no':
            return -1
        elif loan == 'yes':
            return 1  
        
    def encode_begin(self):
        for index, row in self.df[self.relevant].iterrows():
            self.raw.append([row[0],
                     self.encode_job(row[1]),
                     self.encode_marital(row[2]),
                     self.encode_education(row[3]),
                     self.encode_default(row[4]),
                     row[5],
                     self.encode_housing(row[6]),
                     self.encode_loan(row[7])])
        return self.raw  
 
#scale the data into the range[-1, 1]
class scaled():
    def __init__(self, raw):
        self.raw = np.array(raw)
        self.scaled = np.zeros(self.raw.shape)
        self.cache = []
        
    def scale_minmax(self, column):
        column_min = min(column)
        column_max = max(column)
        column_scaled = 1.0 * (column - column_min)/(column_max - column_min) * 2 - 1
        return column_scaled, column_max, column_min  
    
    def scale_variance(self, column):
        column_mean = np.mean(column)
        column_std = np.std(column)
        column_min = column_mean - 4 * column_std
        column_max = column_mean + 4 * column_std
        column_scaled = 1.0 * (column - column_min)/(column_max - column_min) * 2 - 1
        return column_scaled, column_max, column_min
    
    def scale_begin(self):
        for i in range(self.raw.shape[1]):
            column = self.raw[:, i]
            if i != 50:
                column_scaled, column_max, column_min = self.scale_minmax(column)
            else:
                column_scaled, column_max, column_min = self.scale_variance(column)
            self.scaled[:, i] = column_scaled
            self.cache.append([column_max, column_min])
        return self.scaled, self.cache
    
#unscale the data
class unscaled():
    def __init__(self, scaled, cache):
        self.scaled = scaled
        self.cache = cache
        self.unscaled = np.zeros(self.scaled.shape)
        
    def unscale_minmax(self, column_scaled, column_max, column_min):
        column_unscaled = (column_scaled + 1)/2.0 * (column_max - column_min) + column_min
        return column_unscaled
    
#    def unscale_variance(self, column_scaled, column_mean, column_std):
#        column_unscaled = column_scaled * column_std + column_mean
         
    def unscale_begin(self):
        for i in range(self.scaled.shape[1]):
            column = self.scaled[:, i]
            column_unscaled = self.unscale_minmax(column, self.cache[i][0], self.cache[i][1])
            self.unscaled[:, i] = column_unscaled
        return self.unscaled

#round the data
class round_data():
    def __init__(self, unscaled):
        self.unscaled = unscaled
        self.rounded = []
        
    def round_all(self):
        for row in self.unscaled:
            for i in range(len(row)):
                if i < 6:
                    row[i] = np.int(np.round(row[i]))
                elif row[i] > 0 and i != 8:
                    row[i] = 1
                elif row[i] <= 0 and i != 8:
                    row[i] = -1
            self.rounded.append(row)
        return self.rounded









