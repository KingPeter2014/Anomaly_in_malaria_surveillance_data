#Important library importation
import numpy as np
import csv
import warnings
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import seaborn as sns
from itertools import chain
from datetime import datetime
from dask import dataframe as dd #Alternative to read large csv instead of pandas

#Read in CSV file for malaria case data
def readMalariaData(state = 'All'):
    #moduleName = input('Enter module name:')
    frames = []
    cnt = 1
    chunksize = 10 ** 5
    filename = 'integrated_datasetfull.csv'
    for chunk in pd.read_csv(filename, delimiter = '\\t', chunksize=chunksize):
        cleanedabit = chunk.dropna(subset=['notification.year','notification.month','exam.result'])
        if state != 'All':
        	cleanedabit = cleanedabit.loc[cleanedabit['notification.state'] == state]
        frames.append(cleanedabit) #'exam.month','treatment.month','treatment.year','notification.year'
        #print(f" {cnt}: {chunk.shape} : removed empty result data->{cleanedabit.shape}" )
        #print(cnt)
    malariaIntegratedDataset = pd.concat(frames)
    return malariaIntegratedDataset

def combineDates(year,month,day):
    date_time_str = str(year)  + str(month)  + str(day)
    return datetime.strptime(date_time_str, '%d/%m/%y')
def cutString(st):
    return st[1:len(st)]
def removeLastXter(st):
    return st[0:len(st)-1]

def formatColumnData(malariaIntegratedDataset):
	malariaIntegratedDataset = malariaIntegratedDataset.sort_values(by=['notification.year','notification.month'])
	malariaIntegratedDataset = malariaIntegratedDataset.fillna(0)
	malariaIntegratedDataset["notification.year"] = malariaIntegratedDataset["notification.year"].apply(np.int64)
	malariaIntegratedDataset["notification.month"] = malariaIntegratedDataset["notification.month"].apply(np.int64)
	malariaIntegratedDataset["notification.county"] = malariaIntegratedDataset["notification.county"].apply(np.int64)
	malariaIntegratedDataset["scheme"] = malariaIntegratedDataset["scheme"].apply(np.int64)
	malariaIntegratedDataset["qty.parasites"] = malariaIntegratedDataset["qty.parasites"].apply(np.int64)
	return malariaIntegratedDataset

def prepareTimeSeries(malariaIntegratedDataset):
	# Prepare for Time series
	malariaIntegratedDataset["day"] = 1
	malariaIntegratedDataset["day"] = malariaIntegratedDataset["day"].apply(np.int64)
	malariaIntegratedDataset["year"] = malariaIntegratedDataset["notification.year"]
	malariaIntegratedDataset["month"] = malariaIntegratedDataset["notification.month"]
	return malariaIntegratedDataset
def extractNegativesByStateByMonth(malariaIntegratedDataset):
	testbkdown = malariaIntegratedDataset.groupby(['notification.year','notification.month','notification.state','exam.result']).size().reset_index(name='testbreakdown')
	return testbkdown

def getNegativesAndPositives(df):
    neg = df.loc[df['exam.result'].isin(['negative'])]
    pos  = df.loc[df['exam.result'] != 'negative']
    neg = neg.groupby(['Date']).sum().reset_index()
    pos = pos.groupby(['Date']).sum().reset_index()
    return neg,pos
def calcMonthlyPrevalence(negatives,positives):
    totals = list()
    neg = negatives['testbreakdown'];pos = positives['testbreakdown']
    totals = neg + pos
    monthPrevalence = pos/totals
    negatives['monthlyPrevalence'] = monthPrevalence
    negatives['negatives'] = neg
    negatives['positives'] = pos
    negatives['totalTests'] = totals
    stateData = negatives.drop(['testbreakdown', 'day','year','month'], axis=1)
    return stateData