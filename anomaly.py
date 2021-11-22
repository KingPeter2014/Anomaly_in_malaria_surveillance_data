import streamlit as st #streamlit run anomaly.py
#$ streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/streamlit_app.py
# check prophet version
import fbprophet
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import matplotlib
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import Grouper
from pandas import DataFrame
from pandas import to_datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.metrics import mean_absolute_error
import pandas as pd
from matplotlib.dates import DateFormatter
from pandas.plotting import lag_plot, autocorrelation_plot
from sklearn.model_selection import ParameterGrid
import holidays
import random

from celluloid import Camera
from collections import defaultdict
from functools import partial
from tqdm import tqdm

from tsmoothie.utils_func import sim_randomwalk, sim_seasonal_data
from tsmoothie.smoother import *
from tsmoothie.smoother import LowessSmoother
import ffmpeg

import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Call the function to read malaria case dataframe
from functions.amazonmalariadata import prepareTimeSeries,readMalariaData,formatColumnData,extractNegativesByStateByMonth
import warnings

import pickle
import base64
#________________________________________________________________________________
st.title('Malaria Monitoring Dashboard')



#Helper functions________________________________________________________________
def getNegativesAndPositives(df):
    neg = df.loc[df['exam.result'].isin(['negative'])]
    pos  = df.loc[df['exam.result'] != 'negative']
    neg = neg.groupby(['Date']).sum().reset_index()
    pos = pos.groupby(['Date']).sum().reset_index()
    return neg,pos
def calcMonthlyPrevalence(Negatives,Positives):
    totals = list()
    neg = Negatives['testperhr'];pos = Positives['testperhr']
    totals = neg + pos
    monthPrevalence = pos/totals
    Negatives['monthlyPrevalence'] = monthPrevalence
    Negatives['negatives'] = neg
    Negatives['positives'] = pos
    Negatives['totalTests'] = totals
    StateData = Negatives.drop(['testperhr', 'day','year','month'], axis=1)
    return StateData
def returnPrevalenceData(dataset,window):
    dset = dataset.reset_index()
    dset = dset[['Date','monthlyPrevalence']]
    dset['monthlyPrevalence'] = get_moving_average(dset['monthlyPrevalence'], window)
    #dset['monthlyPrevalence'] = get_moving_median(dset['monthlyPrevalence'], window=6)
    dset = dset.dropna().reset_index(drop=True)
    return dset

def slope(y1,y2,x1,x2):
    dy = y2 - y1
    dx = x2 - x1
    if dx !=0:
        s = dy/dx
    else:
        s = np.NaN
    return s

#@st.cache  # 👈 This function will be cached
def signalClass(dset,threshold,lag,region):
    x1=1;x2=2  
    tau = threshold;k = lag
    count = len(dset)
    
    lags = []
    fig2 = plt.subplots(figsize=(10,2))
    dset['label'] = np.ones(len(dset))
    plt.plot(dset['y'], color='black', alpha=0.2)
    for i in range(count-lag):
        y1 = dset['y'][i]
        y2 = dset['y'][i+ lag]
        slop = round(slope(y1,y2,x1,x2),3)
        per_change = round(mean_absolute_percentage_error(y1, y2),2)
        if slop > 0 and per_change > threshold:
            plt.scatter(dset.index[i+lag], y2, color='tab:red', alpha=1, label='Flareup')
            dset['label'].iloc[i] = 2
        elif slop < 0 and per_change > threshold:
            plt.scatter(dset.index[i+lag], y2, color='tab:blue', alpha=1, label='Decline')
            dset['label'].iloc[i] = 0
        else:
            #pass
            plt.scatter(dset.index[i+lag], y2, color='k', alpha=0.1, label='Steady state')
        title = region + ': k = ' + str(lag) + ',' + r'$\tau = $' + str(threshold) + '%'
        plt.title(title, fontsize=6)
        plt.ylabel('Proportion of positives')
        plt.xlabel('Time (Months)')
        lags.append(slop)
    return lags,fig2

def saveModelByPickle(model, pkl_path = "ProphetModel.pkl"):
    with open(pkl_path, "wb") as f:
        # Pickle the 'Prophet' model using the highest protocol available.
        pickle.dump(model, f)
def saveForcast(forecast,path='forecast.pkl'):
    # save the dataframe
    forecast.to_pickle(path)
    print("*** Data Saved ***")

def readModelByPickle(pkl_path = "ProphetModel.pkl",fcastpath='forecast.pkl'):
    # read the Prophet model object
    with open(pkl_path, 'rb') as f:
        m = pickle.load(f)
    fcast = pd.read_pickle(fcastpath)
    return fcast

def testUsing(model,testTimes,testValues): 
    #print(testData)
    future = pd.DataFrame(testTimes, columns = ['ds'])
    y_true = testValues.values
    
    forecast = model.predict(future)
    y_pred = forecast['yhat'].values
    
    # calculate MAE and MAPE between expected and predicted values
    mae = mean_absolute_error(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true,abs(y_pred))
    if MAPE <= 8:# and MAPE >= 5:
        print('MAPE: %.3f' % MAPE)
    return mae,MAPE,y_true,forecast

def getTrainAccuracy(model,testTimes,testValues): 
    #print(testData)
    future = pd.DataFrame(testTimes, columns = ['ds'])
    y_true = testValues.values
    #print(future)
    
    forecast = model.predict(future)
    y_pred = forecast['yhat'].values
    
    # calculate MAE and MAPE between expected and predicted values
    mae = mean_absolute_error(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true,abs(y_pred))
    return mae,MAPE
def plotTrainTest(datafr,testpoints, region='All'):
    #Train-Test Split
    X_tr = datafr.head(len(datafr)-testpoints)
    X_tst = datafr.tail(testpoints)
    pd.plotting.register_matplotlib_converters()
    f, ax = plt.subplots(figsize=(14,5))
    X_tr.plot(kind='line', x='ds', y='y', color='blue', label='Train', marker='o', ax=ax)
    X_tst.plot(kind='line', x='ds', y='y', color='red', label='Test', marker='o', ax=ax)
    plt.title('Traning and Test data for -> ' +  region, fontsize=8)
    plt.xlabel('Years')
    plt.ylabel('Positive case proportion')
    return X_tr,X_tst
def predictedActualPlot(model,test,forcast):
    # Plot the forecast with the actuals
    
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(10)
    ax.scatter(test.ds, test['y'], color='r')
    fig = model.plot(forcast, ax=ax)
    plt.xlabel('Years')
    plt.ylabel('Positive case proportion')
    plt.title(region)
def plotExpectedVsActual(y_true,y_pred,region):
    # plot expected vs actual
    plt.figure(figsize=(5,6))
    plt.plot(y_true, label='Actual',marker='o',linewidth=2, markersize=12)
    plt.plot(y_pred, label='Predicted',marker='*',linewidth=2, markersize=12)
    plt.legend()
    plt.title(region)
    #plt.xlabel(predictionyear)
    plt.ylabel('Positive proportion')
    #plt.legend('Train','a')
    
def makeParameterGrid():   
    params_grid = {'seasonality_mode':('multiplicative','additive'),
               'changepoint_prior_scale':[0.01],
              'holidays_prior_scale':[10],
              'n_changepoints' : [5,10],
            'changepoint_range' :[1]}
    grid = ParameterGrid(params_grid)
    cnt = 0
    for p in grid:
        cnt = cnt+1
    print('Total Possible Models:',cnt)
    return grid

def prophetModelTuning(data,region='All'):
    print('Training model for : ',region)
    best_model = 0
    mape = 100000;best_mae=0
    train,testX = plotTrainTest(data,testpoints,region)
    grid = makeParameterGrid()
    model_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
    for p in grid:
        test = pd.DataFrame()
        #m = Prophet(growth='flat') # for strongly seasonal data
        random.seed(0)
        train_model =Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                         holidays_prior_scale = p['holidays_prior_scale'],
                         n_changepoints = p['n_changepoints'],
                         seasonality_mode = p['seasonality_mode'],
                         changepoint_range=p['changepoint_range'],
                         weekly_seasonality= True,
                         daily_seasonality = False,
                         yearly_seasonality = False,
                         interval_width=0.95)
        train_model.add_country_holidays(country_name='AU')
        train_model.fit(train)
        train_forecast = train_model.make_future_dataframe(periods=360, freq='D',include_history = False)
        train_forecast = train_model.predict(train_forecast)
        testX = data['ds'].tail(testpoints)
        trainTimes = data['ds'].head(len(data) - testpoints)
        testValues = data['y'].tail(testpoints)
        trainValues = data['y'].head(len(data) - testpoints)
        
        mae,MAPE,Actual,predicted = testUsing(train_model,testX,testValues)
        maetrain, MAPEtrain = getTrainAccuracy(train_model,trainTimes,trainValues)
        #Actual = toPlot2[(toPlot2['ds']>=strt) & (toPlot2['ds']<=end)]
        #Actual = testX
       
        if MAPE < mape:
            best_model = train_model
            mape = MAPE
            best_mae = mae
        #if(MAPE < 10 ): #and MAPE >= 7
        #    best_model = train_model
         #   mape = MAPE
          #  best_mae = mae

        model_parameters = model_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
        #print('Mean Absolute Percentage Error(MAPE)-------------------',MAPE)
    predictedActualPlot(train_model,data,predicted)
    parameters = model_parameters.sort_values(by=['MAPE'])
    parameters = parameters.reset_index(drop=True)
    #print(Actual,predicted['yhat'])
    plotExpectedVsActual(Actual,predicted['yhat'],region)
    fig = best_model.plot(predicted)
    plt.title(region)
    plt.ylabel('Proportion of positive cases')
    a = add_changepoints_to_plot(fig.gca(), best_model, predicted)
    return maetrain, MAPEtrain,mae,mape, parameters['Parameters'][0],best_model

def get_moving_average(series, window=3):
    rolling_mean = series.rolling(window=window).mean()
    return rolling_mean
def get_moving_median(series, window=3):
    rolling_median = series.rolling(window=window).median()
    return rolling_median

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def prepareExpectedNames(datafr):
    # prepare expected column names
    datafr.columns = ['ds', 'y']
    datafr['ds']= to_datetime(datafr['ds'])
    return datafr
def getDataSubset(dataSet,strt,end):
    subSet = dataSet[( dataSet['ds']>=strt) & ( dataSet['ds']<=end)]
    nsubSet = subSet.reset_index()
    return subSet
def runSimulation(newData):
    rollingmean = get_moving_average(newData, window)
    newData['y'] = rollingmean['y']
    newData = newData.dropna()
    newData['y']  = round(newData['y'], 3).apply(np.float64)#.apply(np.int64)
    newData.head()
    maetrain, MAPEtrain,mae,mape, best_param,best_model = prophetModelTuning(newData ,region)
    print(f" MAE train:{maetrain},MAPE train: {MAPEtrain},MAE test: {mae},MAPE test: {mape}, Best training hyperparameters:{best_param}")
    return  best_model, MAPEtrain,mape

#@st.cache  # 👈 This function will be cached
def plotRandomWalk(data,region,confidence=0.70):
    fig3 = plt.figure(figsize=(12,3))
    
    # operate smoothing: LOESS (locally estimated scatterplot smoothing) and LOWESS (locally weighted scatterplot smoothing)
    smoother = LowessSmoother(smooth_fraction=0.1, iterations=5)
    smoother.smooth(data)

    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=2) #
    low, up = smoother.get_intervals('prediction_interval',confidence=confidence)
    #plt.subplot(1,numtoplot,i+1)
    plt.plot(smoother.smooth_data[0], linewidth=1, color='tab:blue', alpha=1)
    plt.plot(smoother.data[0], '.k',alpha=0.3)
    tit = region #+ ', Confidence Level = ' + str(confidence)
    plt.title(tit,fontsize=10); plt.xlabel('time (Months)',fontsize=8)
    plt.ylabel('Proportion of positives',fontsize=8)
    
    #flag flareup in case numbers
    for j in range(data.shape[0]):
        if up[0][j] < data[j]:
            plt.scatter(j, data[j],s=100, marker='8',
                   c='red', alpha=1.)
    
    # flag decline in case numbers
    for j in range(data.shape[0]):
        if low[0][j] > data[j]:
            plt.scatter(j, data[j], s=50, marker='*',c='green', alpha=1.)
            

    plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)
    st.write(fig3)

def addMonthsToDate(startDate,window):
    date_format = '%Y-%m-%d'
    dtObj = datetime.strptime(startDate, date_format)
    # Add months to a given datetime object
    future_date = dtObj + relativedelta(months=window)
    # Convert datetime object to string in required format
    future_dt_str = future_date.strftime(date_format)
    return future_dt_str
def compute_n_lag(dset,n=6,threshold = 2,region = 'Test'):
    lags = []
    tau = threshold
    for i in range(n):
        k = i+1
        lag = signalClass(dset,tau,k,region)
        lags.append(lag)
    return lags
def computeEpidemicState(lags,window):
    steady = 0;decline = 0;flareup=0
    epiState=''; confidence_level=0
    #st.write(lags[0][1])
    for i in range(window-1):
        for j in lags[0][0]:
            if j > 0:
                flareup = flareup + 1
            elif j < 0:
                decline = decline + 1
            else:
                steady = steady + 1
    if steady == max(steady,decline,flareup):
        epiState = 'Steady state'
        confidence_level = steady/(steady + decline + flareup)
    elif decline == max(steady,decline,flareup):
        epiState = 'Decline'
        confidence_level = decline/(steady + decline + flareup)
    else:
        epiState = 'Flareup'
        confidence_level = flareup/(steady + decline + flareup)
    return epiState, confidence_level

### UTILITY FUNCTION FOR PLOTTING ###

def plot_history(ax, i, is_anomaly, window_len, color='blue', **pltargs):
    
    posrange = np.arange(0,i)
    
    ax.fill_between(posrange[window_len:], 
                    pltargs['low'][1:], pltargs['up'][1:], 
                    color=color, alpha=0.2)
    if is_anomaly:
        ax.scatter(i-1, pltargs['original'][-1], c='red')
    else:
        ax.scatter(i-1, pltargs['original'][-1], c='black')
    ax.scatter(i-1, pltargs['smooth'][-1], c=color)
    
    ax.plot(posrange, pltargs['original'][1:], '.k')
    ax.plot(posrange[window_len:], 
            pltargs['smooth'][1:], color=color, linewidth=3)
    
    if 'ano_id' in pltargs.keys():
        if pltargs['ano_id'].sum()>0:
            not_zeros = pltargs['ano_id'][pltargs['ano_id']!=0] -1
            ax.scatter(not_zeros, pltargs['original'][1:][not_zeros], 
                       c='red', alpha=1.)
def  realTimeAnomalyDetection():
	fig = plt.figure(figsize=(15,10))
	camera = Camera(fig)

	axes = [plt.subplot(n_series,1,ax+1) for ax in range(n_series)]
	series = defaultdict(partial(np.ndarray, shape=(n_series,1), dtype='float32'))

	n_sigma = 1
	for i in tqdm(range(timesteps+1), total=(timesteps+1)):
		if i>window_len:
			smoother = ConvolutionSmoother(window_len=window_len, window_type='ones')
			smoother.smooth(series['original'][:,-window_len:])

			series['smooth'] = np.hstack([series['smooth'], smoother.smooth_data[:,[-1]]]) 
			_low, _up = smoother.get_intervals('sigma_interval', n_sigma=n_sigma)# : 
			series['low'] = np.hstack([series['low'], _low[:,[-1]]])
			series['up'] = np.hstack([series['up'], _up[:,[-1]]])
			is_anomaly = np.logical_or(
				series['original'][:,-1] > series['up'][:,-1], 
				series['original'][:,-1] < series['low'][:,-1]
				).reshape(-1,1)
			if is_anomaly.any():
				series['ano_id'] = np.hstack([series['ano_id'], is_anomaly*i]).astype(int)

			for s in range(n_series):
				pltargs = {k:v[s,:] for k,v in series.items()}
				plot_history(axes[s], i, is_anomaly[s], window_len,**pltargs)
				axes[s].set_title(healthregions[s] + ', n_sigma = ' + str(n_sigma),fontsize=8)
			camera.snap()

		if i>=timesteps:
			continue

		series['original'] = np.hstack([series['original'], data[:,[i]]])
    
  
	#st.write('CREATING GIF...')  # it may take a few seconds

	camera._photos = [camera._photos[-1]] + camera._photos
	animation = camera.animate()
	animation.save(filename)
	plt.close(fig)
	#st.write('DONE')
#End of Helper Functions________________________________________________________________

#Define the side bar for input parameters before running the models

st.sidebar.write('Parameter Settings:')

highPrevalenceStates = ['AM','PA','AC','AP','RR','RO','MA','MT'] #['Para (PA)']
#topLevelRegion = st.sidebar.selectbox(
#    'Select region(State) to analyse:',
#     highPrevalenceStates)
slide_window = st.sidebar.slider('Choose Sliding Window (Months):', min_value=1, max_value=12, value=6)  # 👈 this is a widget
threshold = st.sidebar.slider('What percentage change do you want to detect:', min_value=0, max_value=100, value=2)  # 👈 this is a widget
confidence = st.sidebar.radio('Confidence level for time series smoothing:',
     [0.5,0.05, 0.25, 0.75,0.95])

st.write('Anomaly will be analysed in ', slide_window, ' months time steps')
st.write('Percentage case change to detect (tau): ', threshold, '%')
st.write('Confidence or tolerance level for regression (tau): ', confidence)



#Load pre-processed State's case data
stateLevel = read_csv('PA.csv', header=0, index_col=0,parse_dates=True, squeeze=True)
hd_s = stateLevel.head(10)

#Load pre-processed heath region's case data in the selected state
perHr = read_csv('PASIVEPDailyPerHr.csv', header=0,parse_dates=True, squeeze=True)



#Process Municipality or Health region data further
healthregions = list(perHr['notification.hr'].unique())
num_of_hr = len(healthregions)
dataset_per_hr = []
for hr in healthregions:
    dsHr = perHr.loc[(perHr[['notification.hr']] == hr).all(axis=1)]
    dsHr =prepareTimeSeries(dsHr)
    dsHr["Date"] = pd.to_datetime(dsHr[['year','month','day']],format='%Y%m%d', errors='coerce')
    dataset_per_hr.append(dsHr)

hd = dataset_per_hr[0].drop(['notification.year', 'notification.month'], axis=1).head(10).head(10)


#Get proportion of positives
dataset_per_hr_new = []
for hr in range(len(dataset_per_hr)):
    neg,pos = getNegativesAndPositives(dataset_per_hr[hr])
    hrNewData= calcMonthlyPrevalence(neg,pos).dropna()
    dataset_per_hr_new.append(hrNewData)
hd1 = dataset_per_hr_new[0].head()


st.write("Visualise proportion of positives from A State and some health regions:")
#Compute Statewide Lag Features:
subdata = returnPrevalenceData(dataset_per_hr_new[0],slide_window)
subdata=prepareExpectedNames(subdata)
stateWide = stateLevel
lag=slide_window
sWide = returnPrevalenceData(stateWide,slide_window)
sWide = prepareExpectedNames(sWide)
lags210,fig2 = signalClass(sWide,threshold,lag,'State Wide (PA)')
st.write(fig2[0])
sWide= sWide.drop(['label'],axis = 1)


numhr = len(dataset_per_hr_new)




### SIMULATE PROCESS REAL-TIME AND CREATE GIF ###
data =np.array([dataset_per_hr_new[0]['monthlyPrevalence'],dataset_per_hr_new[1]['monthlyPrevalence'],dataset_per_hr_new[2]['monthlyPrevalence']])
n_series = data.shape[0]
timesteps = data.shape[1]
window_len = slide_window


st.write('Creating realtime anomaly detection plots for each Municipality...')
filename = 'realtimecasesanalysis.gif' 
realTimeAnomalyDetection()
	




# plot the smoothed timeseries with intervals
numtoplot =13
confidencelevels = [confidence] #,0.25,0.5,0.75,0.95
#confidence=0.5
n_sigma = 1
for conf in confidencelevels:
    plotRandomWalk(sWide['y'],'Para (PA) State',conf)
for confidence in confidencelevels:
    for i in range(numtoplot):
        data = dataset_per_hr_new[i]['monthlyPrevalence']
        plotRandomWalk(data,healthregions[i],conf)

# Data subsets
#Portion of Trainset/test to use for different epidemic states - This could be manually or automatically assigned
steadyStart='2009-01-01';steadyEnd='2012-04-01' # Start and end of steady state timeseries
declineStart='2012-05-01';declineEnd='2015-01-01'
#flareUpStart='2015-02-01';flareUpEnd='2018-10-01'
flareUpStart='2018-03-01';flareUpEnd='2018-10-01'

#General Test set
testStart='2018-11-01';testEnd='2019-12-01'

dataSet = sWide
PACrossValidationTestSet = getDataSubset(dataSet,testStart,testEnd) #Test all models on this
#PACrossValidationTestSet.head()

steadyStateData = getDataSubset(dataSet,steadyStart,steadyEnd)
declineStateData = getDataSubset(dataSet,declineStart,declineEnd)
flareUpStateData = getDataSubset(dataSet,flareUpStart,flareUpEnd)


st.write('Correlation between State Wide and health regional changes') 

warnings.filterwarnings('ignore')
endDates = []
correlationsAll = []
#confidence = 0.05
#threshold = 2 # Percentage tolerance for case increase
totalTestDates = len(flareUpStateData)
totalIntervals = int(totalTestDates/slide_window)
start = flareUpStart
for itv in range(totalIntervals):
    correlationsPerEndDate = []
    future_date_str = addMonthsToDate(start,slide_window)
    endDates.append(future_date_str)
    stateLevelData = getDataSubset(flareUpStateData,start,future_date_str)
    
    
    #Compute lags to determine the epistate of current outbreak
    lags = compute_n_lag(stateLevelData.reset_index(drop=True),slide_window,threshold,'Para (PA) State')
    epiState, confidence_level = computeEpidemicState(lags,slide_window)
    #print(epiState, confidence_level)
    
    region_title = 'Para (PA) State->' + 'Current Epi-state:' + epiState + '(' + str(round(confidence_level*100,2)) + ' % confident)'
    plotRandomWalk(stateLevelData['y'].values,region_title,confidence)
    plt.xlabel(start + ' - ' + future_date_str,fontsize=6)
    
    
    print(start + '---------------------------->' + future_date_str )
    for hr in range(len(dataset_per_hr)):
        hrdata = returnPrevalenceData(dataset_per_hr_new[hr],slide_window)
        hrdata=prepareExpectedNames(hrdata)
        flareUpHrData = getDataSubset(hrdata,start,future_date_str)
        corr1 = round(pd.Series(stateLevelData['y'].values).corr(pd.Series(flareUpHrData['y'].values)),2)
        
        #Compute lags to determine the epistate of current outbreak
        lags = compute_n_lag(flareUpHrData.reset_index(drop=True),slide_window,threshold,healthregions[hr])
        epiState, confidence_level = computeEpidemicState(lags,slide_window)
        region_title = healthregions[hr] + '->Current Epi-state:' + epiState + '(' + str(round(confidence_level*100,2)) + ' % confident)'
        plotRandomWalk(flareUpHrData['y'].values,region_title,confidence)
        #plt.xlabel(start + ' - ' + future_date_str)
        #print(str(healthregions[hr])+ ':' + str(corr1))
        
        correlationsPerEndDate.append(corr1)
    correlationsAll.append(correlationsPerEndDate)
    start = future_date_str


#Generate Real time anomaly detection



file_ = open(filename, "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="Real time anomaly detection">',
    unsafe_allow_html=True,
)


st.write("Sample State Data:")
st.write(hd_s)
st.write("Sample Muncipality data in the State Data:")
st.write(hd)
st.write("Sample Muncipality data from ARAGUAIA:")
st.write(hd1)

numtoplot = numhr # Set the number of health regions to plot out of the total number of health regions in a state
fig, ax = plt.subplots(figsize=(12,6))
for hr in range(numtoplot):
    plt.plot(dataset_per_hr_new[hr]['Date'],dataset_per_hr_new[hr]['monthlyPrevalence'],linewidth=3)
plt.legend(healthregions[:numtoplot])
plt.ylabel('Proportion of positives',fontsize=6)
plt.title('Health regions in  Para(PA) State', fontsize=5)
ymin=0; ymax=1
st.write(fig)
#Train the parent (State level) model