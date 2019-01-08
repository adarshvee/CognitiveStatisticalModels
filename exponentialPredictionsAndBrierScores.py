# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:39:04 2018

@author: Adarsh V
"""

# Install packages

! pip install pandas
! pip install numpy

import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="white", color_codes=True)
import math
import matplotlib.pyplot as plt

from random import seed
from random import random

seed(123)

TRAIN_SESSIONS = list(range(1, 11))

#Read stimulus and response 
colNames = ("subject", "session", "stimulus", "response", "percChng", "secondTht", "trueP", "trueChngInd")
stimAndRes = pd.read_table("G:\\Capstone\\Working\\StimulusAndResponse.txt", sep = '\t', header = None, names = colNames)

cNames = ("expCode", "subId", "ssnNo", "stimulus", "response", "chKey", "secTht", "reacTime", "modelP", "objChPt", "trueChP", "stepHt", "stepWidth")
matlabData = pd.read_table("G:\\Capstone\\Working\\MatlabData.txt", sep = '\t', header = None, names = cNames)

cName = ("matlabModel")
matlabModelData = pd.read_table("G:\\Capstone\\Working\\matlabCPSubj1.txt", sep = '\t', header = None, names = cName) 


NUMTRAININGRECORDS = 10000
DISABLE_RANDOM_SLIDE = False
"""
Function to calculate Brier's score for a single prediction. Takes in the forecast and truth as inputs
"""
def findBriersForRow(forecast, truth):
    return ((truth * (1-forecast)) ** 2 ) + (((1-truth) * forecast) ** 2)
    
    
def randomSlideMove(curr, next, noChangeCount):
    if (DISABLE_RANDOM_SLIDE):
        return next
    rd = random()
    lengthFactor = math.sqrt((noChangeCount/250) ** 2 + ((next - curr)/100) **2)
    shouldChange = rd * lengthFactor
    #shouldChange = 1 - math.exp(-math.sqrt((noChangeCount/1000) ** 2 + (next - curr) **2)  * rd * 100 /40)
    #print('lengthFactor', lengthFactor, 'rd', rd, 'shouldCh', shouldChange)
    if (rd < 0.74 and rd > 0.7) :
        return next
    return next if  lengthFactor > 0.4  else curr
    
"""
Function to calculate Logarithmic's score for a single prediction. Takes in the forecast and truth as inputs
"""
def findLogarithmicScoreForRow(forecast, truth):
    if (forecast == 0) :
        forecast = 0.00001
    return (-1 * truth * math.log(forecast)) - ((1 - truth) * math.log(1-forecast)) 
    

def expMovAvgCalculator(stimulusList, maxI, memory, initial) :
    forecasts = []
    prevSession = 0  #Initiate prevSession for base case of recursion
    for i in range(maxI):
        #Check if it is a new session
        if (prevSession == 0 or prevSession != stimulusList.iloc[i, 1]): 
            prevSession = prevSession+1
            forecasts.append(initial)
        else:
            value = (1-memory) * forecasts[i-1] + memory * stimulusList.iloc[i-1, 2]
            forecasts.append(value)
    return forecasts

forecasts1 = expMovAvgCalculator(stimAndRes, 10000, 1/7, 1/2)
forecasts2 = expMovAvgCalculator(stimAndRes, 10000, 1/10, 1/2)
forecasts3 = expMovAvgCalculator(stimAndRes, 10000, 1/13, 1/2)

wt1, wt2, wt3 = 1, 1, 1

class Experts :
    def __init__(self, num, memory, iniWt, currWt, forecasts, weights):
        self.num = num
        self.memory = memory
        self.iniWt = iniWt
        self.currWt = currWt
        self.forecasts = forecasts
        self.weights = weights

expert1 = Experts(1, 1/7, 1, 1, forecasts1, [])
expert2 = Experts(2, 1/10, 1, 1, forecasts2, [])
expert3 = Experts(3, 1/13, 1, 1, forecasts3, [])

experts = [expert1, expert2, expert3]

def getExpertBasedTerm(weight, prob, isGreen):
    prob = (1 - prob) if isGreen else prob
    result = weight * math.exp(-2 * (prob ** 2))
    return result

#Calculate Gn
brierForecast = []
randomizedSliderMove = []
sList = g0List = g1List = []
noChangeCount = 0
for i in range(10):
    for j in range(1000):
        g0 = g1 = 0
        for exp in experts :
            if (j == 0):
                exp.currWt = 1
            exp.weights.append(exp.currWt)
            g0 += getExpertBasedTerm(exp.currWt, exp.forecasts[i*1000 + j], 0)
            g1 += getExpertBasedTerm(exp.currWt, exp.forecasts[i*1000 + j], 1)
        g0 = -1 * math.log(g0)
        g1 = -1 * math.log(g1)
        g0List.append(g0)
        g1List.append(g1)
    
        #Add condition here
        #Both g0 and g1 are positive
        
        s = (g0 + g1 + 2) / 2
        if (g0 == 0 and g1 == 0):
            s = 2
        #print(g0, g1, s)
        sList.append(s)
        pr1 = (s - g1)/2 if (s - g1)/2 > 0 else 0
        if (j > 0):
            noChangeCount = noChangeCount + 1
            randomizedMovement = randomSlideMove(randomizedSliderMove[-1], pr1, noChangeCount)
        else :
            randomizedMovement = pr1
        #If there is no slider movement, increase the count and guess another seq length
        if (randomizedMovement == pr1) :
            noChangeCount = 0
        brierForecast.append(pr1)
        randomizedSliderMove.append(randomizedMovement)
        #print(pr1, randomizedMovement)
        totalWt = 0
        for exp in experts :
            #exp.currWt * math.exp(-2 * ((1-stimAndRes.iloc[(i*10 + j)-1, 2]) ** 2))
            exp.currWt = getExpertBasedTerm(exp.currWt, exp.forecasts[i*1000 + j], stimAndRes.iloc[(i*1000 + j), 2])
            totalWt += exp.currWt
        for exp in experts :
            exp.currWt = (exp.currWt * 3) / totalWt


g_s_df = pd.DataFrame({"g0" : g0List, "g1" : g1List, "s" : sList})
g_s_df.to_csv('G:\\Capstone\\Working\\results\\gs_DataFrame.csv', sep = ',')

final_df = pd.DataFrame({"Forecast1" : expert1.forecasts, "Forecast2" : expert2.forecasts, "Forecast3" : expert3.forecasts, 
            "brierExpForecast" : brierForecast, "Exp1 Wt" : expert1.weights, 
            "Exp2 Wt" : expert2.weights, "Exp3 Wt" : expert3.weights, 
            "randomizedSliderMove" : randomizedSliderMove})


final_df['truth'] = stimAndRes['trueP'][0:10001]
final_df['bayesianModelP'] = matlabData['modelP'][0:10001]
final_df['sessionNum'] = matlabData['ssnNo'][0:10001]
final_df['response'] = matlabData['response'][0:10001]
final_df['stimulus'] = matlabData['stimulus'][0:10001]
final_df['matlabCPModel'] = matlabModelData['m']

#bayesBrier = expertBrier = []
bayesBrier = []
expertsCombinedBrier = []
expert1Brier = []
expert2Brier =[]
expert3Brier = []
subjectBrier = []
matlabCPBrier = []
randomizedSliderBrier = []

for index, row in final_df.iterrows():
    bayesBrier.append(findBriersForRow(row['bayesianModelP'], row['stimulus']))
    expertsCombinedBrier.append(findBriersForRow(row['brierExpForecast'], row['stimulus']))
    expert1Brier.append(findBriersForRow(row['Forecast1'], row['stimulus']))
    expert2Brier.append(findBriersForRow(row['Forecast2'], row['stimulus']))
    expert3Brier.append(findBriersForRow(row['Forecast3'], row['stimulus']))
    subjectBrier.append(findBriersForRow(row['response'], row['stimulus']))
    matlabCPBrier.append(findBriersForRow(row['matlabCPModel'], row['stimulus']))
    randomizedSliderBrier.append(findBriersForRow(row['randomizedSliderMove'], row['stimulus']))
    
final_df["brierBayesModel"] = bayesBrier
final_df["brierCombinedExpertPred"] = expertsCombinedBrier
final_df["expert1Brier"] = expert1Brier
final_df["expert2Brier"] = expert2Brier
final_df["expert3Brier"] = expert3Brier
final_df["subjectBrier"] = subjectBrier
final_df["matlabCPBrier"] = matlabCPBrier
final_df["randomizedSliderBrier"] = randomizedSliderBrier

print('Brier error for Bayes model', final_df['brierBayesModel'].sum()/10000)
print('Brier error for combined experts', final_df['brierCombinedExpertPred'].sum()/10000)
print('Brier error for 1st expert', final_df['expert1Brier'].sum()/10000)
print('Brier error for 2nd expert', final_df['expert2Brier'].sum()/10000)
print('Brier error for 3rd expert', final_df['expert3Brier'].sum()/10000)
print('Brier error for Subject', final_df['subjectBrier'].sum()/10000)
print('Brier error for GKLML model', final_df['matlabCPBrier'].sum()/10000)
print('Brier error for degraded Brier smoothening algorithm', final_df['randomizedSliderBrier'].sum()/10000)

print('Variance of Brier error for Bayes model', final_df['brierBayesModel'].var())
print('Variance of Brier error for combined experts', final_df['brierCombinedExpertPred'].var())
print('Variance of Brier error for 1st expert', final_df['expert1Brier'].var())
print('Variance of Brier error for 2nd expert', final_df['expert2Brier'].var())
print('Variance of Brier error for 3rd expert', final_df['expert3Brier'].var())
print('Variance of Brier error for Subject', final_df['subjectBrier'].var())
print('Variance of Brier error for Matlab CP model', final_df['matlabCPBrier'].var())
print('Variance of Brier error for randomized slider move', final_df['randomizedSliderBrier'].var())


grouped_df = final_df.groupby(['sessionNum'])['subjectBrier',  'expert1Brier', 'expert2Brier', 
                    'expert3Brier', 'brierCombinedExpertPred', 'brierBayesModel', 'matlabCPBrier', 'randomizedSliderBrier'].agg('sum')/10000
grouped_df_for_report = grouped_df
grouped_df_for_report = grouped_df_for_report.append(grouped_df_for_report.sum(numeric_only=True), ignore_index=True)
grouped_df_for_report = grouped_df_for_report.sort(['subjectBrier'])
grouped_df_for_report.loc['Total']= grouped_df_for_report.sum()
grouped_df_for_report.T.to_csv('G:\\Capstone\\Working\\results\\grouped_df.csv', sep = ',')

ax =  grouped_df.plot.line( y='brierBayesModel', use_index=True, label = 'Bayes Model', figsize=(12,11))
grouped_df.plot.line(y = 'brierCombinedExpertPred', use_index=True, label = 'Combined Expert prediction', ax= ax)

grouped_df.plot.line(y = 'expert1Brier', use_index=True, label = 'Expert 1/7', ax= ax)
grouped_df.plot.line(y = 'expert2Brier', use_index=True, label = 'Expert 1/10', ax= ax)
grouped_df.plot.line(y = 'expert3Brier', use_index=True, label = 'Expert 1/13', ax= ax)

grouped_df.plot.line(y = 'subjectBrier', use_index=True, label = 'Subject', ax= ax)
grouped_df.plot.line(y = 'matlabCPBrier', use_index=True, label = 'Matlab CP', ax= ax)


ax.set_xlabel("Sessions")
ax.set_ylabel("Brier score")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()


sess_wise_dfs = np.array_split(final_df[:], 10)

#Plot session wise change points
for i in range(len(sess_wise_dfs)) :
    currDf = sess_wise_dfs[i].copy(deep = False)
    axis =  currDf.plot.line( y='randomizedSliderMove', use_index=False, label = 'Degraded Brier smoothening algorithm' + str(i + 1), figsize=(12,11))
    currDf.plot.line(y = 'response', use_index=False, label = 'Subject slider move', ax= axis)
    axis.set_xlabel("Trial #")
    axis.set_ylabel("Estimated probability of getting a green ring in the next trial")
    #sess_wise_dfs[i].plot.line(y = 'matlabCPModel', use_index=True, label = 'Changepoint model slider move', ax= axis)
    #sess_wise_dfs[i].plot.line(y = 'truth', use_index=True, label = 'Changepoint model slider move', ax= axis)
    #sess_wise_dfs[i].plot.line(y = 'truth', use_index=True, label = 'Changepoint model slider move', ax= axis)


sliderWidths = []
subjectSliderWidth = []
sliderHeigths = []
subjectSliderHeights = []
cpSliderHeights = []
cpSliderWidths = []

for i in range(len(sess_wise_dfs)) :
    noChangeCount = 0
    noSubChangeCount = 0
    noCPChangeCount = 0
    sess_wise_dfs[i]['brierSliderHeight'] = sess_wise_dfs[i]['randomizedSliderMove'].diff()
    sess_wise_dfs[i]['subjectSliderHeight'] = sess_wise_dfs[i]['response'].diff()
    for j in range(len(sess_wise_dfs[i])) :
        curr = sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j], 'randomizedSliderMove']
        if (j != 0 ) :
            prev = sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j-1], 'randomizedSliderMove']
        else :
            prev = sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j], 'randomizedSliderMove']
        if (prev != curr) :
            sliderHeigths.append(sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j], 'randomizedSliderMove'] - sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j-1], 'randomizedSliderMove'])
            
            sliderWidths.append(noChangeCount)
            noChangeCount = 0
        else :
            noChangeCount = noChangeCount+ 1
        
    
        currSub = sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j], 'response']
        if (j != 0 ) :
            prevSub = sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j-1], 'response']
        else :
            prevSub = sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j], 'response']
        if (prevSub != currSub) :
            subjectSliderWidth.append(noSubChangeCount)
            subjectSliderHeights.append(sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j], 'response'] - sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j-1], 'response'])
            noSubChangeCount = 0
        else :
            noSubChangeCount = noSubChangeCount+ 1
            
        currCP = sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j], 'matlabCPModel']
        if (j != 0 ) :
            prevCP = sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j-1], 'matlabCPModel']
        else :
            prevCP = sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j], 'matlabCPModel']
        if (prevCP != currCP) :
            cpSliderWidths.append(noCPChangeCount)
            cpSliderHeights.append(sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j], 'matlabCPModel'] - sess_wise_dfs[i].loc[sess_wise_dfs[i].index[j-1], 'matlabCPModel'])
            noCPChangeCount = 0
        else :
            noCPChangeCount = noCPChangeCount+ 1    
        
        
merged_df = pd.concat(sess_wise_dfs)
a = (sns.jointplot(subjectSliderWidth, subjectSliderHeights).set_axis_labels("Width", "Height"))
a.fig.suptitle("Subject's joint distribution")
a.fig.subplots_adjust(top=.9)

b = (sns.jointplot(sliderWidths, sliderHeigths).set_axis_labels("Width", "Height"))
b.fig.suptitle("Degraded brier smoothening algorithm's joint distribution")
b.fig.subplots_adjust(top=.9)

c = (sns.jointplot(cpSliderWidths, cpSliderHeights).set_axis_labels("Width", "Height"))
c.fig.suptitle("Changepoint algorithm's algorithm's joint distribution")
c.fig.subplots_adjust(top=.9)


final_df.to_csv('G:\\Capstone\\Working\\results\\final_df.csv', sep = ',')
#identify best experts
def identifyBestNExperts(n):
    from heapq import nsmallest
    brierErrors = []
    for i in range(1, 200, 1) :
        currForecastList = expMovAvgCalculator(stimAndRes, NUMTRAININGRECORDS, 1/i, 1/2)
        currentError = 0
        for i in range(len(currForecastList)):
            currentError += findBriersForRow(currForecastList[i], matlabData.iloc[i]['stimulus'])
        brierErrors.add(currentError)
    nsmallest(n, enumerate(brierErrors), key = lambda x : x[1])
    # for 3, [(9, 1526.0590516217076), (8, 1526.3411758743737), (10, 1527.411997766588)]




