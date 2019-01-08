# -*- coding: utf-8 -*-
'''
Created on Tue Sep 18th 

@author: Adarsh V
'''

#Import libraries
import numpy
import pandas as pd
import math
import operator
from scipy.special import beta
#Start by defining constants

#Pc is the underlying true probability. We start with a prior for PcHyp
#As per recommendation in the paper, we use Jeffry's prior
PC_HYP = {"alpha" : 0.5, "beta" : 0.5}

#Set value for the first hyper parameter. This hyper parameter is used in Stage 1 to decide if there is a change
#in the underlying probability. We set values as recommended in the paper
KLCRIT = 0.23

#Set value for the second hyper parameter. This is used in the second stage to perform Bayesian comparision
#of two models. 
BFCRIT = 1

#Pg is the true probability of green balls in the urn. PgHyp is the hyperparameter that sets the prior 
#for this probability. Using Bayes-Laplace prior as recommended in the paper
PG_HYP = {"alpha" : 1, "beta" : 1}

#Set number of training subjects and sessions
TRAIN_SUBJECTS = list(range(1, 2))
TRAIN_SESSIONS = list(range(1, 2)) #Assume the number of train subjects is the same for all subjects. Has to be modified otherwise

def betaFunction(alpha_val, beta_val):
    '''Function to calcualate beta function using the given alpha and beta values
    Simple use of mathematical formula
    '''
    return alpha_val/(alpha_val + beta_val)
    
def calculateObservedProbability(numGreens, numTotal) :
    '''
    Function that calculates observed probability, which is the number of green balls
    divided by the total number of balls. Very simple calculation
    '''
    assert (numTotal != 0), "Total probability is 0, probability becomes meaningless"
    if (numGreens == 0) :
        return 0
    p = numGreens/numTotal
    assert (p >= 0 and p <= 1), "Nonsensical probability"
    return p
    
def calculateKLDistance(p1, p2) :
    '''
    Function to calculate Kullbeck-Liebler distance b/w 2 probabilities
    Input :
        p1 : first probability
        p2 : second probability
    Output : the KL distance
    Throws an exception in there is a math error. 
    '''
    if (p1 == p2) :
        return 0
    if (p2 == 0 or p2 == 1) :
        return math.inf
    q1 = 1 - p1
    q2 = 1- p2
    if (p1 == 0) :
        return -1 * math.log(q2)
    if (p1 == 1) :
        return -1 * math.log(p2)
   
    try :
        return (p1 * math.log(p1/p2)) + (q1 * math.log(q1/q2))
    except Exception :
         print('Error', p1, p2, q1, q2)
         
def calculatePriorOdds(num_trials, p_c):
    '''
    Calculate priod odds using the formula given from the reference paper
    Input :
        num_trials : the number of trials in the current sequence
        pc : the current estimated value for the probability of a change in the tru probability
    Output : the prior
    '''
    return (num_trials * p_c)/ (1 - p_c)

def findChangePointInSeriesByLog(first_p_g, cum_greens, cum_reds, currPg, p_c) :
    print('cum_greens', cum_greens, 'cum_reds', cum_reds, 'first_p_g', first_p_g, 'currPg', currPg, 'p_c',p_c)
    maximizingPGAfter = 0
    maximizingIndex = 0
    logLikelihoods = []
    for i in range(len(cum_greens)) :
        n_g_lt = cum_greens[i]
        n_r_lt = cum_reds[i]
        n_g_gt = cum_greens[-1] - cum_greens[i]
        n_r_gt = cum_reds[-1] - cum_reds[i]
        firstTerm = math.log(beta(n_g_lt + currPg.get("alpha"),  n_r_lt + currPg.get("beta")))
        secondTerm = math.log(beta(n_g_gt + PG_HYP.get("alpha"),  n_r_gt + PG_HYP.get("beta")))
        logLikFun = firstTerm + secondTerm
        logLikelihoods.append(logLikFun)
    logLikelihoods = list(map(lambda likelihood: likelihood + abs(max(logLikelihoods)), logLikelihoods))
    logLikelihoods = list(map(lambda likelihood: math.exp(likelihood), logLikelihoods))
    
    priorOdds = calculatePriorOdds(len(cum_reds), p_c)
    relLikelihood = sum(logLikelihoods)/len(cum_reds)/logLikelihoods[-1]
    
    cPMean = 0
    for i in range(len(cum_greens)):
        cPMean = cPMean + ((i+1) * logLikelihoods[i])
    cPMean = (cPMean / sum(logLikelihoods))
    odds = priorOdds * relLikelihood
    id =round(cPMean) - 1
    print('id', id)
    n_g_gt = cum_greens[-1] - cum_greens[id]
    n_r_gt = cum_reds[-1] - cum_reds[id]
    nextPg = {"alpha" : n_g_gt + PG_HYP.get("alpha"), "beta" : n_r_gt + PG_HYP.get("beta")}
    print('relLikelihood', relLikelihood, 'logLikelihoods', logLikelihoods, 'cPMean', cPMean, 'odds', odds)
    return {'changePointIndex' : id, 'nextPg' : nextPg, 'maxRelLik' : odds}
        
def calculatePostOdds(cum_reds, cum_greens, pc ):
    priorOdds = calculatePriorOdds(len(cum_reds), pc)
    maximalMarginalLikelihood1 = calculateMMLForOneChgMdl(cum_reds, cum_greens)
    maximalMarginalLikelihood2 = calculateMMLForNoChgMdl(cum_reds, cum_greens)
    return priorOdds * maximalMarginalLikelihood1 / maximalMarginalLikelihood2

def recalculateSequence(observedSeq, penultimateSeq):
    cum_greens = list(numpy.cumsum(observedSeq))
    penultimate_cum_greens = list(numpy.cumsum(penultimateSeq))
    totals = list(range(1, len(cum_greens) + 1))
    cum_reds = list(map(operator.sub, totals, cum_greens))
    penultimate_totals = list(range(1, len(penultimate_cum_greens) + 1))
    penultimate_cum_reds = list(map(operator.sub, penultimate_totals, penultimate_cum_greens))
    return cum_greens, cum_reds, penultimate_cum_greens, penultimate_cum_reds

def countGeensAndReds(observedSeq):
    return observedSeq.count(1), observedSeq.count(0)

def implementChangePointAlgorithm(stimulus, num_changes, num_trials):
    '''
    The main algorithm goes here. Use the observed stimulus data and the ChangePoint algorithm
    from the paper to calculate Bayesian probabilities. The program is commented extensively
    Input : 
        stimulus : A list of stimulus data for a specific subject and session. 1 for a green ball and 0 for red.
        num_changes : the number of change points detected for the given subject across all sessions
        num_trials : the total number of trials detected for the given subject across all sessions
    Output : A dictionary. Details coming soon :)
    '''
    #A variable to store the sequence of observations since the penultimate change point
    observedSeq = []
    #A variable to store the penultimate sequence. This will be reqd if we eventually determine that the current
    #CP is wrong
    penultimateSeq = []
    #Observerd probability
    num_greens = 0
    num_reds = 0
    cum_greens = []      #A vector storing cumulative successes after each trial in the sequence
    penultimate_cum_greens = []  #A vector storing cumulative successes after each trial in the penultimate sequence
    cum_reds = []          #A vector storing cumulative failures (reds) after each trial in the sequence
    penultimate_cum_reds = []  #A vector storing cumulative failures after each trial in the penultimate sequence
    pg = betaFunction(PG_HYP.get("alpha"), PG_HYP.get("beta"))   #calculate prior probability of pg, our estimate of reqd probability
    
    pgList = []
    changePointCount = 0
    
    currPg =  {"alpha" : PG_HYP.get("alpha"), "beta" : PG_HYP.get("beta")}
    prevPg = {"alpha" : PG_HYP.get("alpha"), "beta" : PG_HYP.get("beta")}
    currPcParams =  {"alpha" : PC_HYP.get("alpha"), "beta" : PC_HYP.get("beta")}
    pc = betaFunction(PC_HYP.get("alpha"), PC_HYP.get("beta"))
    #iterate through each trial
    detectedChangeList = []
    detectedChange = 0
    for trial in stimulus :
        print('_________________________________________________________________')
        num_trials = num_trials + 1 #increase total num trials
        pgList.append(pg)
        observedSeq.append(trial)
        if (trial == 1) :
            num_greens = num_greens + 1
        else :
            num_reds = num_reds + 1
        print(num_trials, 'pg', pg, 'pc', pc, 'obs', observedSeq, 'penul',penultimateSeq,'detectedChange',detectedChange)
        cum_greens.append(num_greens)
        cum_reds.append(num_reds)
        #STEP 1 : Calculate observed probability
        p_obs = calculateObservedProbability(num_greens, len(observedSeq))
        
        #STEP 2 : Calculate KL divergence b/w current estimate and observed probability
        dist = calculateKLDistance(p_obs, pg)
        
        #STEP 3 : calculate evidence of something being wrong
        evidence = dist * len(observedSeq)
        
        print('evidence', evidence)
        
        #STEP 4 : Stage 1 test - compare evidence with stage 1 threshold
        if (evidence > KLCRIT and detectedChange == 0) :
            #Recalculate PC since we will use it in our calculation of odds ratio
            currPcParams.update({"alpha" : num_changes + PC_HYP.get("alpha"), "beta" : num_trials + PC_HYP.get("beta")- num_changes })
            print('currPcParams', currPcParams, 'num_trials',num_trials, 'num_changes', num_changes )
            pc = betaFunction(currPcParams.get("alpha"), currPcParams.get("beta"))
            
            print('Some change in probability estimates', 'evidence', evidence,'ALSO updated pc',pc, num_trials)
            #There is significant difference b/w observed and estimated probabilities. Proceed to next stage
            #STEP 5 - posterior odds for Bayesian comparision
            changepointDict = findChangePointInSeriesByLog(pg, cum_greens, cum_reds, currPg, pc)
            posteriorOdds = changepointDict.get("maxRelLik")
            #STEP 6 - compare the likelihood of there being a change with second decision criteria
            if (posteriorOdds > BFCRIT) :
                num_changes = num_changes + 1
                print('New CP required at ', changepointDict.get("changePointIndex"), 'posteriorOdds', posteriorOdds, num_trials)
                #STEP 7 : use the maximing pg from the new change point as the new pg
                prevPg = {"alpha" : currPg.get("alpha"), "beta" : currPg.get("beta")}
                currPg = changepointDict.get('nextPg')
                print('currPg', currPg)
                pg = betaFunction(currPg.get("alpha"), currPg.get("beta"))
                
                changePointCount = changePointCount+ 1
                #STEP 8 : Use the new CP, reinitialize penultimate and current sequences
                penultimateSeq = observedSeq[0:changepointDict.get("changePointIndex") + 1]
                observedSeq = observedSeq[changepointDict.get("changePointIndex") + 1:]
                num_greens, num_reds = countGeensAndReds(observedSeq)
                cum_greens, cum_reds, penultimate_cum_greens, penultimate_cum_reds = recalculateSequence(observedSeq, penultimateSeq)
                #Mark detection point
                detectedChange = 1
            elif ( changePointCount > 0 and posteriorOdds < BFCRIT):
                #STEP 9 : Temperorily remove the previous changepoint and identify a new CP in the combined sequence
                #with penultimate and current sequence
                print('Not a new CP. Investigating further', num_trials)
                combined_observed_seq = penultimateSeq + observedSeq
               
                combined_cum_greens = list(numpy.cumsum(combined_observed_seq))
                comb_totals = list(range(1, len(combined_cum_greens) + 1))
                combined_cum_reds = list(map(operator.sub, comb_totals, combined_cum_greens))
                
                changepointDict = findChangePointInSeriesByLog(pg, combined_cum_greens, combined_cum_reds, PG_HYP, pc)
                posteriorOdds = changepointDict.get("maxRelLik")
                print('posteriorOdds after temp removing CP', posteriorOdds, num_trials)
                if (posteriorOdds > BFCRIT) :
                    print('Correcting CP added previously', num_trials)
                    #STEP  : use the maximing pg from the new change point as the new pg
                    prevPg = {"alpha" : currPg.get("alpha"), "beta" : currPg.get("beta")}
                    currPg = changepointDict.get('nextPg')
                    pg = betaFunction(currPg.get("alpha"), currPg.get("beta"))
                    #if (changepointDict.get("changePointIndex")  > 0) :
                    changePointCount = changePointCount+ 1
                    #STEP  : Use the new CP, reinitialize penultimate and current sequences
                    penultimateSeq = combined_observed_seq[0:changepointDict.get("changePointIndex") + 1]
                    observedSeq = combined_observed_seq[changepointDict.get("changePointIndex") + 1:]
                    num_greens, num_reds = countGeensAndReds(observedSeq)
                    cum_greens, cum_reds, penultimate_cum_greens, penultimate_cum_reds = recalculateSequence(observedSeq, penultimateSeq)
                    #Mark detection point
                    detectedChange = 1
                else :
                    print("Remove previous CP")
                    num_changes = num_changes - 1
                    currPcParams.update({"alpha" : num_changes + PC_HYP.get("alpha"), "beta" : num_trials + PC_HYP.get("beta")- num_changes })
                    observedSeq = combined_observed_seq[:]
                    penultimateSeq = []
                    num_greens, num_reds = countGeensAndReds(observedSeq)
                    cum_greens, cum_reds, penultimate_cum_greens, penultimate_cum_reds = recalculateSequence(observedSeq, penultimateSeq)
                    #Mark detection point
                    detectedChange = 1
                    
                    currPg = {"alpha" : prevPg.get("alpha") + num_greens, "beta" : prevPg.get("beta") + num_reds}
                    print("prevPg", prevPg, "currPg", currPg, 'num_greens', num_greens, "num_reds", num_reds)
                    pg = betaFunction(currPg.get("alpha"), currPg.get("beta"))
                    
            else :
                print('No penultimate seq, calculating probabiity directly ')
                currPg.update({"alpha" : cum_greens[-1] + PG_HYP.get("alpha"), "beta" : PG_HYP.get("beta") + cum_reds[-1]})
                pg = betaFunction(currPg.get("alpha"), currPg.get("beta"))
                changePointCount = changePointCount+ 1
                num_greens, num_reds = countGeensAndReds(observedSeq) 
                cum_greens = list(numpy.cumsum(observedSeq))  # returns a numpy.ndarray
                totals = list(range(1, len(cum_greens) + 1))
                cum_reds = list(map(operator.sub, totals, cum_greens))
        else :
            detectedChange = 0
        detectedChangeList.append(detectedChange)
    return {'pgList' : pgList, 'num_changes' : num_changes, 'num_trials' : num_trials, 'detectedChangeList' : detectedChangeList}
           


#Define a list of all column names. Stimulus is the column we are most concerned with
cNames = ("expCode", "subId", "ssnNo", "stimulus", "response", "chKey", "secTht", "reacTime", "modelP", "objChPt", "trueChP", "stepHt", "stepWidth")
#Read the input file
stimulusData = pd.read_table("G:\\Capstone\\Working\\MatlabData.txt", sep = '\t', header = None, names = cNames)    
    

estimatedProbList = []
DPList = []
#Extract stimulus data for the required subject
for sbj in TRAIN_SUBJECTS:
    #Initiate the value of NC - the number of changepoints, and N - the number of trials
    nc = PC_HYP.get("alpha")
    n = 0
    pc = betaFunction(PC_HYP.get("alpha"), PC_HYP.get("beta"))   #calculate prior probability of pc
    
    for ssn in TRAIN_SESSIONS :
        stimulus = stimulusData.loc[((stimulusData['subId'] == sbj) &  (stimulusData['ssnNo'] == ssn)), ('stimulus')]
        changePointData = implementChangePointAlgorithm(stimulus, nc, n)
        estimatedProbList.extend(changePointData.get('pgList'))
        DPList.extend(changePointData.get('detectedChangeList'))
        n = changePointData.get('num_trials')
        nc = changePointData.get('num_changes')

cName = ('matlabModel')
matlabModelData = pd.read_table("G:\\Capstone\\Working\\matlabCPModelProbabilities.txt", sep = '\t', header = None, names = cName) 
changePoingDf = pd.DataFrame({'stimulus' : stimulusData.loc[((stimulusData['subId'] == 1) &  (stimulusData['ssnNo'] == 1)), ('stimulus')] , 
'subject' : stimulusData.loc[((stimulusData['subId'] == 1) &  (stimulusData['ssnNo'] == 1)), ('response')],
'Estimated_P_Gs' : estimatedProbList, 'matlabModel' :  matlabModelData['m'], 'DPs' : DPList})
changePoingDf.to_csv('G:\\Capstone\\Working\\results\\changePointData.csv', sep = ',')
ax = changePoingDf[:501].plot.line(y='subject', use_index=True, label = 'Subject', figsize=(14,12))
changePoingDf[:501].plot.line(y = 'Estimated_P_Gs', use_index=True, label = 'Changepoint Model', ax= ax)   
changePoingDf[:501].plot.line(y = 'matlabModel', use_index=True, label = 'Matlab Changepoint Model', ax= ax) 

ax2 = changePoingDf[501:].plot.line(y='subject', use_index=True, label = 'Subject', figsize=(14,12))
changePoingDf[501:].plot.line(y = 'Estimated_P_Gs', use_index=True, label = 'Changepoint Model', ax= ax2)   
changePoingDf[501:].plot.line(y = 'matlabModel', use_index=True, label = 'Matlab Changepoint Model', ax= ax2) 





