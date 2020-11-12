"""
file: OneDimClassifier.py
language: python3.7
author: np9603@cs.rit.edu Nihal Surendra Parchand
date: 09/20/2019
"""

''' Importing libraries'''

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def quantize(original_data):
    """
    This function is used for quantizing the data to the nearest 0.5 mph
    :param original_data: Dataframe with original data
    """
    ''' Initialising list for storing quantized speeds '''
    quantized_data_list = []

    ''' Initialising dataframe for storing quantized speeds and their corresponding Aggressive value
     from original data '''
    quantized_data_df = pd.DataFrame()

    ''' Iterating through the original data and quantizing the speeds to their nearest 0.5mph 
    values and appending it to the quantized data list'''
    for speed in original_data['Speed']:
        speed = round(speed/0.5)*0.5
        quantized_data_list.append(speed)

    ''' Storing the calculated quantized data in a dataframe '''
    quantized_data_df["Speed"] = quantized_data_list
    quantized_data_df["Aggressive"] = original_data["Aggressive"]

    ''' Passing the quantized dataframe and calling the oneDimClassifier method '''
    print("-------- Cost function = number_of_missed_speeders + number_of_false_alarms --------\n")
    oneDimClassifier(quantized_data_df)

    print("\n-------- Cost function = number_of_missed_speeders + (2 * "
          "number_of_false_alarms) --------\n")
    oneDimClassifier2(quantized_data_df)
    print("--------------------------------------------------------------------------")


def oneDimClassifier(quantized_data_df):
    """
    This function is used for classification using normal cost function
    :param quantized_data_df: The dataframe with quantized speeds
    """

    ''' Initialising the values to infinity (sentinal values) '''

    best_threshold = math.inf
    best_number_wrong = math.inf
    best_false_alarm_rate = math.inf
    best_true_positive_rate = math.inf
    false_alarm_rate_list = []
    true_positive_rate_list = []
    cost_function_list = []


    ''' Iterating through the speeds 45mph to 80mph (Iterate till 80.5 because it is exclusive) '''
    for threshold in np.arange(45,80.5,0.5):
        ''' Initialising the values to calculate the missed speeders and false alarms '''
        number_of_missed_speeders = 0
        number_of_false_alarms = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        ''' Iterating through the quantized speedlist '''
        for index in range(len(quantized_data_df)):
            ''' Storing each speed and aggressive value in variables '''
            speed = quantized_data_df.loc[index,"Speed"]
            aggressive = quantized_data_df.loc[index,"Aggressive"]
            ''' Calculating FN,FP,TN,TP for every speed in speedlist '''
            if speed <= threshold and aggressive == 1:
                number_of_missed_speeders += 1
                false_negative += 1
            elif speed <= threshold and aggressive == 0:
                true_negative += 1
            elif speed > threshold and aggressive == 0:
                number_of_false_alarms += 1
                false_positive += 1
            elif speed > threshold and aggressive == 1:
                true_positive += 1

        ''' Number of wrong/Cost function '''
        number_of_wrong = number_of_missed_speeders + number_of_false_alarms

        ''' Appending cost function to a list '''
        cost_function_list.append(number_of_wrong)

        ''' Finding the best cost function, best threshold, best FAR and best TPR'''
        if number_of_wrong <= best_number_wrong:
            best_number_wrong = number_of_wrong
            best_threshold = threshold
            best_false_alarm_rate = false_positive / (false_positive + true_negative)
            best_true_positive_rate = true_positive / (true_positive + false_negative)


        ''' Calculating False alarm rate and True Positive Rate '''
        false_alarm_rate = false_positive / (false_positive + true_negative)
        true_positive_rate = true_positive / (true_positive + false_negative)

        ''' Appending the FAR and TPR rates to a list '''
        false_alarm_rate_list.append(false_alarm_rate)
        true_positive_rate_list.append(true_positive_rate)

    ''' Printing out the results '''
    print("Best number wrong/cost function  = " + str(best_number_wrong))
    print("Best threshold  = " + str(best_threshold) + " mph")
    print("Best false alarm rate = " + str(best_false_alarm_rate))
    print("Best true positive rate = " + str(best_true_positive_rate))

    ''' Plotting cost function as a function of threshold '''
    threshold_list = [threshold for threshold in np.arange(45.0,80.5,0.5)]
    plt.axvline(best_threshold,linestyle = 'dotted',color='g')
    ''' Marking the minimum cost function for best threshold '''
    mark = [cost_function_list.index(min(cost_function_list))]
    plt.annotate(s=str(best_threshold) + " mph, " + str(min(cost_function_list)),
                 xy=(best_threshold, min(cost_function_list)),
                 xytext=(best_threshold, min(cost_function_list) + 5))
    plt.plot(threshold_list,cost_function_list,markevery = mark, marker = "o", markersize=5,
             mfc = 'r')
    plt.xlim([45, 80])
    plt.xlabel("Threshold values (Speed in mph)")
    plt.ylabel("Cost function")
    plt.title("Threshold vs Cost function")
    plt.show()


    ''' Plotting the ROC Curve '''

    ''' Marking random points on ROC curve '''
    markers = [x for x in range(0,len(false_alarm_rate_list),5) ]
    plt.plot(best_false_alarm_rate, best_true_positive_rate, marker='o', markersize=7, color="red")
    plt.annotate(s=str(best_threshold)+ " mph",xy=(best_false_alarm_rate, best_true_positive_rate),
                 xytext=(best_false_alarm_rate , best_true_positive_rate + 0.05))
    plt.plot(false_alarm_rate_list,true_positive_rate_list, lw = 1.5,markevery=markers,marker =
    'o',markersize=5,clip_on=False)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--' )
    plt.xlabel("False Alarm Rate (False Positive Rate)")
    plt.ylabel("Correct Hit Rate (True Positive Rate)")
    plt.title("ROC curve by Threshold")
    plt.show()

    ''' To calculate the missed/incorrect cases '''
    number_of_missed_speeders = 0
    number_of_false_alarms = 0
    for index in range(len(quantized_data_df)):
        speed = quantized_data_df.loc[index, "Speed"]
        aggressive = quantized_data_df.loc[index, "Aggressive"]
        if speed <= best_threshold and aggressive == 1:
            number_of_missed_speeders += 1
        if speed > best_threshold and aggressive == 0:
            number_of_false_alarms += 1


    print("Number of aggressive speeders who did not get caught = " + str(number_of_missed_speeders))
    print("Number of non-reckless drivers who got pulled over = " + str(
        number_of_false_alarms))


def oneDimClassifier2(quantized_data_df):
    ''' Initialising the values to infinity (sentinal values) '''

    best_threshold = math.inf
    best_number_wrong = math.inf
    best_false_alarm_rate = math.inf
    best_true_positive_rate = math.inf
    false_alarm_rate_list = []
    true_positive_rate_list = []
    modified_cost_function_list = []

    ''' Iterating through the speeds 45mph to 80mph (Iterate till 80.5 because it is exclusive) '''
    for threshold in np.arange(45, 80.5, 0.5):
        ''' Initialising the values to calculate the missed speeders and false alarms '''
        number_of_missed_speeders = 0
        number_of_false_alarms = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        ''' Iterating through the quantized speedlist '''
        for index in range(len(quantized_data_df)):
            ''' Storing each speed and aggressive value in variables '''
            speed = quantized_data_df.loc[index,"Speed"]
            aggressive = quantized_data_df.loc[index,"Aggressive"]
            ''' Calculating FN,FP,TN,TP for every speed in speedlist '''
            if speed <= threshold and aggressive == 1:
                number_of_missed_speeders += 1
                false_negative += 1
            elif speed <= threshold and aggressive == 0:
                true_negative += 1
            elif speed > threshold and aggressive == 0:
                number_of_false_alarms += 1
                false_positive += 1
            elif speed > threshold and aggressive == 1:
                true_positive += 1

        ''' Number of wrong/Cost function (Modified cost function with regularization '''
        number_of_wrong = number_of_missed_speeders + (2 * number_of_false_alarms)

        ''' Appending cost function to a list '''
        modified_cost_function_list.append(number_of_wrong)

        ''' Finding the best cost function, best threshold, best FAR and best TPR'''
        if number_of_wrong <= best_number_wrong:
            best_number_wrong = number_of_wrong
            best_threshold = threshold
            best_false_alarm_rate = false_positive / (false_positive + true_negative)
            best_true_positive_rate = true_positive / (true_positive + false_negative)


        ''' Calculating False alarm rate and True Positive Rate '''
        false_alarm_rate = false_positive / (false_positive + true_negative)
        true_positive_rate = true_positive / (true_positive + false_negative)

        ''' Appending the FAR and TPR rates to a list '''
        false_alarm_rate_list.append(false_alarm_rate)
        true_positive_rate_list.append(true_positive_rate)

    ''' Printing out the results '''
    print("Best number wrong/cost function with double false alarms = " + str(best_number_wrong))
    print("Best threshold with double false alarms = " + str(best_threshold) + " mph")
    print("Best false alarm rate = " + str(best_false_alarm_rate))
    print("Best true positive rate = " + str(best_true_positive_rate))


    ''' Plotting cost function as a function of threshold '''
    threshold_list = [threshold for threshold in np.arange(45.0, 80.5, 0.5)]
    plt.axvline(best_threshold, linestyle='dotted', color='g')
    mark = [modified_cost_function_list.index(min(modified_cost_function_list))]
    plt.annotate(s=str(best_threshold) + " mph, " + str(min(modified_cost_function_list)),
                 xy=(best_threshold, min(modified_cost_function_list)),
                 xytext=(best_threshold, min(modified_cost_function_list) + 5))

    plt.plot(threshold_list, modified_cost_function_list, markevery=mark, marker="o", markersize=5,
             mfc='r')
    plt.xlim([45, 80])
    plt.xlabel("Threshold values (Speed in mph)")
    plt.ylabel("Cost function")
    plt.title("Threshold vs Modified Cost function ")
    plt.show()

    ''' Plotting the ROC Curve '''

    ''' Marking random points on ROC curve '''

    markers = [x for x in range(0,len(false_alarm_rate_list),5) ]

    plt.plot(best_false_alarm_rate, best_true_positive_rate, marker='o', markersize=7, color="red")
    plt.annotate(s=str(best_threshold)+ " mph" ,xy=(best_false_alarm_rate,
                                                     best_true_positive_rate),
                 xytext=(best_false_alarm_rate , best_true_positive_rate + 0.05))
    plt.plot(false_alarm_rate_list,true_positive_rate_list, lw = 1.5,markevery=markers,marker =
    'o',markersize=5,clip_on=False)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--' )
    plt.xlabel("False Alarm Rate (False Positive Rate)")
    plt.ylabel("Correct Hit Rate (True Positive Rate)")
    plt.title("ROC curve by Threshold (with double false alarms)")
    plt.show()

    ''' To calculate the missed/incorrect cases '''
    number_of_missed_speeders = 0
    number_of_false_alarms = 0
    for index in range(len(quantized_data_df)):
        speed = quantized_data_df.loc[index, "Speed"]
        aggressive = quantized_data_df.loc[index, "Aggressive"]
        if speed <= best_threshold and aggressive == 1:
            number_of_missed_speeders += 1
        if speed > best_threshold and aggressive == 0:
            number_of_false_alarms += 1


    print("Number of aggressive speeders who did not get caught = " + str(number_of_missed_speeders))
    print("Number of non-reckless drivers who got pulled over = " + str(
        number_of_false_alarms))

def main():

    ''' Reading the data and storing it in a dataframe '''
    original_data = pd.read_csv("DATA_v2191_FOR_CLASSIFICATION_using_Threshold.csv")
    quantize(original_data)

main()