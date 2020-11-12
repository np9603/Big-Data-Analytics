"""
file: mysterydata.py
language: python3.7
author: np9603@cs.rit.edu Nihal Surendra Parchand
"""


import re
import numpy as np
from scipy import stats


def main():
    ''' Declaring a speed list to store the different speed values '''
    speedlist = []


    ''' Reading the data file line by line '''
    with open('Mystery_Data.csv','r') as file:
        for line in file:
            ''' Removing leading and trailing spaces '''
            line = line.strip()
            ''' Using regex to extract numeric values from the line '''
            for word in re.findall(r'(\d+\.\d*|\d+)', line):
                ''' Appending the values in the speed list '''
                speedlist.append(float(word))

    print("Original Data: ")
    median = np.median(speedlist)
    mean = np.mean(speedlist)
    mode = stats.mode(speedlist)
    midrange = (max(speedlist) + min(speedlist)) / 2
    average = np.average(speedlist)
    standard_deviation = np.std(speedlist, dtype=np.float64)
    print("Median: ", median)
    print("Mean: ", mean)
    print("Mode: ",mode)
    print("Midrange: ",midrange)
    print("Average: ",average)
    print("Standard Deviation: ",standard_deviation)
    print("-------------------------------------------------------------------")
    print("Removing last value of data")
    speedlist2 = speedlist[:len(speedlist)-1]

    median = np.median(speedlist2)
    mean = np.mean(speedlist2)
    mode = stats.mode(speedlist2)
    midrange = max(speedlist2) - min(speedlist2) / 2
    average = np.average(speedlist2)
    standard_deviation = np.std(speedlist2, dtype=np.float64)
    print("Median: ", median)
    print("Mean: ", mean)
    print("Mode: ", mode)
    print("Midrange: ", midrange)
    print("Average: ", average)
    print("Standard Deviation: ", standard_deviation)
    print("-------------------------------------------------------------------")
    print("Removing last 16 values of data")
    speedlist3 = speedlist[:len(speedlist) - 16]

    median = np.median(speedlist3)
    mean = np.mean(speedlist3)
    mode = stats.mode(speedlist3)
    midrange = max(speedlist3) - min(speedlist3) / 2
    average = np.average(speedlist3)
    standard_deviation = np.std(speedlist3, dtype=np.float64)
    print("Median: ", median)
    print("Mean: ", mean)
    print("Mode: ", mode)
    print("Midrange: ", midrange)
    print("Average: ", average)
    print("Standard Deviation: ", standard_deviation)
main()