"""
file: Otsu.py
language: python3.7
author: np9603@cs.rit.edu Nihal Surendra Parchand
"""

''' Importing libraries'''
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def Otsu(speedlist,total_speed):
    '''Initialising the values to infinity (sentinal values) '''
    best_mixed_variance = math.inf
    best_threshold = math.inf
    mixed_variance_list = []

    ''' Iterating throught the speeds 40mph to 80mph '''
    for threshold in range(40,80):
        speed_under = []
        speed_over = []
        speed_under_sum = 0
        speed_over_sum = 0
        ''' For every speed in the given data '''
        for speed in speedlist:
            ''' Check if it is below threshold or not '''
            if float(speed) <= threshold:
                speed_under.append(speed)
                speed_under_sum += speed
            else:
                speed_over.append(speed)
                speed_over_sum += speed

        ''' Calculating weights and variance '''
        weight_under = speed_under_sum / total_speed
        variance_under = 0.0 if not speed_under else np.var(speed_under)
        weight_over = speed_over_sum / total_speed
        variance_over = 0.0 if not speed_over else np.var(speed_over)

        ''' Calculating mixed or weighted variance'''
        mixed_variance = weight_under * variance_under + weight_over * variance_over
        mixed_variance_list.append(mixed_variance)

        ''' Finding the best mixed variance and best threshold'''
        if mixed_variance < best_mixed_variance:
            best_mixed_variance = mixed_variance
            best_threshold = threshold


    below_threshold = []
    over_threshold = []

    ''' Appending the speeds according to the best threshold '''
    for speed in speedlist:
        if speed <= best_threshold:
            below_threshold.append(speed)
        else:
            over_threshold.append(speed)

    ''' Plotting the histograms '''
    bins = np.linspace(40,80,40)
    plt.hist(sorted(speedlist), bins=bins , edgecolor = 'black')
    plt.axvline(best_threshold, color = 'blue', linestyle = 'dotted')
    x_axis = [ threshold for threshold in range(40,80)]
    mark = [mixed_variance_list.index(min(mixed_variance_list))]
    plt.plot(x_axis,mixed_variance_list,markevery = mark, marker = "o", markersize=5 )
    plt.hist(below_threshold,bins=bins)
    plt.hist(over_threshold, bins=bins)
    plt.grid(alpha = 0.2)
    plt.xlabel("Speed (mph)")
    plt.ylabel("Mixed variance")
    plt.title("Histogram of Mixed variance vs Speed (mph) without regularization")
    print("Best threshold without regularization: ", best_threshold)
    print("Best mixed variance without regularization: ", best_mixed_variance)
    plt.show()

    ''' Plotting the speeds with respect to the best threshold '''
    plt.hist(below_threshold, bins=bins)
    plt.hist(over_threshold, bins=bins)
    plt.xlabel("Speed (mph)")
    plt.title("Histogram of Speed (mph)")
    plt.grid(alpha=0.2)
    plt.show()


    ''' With Regularization '''

    best_mixed_variance = math.inf
    best_threshold = math.inf
    cost_function_list = []

    for threshold in range(40, 80):
        speed_under = []
        speed_over = []
        speed_under_sum = 0
        speed_over_sum = 0
        for speed in speedlist:
            if float(speed) <= threshold:
                speed_under.append(speed)
                speed_under_sum += speed
            else:
                speed_over.append(speed)
                speed_over_sum += speed
        weight_under = speed_under_sum / total_speed
        variance_under = 0.0 if not speed_under else np.var(speed_under)
        weight_over = speed_over_sum / total_speed
        variance_over = 0.0 if not speed_over else np.var(speed_over)

        mixed_variance = weight_under * variance_under + weight_over * variance_over
        ''' Initializing the alpha values in a list '''
        alpha = [100,1,0.2,0.1,0.05,0.04,0.02,0.01,0.001]
        for value in alpha:
            ''' Calculating regularization '''
            regularization = abs((len(speed_under) - len(speed_over))) / (50 * value)
            ''' Calculating the cost function '''
            cost_function = mixed_variance + regularization

            if cost_function < best_mixed_variance:
                best_mixed_variance = cost_function
                best_threshold = threshold
        ''' Appending the best cost function '''
        cost_function_list.append(best_mixed_variance)

    below_threshold = []
    over_threshold = []

    for speed in speedlist:
        if speed <= best_threshold:
            below_threshold.append(speed)
        else:
            over_threshold.append(speed)

    bins = np.linspace(40, 80, 40)
    plt.hist(sorted(speedlist), bins=bins, edgecolor='black')
    plt.axvline(best_threshold, color='blue', linestyle='dotted')
    x_axis = [threshold for threshold in range(40, 80)]
    mark = [cost_function_list.index(min(cost_function_list))]
    plt.plot(x_axis, mixed_variance_list, markevery=mark, marker="o", markersize=5 )
    plt.hist(below_threshold, bins=bins)
    plt.hist(over_threshold, bins=bins)
    plt.grid(alpha = 0.2)
    plt.xlabel("Speed (mph)")
    plt.ylabel("Mixed variance")
    plt.title("Histogram of Mixed variance vs Speed (mph) with regularization")
    plt.show()
    print("Best threshold with regularization: ", best_threshold)
    print("Best mixed variance with regularization: ", best_mixed_variance)


def main():
    ''' Declaring a speed list to store the different speed values '''
    speedlist = []
    total_speed = 0
    ''' Reading the data file line by line '''
    with open('DATA_v2191_FOR_CLUSTERING_using_Otsu.csv','r') as file:
        for line in file:
            ''' Removing leading and trailing spaces '''
            line = line.strip()
            ''' Using regex to extract numeric values from the line '''
            for word in re.findall(r'(\d+\.\d*|\d+)', line):
                ''' Appending the values in the speed list '''
                speedlist.append(math.floor(float(word)))
                total_speed = total_speed + float(word)

    Otsu(speedlist,total_speed)
main()