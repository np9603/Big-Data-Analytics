"""
file: HW07_Parchand_Nihal_kMeans.py
language: python3.7
author: np9603@cs.rit.edu Nihal Surendra Parchand
date: 11/10/2019
"""

''' Importing libraries'''

import math
import numpy as np
import pandas as pd
import time # For calculating time in seconds
from scipy.spatial import distance # For calculating mahalanobis distance
import random # To generate a random number
import copy # For creating a deep copy
import matplotlib.pyplot as plt # For plotting graphs

def covariance(data):
    """
    This function is used to calculate the inverse covariance of the given data
    :param data: Original k-means data
    :return: Inverse covariance matrix
    """

    data_covariance = np.cov(data.T)
    inverse_covariance = np.linalg.inv(data_covariance)

    return inverse_covariance

def select_random_centroids(k_clusters,data):
    """
    This function is used to calculate the initial centers by randomly choosing a number from 0 to 1303 and selecting
    one of the data points as cluster centers
    :param k_clusters: Number of clusters
    :param data: Original k-means data
    :return: random_centroid_list which stores the initial randomly selected centroids
    random_number_list which stores the random numbers selected using random.randint method
    """

    ''' Initializing two lists '''
    random_number_list = []
    random_centroid_list = []

    ''' Generating a list of k random numbers '''
    for random_number in range(0,k_clusters):
        random_number_list.append(random.randint(0,1303))

    ''' Generating a list of k random centroids '''
    for number in random_number_list:
        random_centroid_list.append(data[number])

    return random_centroid_list,random_number_list

def calculate_mahalanobis_distance(point,centroid,inverse_covariance):
    """
    This function is used to calculate the mahalanobis distance of a data point from a cluster centroid by using the
    inverse covariance matrix. If the mahalanobis distance is more than 3, ignore the point and return infinite. Else
    it returns the calculated distance.
    :param point: One of the data point in the data
    :param centroid: Centroid point
    :param inverse_covariance: Inverse covariance matrix
    :return: calculated mahalanobis_distance
    """

    ''' This is a builtin function from the scipy module for calculating mahalanobis distance between two points.
     It uses the two points and the inverse covariance matrix.'''
    mahalanobis_distance = distance.mahalanobis(point,centroid,inverse_covariance)

    ''' Ignore points that have mahalanobis distance greater than 3 '''
    if mahalanobis_distance > 3:
        return math.inf

    return mahalanobis_distance


def calculate_new_centroids(points):
    """
    This function is used to calculate the new centroids by taking in the old centroids values and calculating the
    center of mass.
    :param points: All the values that belong to a particular centroid
    :return: new_centroid is the new calculated center that will be used in the next iterations
    """

    ''' Initializing the x,y,z coordinates for new centroid '''
    sum_of_x = 0
    sum_of_y = 0
    sum_of_z = 0

    ''' Iterating through all the points for calculating new centroid '''
    for point in points:
        sum_of_x += point[0]
        sum_of_y += point[1]
        sum_of_z += point[2]

    ''' Rounding off the centroid to the nearest one significant digits '''
    centroid_x = round((sum_of_x / len(points)),1)
    centroid_y = round((sum_of_y / len(points)),1)
    centroid_z = round((sum_of_z / len(points)),1)

    ''' Storing the new centroid points in a list '''
    new_centroid = [centroid_x,centroid_y,centroid_z]

    return new_centroid


def main():

    ''' Reading the input data for k-Means '''

    original_data = pd.read_csv("HW_KMEANS_DATA_v810.csv",header=None)

    ''' Converting dataframe to numpy array '''

    original_data_numpy_array = original_data.values

    ''' Calculating the inverse covariance matrix '''
    inverse_covariance = covariance(original_data_numpy_array)

    # centroid_dict = {key:[] for key in range(0,k)}
    # # print(np.array_equal(np.array([1, 2]), [1,2,3]))
    # previous_centroid = np.array([[0,0,0]]*k)
    ''' Initializing the lists '''
    best_sse_list = []
    total_time_list = []

    ''' Iterating through 2 to 12 clusters '''
    for k_clusters in range(11,13):
        print("CALCULATING FOR",k_clusters,"CLUSTERS")
        ''' Starting the timer to calculate the total time taken for each cluster '''
        start_time = time.time()

        ''' Initializing the lists '''
        centroids_list = []
        sse_list = []


        ''' For every k clusters, we perform k-means for 500 iterations to find the best centroid and SSE '''
        for iteration in range(0,500):
            ''' Randomly assign centroids for initial calculation '''
            random_centroid_list, random_number_list = select_random_centroids(k_clusters, original_data_numpy_array)
            ''' Creating a deep copy makes two different copies of the randomly selected centroid '''
            current_centroids = copy.deepcopy(random_centroid_list)
            # new_centroids_list = np.array([[0,0,0]]*k)
            # Add a breakpoint according to the value of k cluster to avoid
            ''' Initializing the value of flag to True so that we can loop till we get same cluster centroids for
            consecutive steps of that iteration '''
            flag = True
            loop_iterations = 0
            while flag and loop_iterations < 50:
                loop_iterations += 1
                ''' Initializing a dictionary to store the centroids as keys and their corresponding assigned cluster points '''
                mahalanobis_distance_dict = {}
                ''' Iterating through the data row by row '''
                for row in range(0,len(original_data_numpy_array)):
                    ''' Storing the current data point in a variable for better readability '''
                    data_point = original_data_numpy_array[row]
                    ''' Initializing a list to store the mahalanobis distance of a point to the k different centroids '''
                    mahalanobis_distance_list = []
                    ''' Iterating through the centroid list to find mahalanobis distances and store it in the list '''
                    for centroid_index in range(0,len(current_centroids)):
                        mahalanobis_distance_list.append(calculate_mahalanobis_distance(data_point,current_centroids[centroid_index],inverse_covariance))

                    ''' Find the minimum mahalanobis distance '''
                    min_mahal_index = mahalanobis_distance_list.index(min(mahalanobis_distance_list))
                    ''' Converting the current data point from a numpy array to a list '''
                    data = np.ndarray.tolist(data_point)
                    ''' Assigning cluster centroid by using the minimum mahalanobis distance '''
                    belong_to_cluster = current_centroids[min_mahal_index]
                    ''' Converting centroid to a string for storing it as key in dictionary '''
                    cluster_key = np.array2string(belong_to_cluster)

                    ''' Appending cluster centroid and its corresponding cluster value as key-value pair to the dictionary '''
                    if cluster_key not in mahalanobis_distance_dict:
                        mahalanobis_distance_dict[cluster_key] = []
                    mahalanobis_distance_dict[cluster_key].append(data)

                ''' Finding new cluster centers '''
                new_centroid_list = []
                for key in mahalanobis_distance_dict.keys():
                    new_centroid = calculate_new_centroids(mahalanobis_distance_dict[key])
                    new_centroid_list.append(np.array(new_centroid))

                ''' Checking if the current centroid is the same as newly calculated centroids '''
                if len(current_centroids) == len(new_centroid_list):
                    flag = not np.allclose(current_centroids,new_centroid_list)

                if len(current_centroids) != len(new_centroid_list):
                    print(current_centroids,new_centroid_list)
                ''' If not same then assign the new calculated centroids as current centroids and iterate again until
                they are same'''
                if flag:
                    current_centroids = new_centroid_list


            ''' Appending centroids to a list for further calculations '''
            centroids_list.append(current_centroids)
            # if iteration < 20:
            #     print("================================================================")
            #     print("New centroids after iteration",iteration,": ",str(new_centroid_list))
            #     print("================================================================")
            if iteration > 20:
                print("Calculating for other iterations...",iteration)


            ''' Calculating SSE '''
            index = 0
            total_sum_of_squared_errors = 0
            ''' Iterating through all the values of dictionary to calculate the SSE '''
            for values in mahalanobis_distance_dict.values():
                sum_of_squared_errors = 0
                for points in values:
                    sum_of_squared_errors += math.pow((points[0]-new_centroid_list[index][0]),2) \
                                     + math.pow((points[1]-new_centroid_list[index][1]),2) \
                                     + math.pow((points[2]-new_centroid_list[index][2]),2)
                ''' Incrementing index to calculate for the next centroid '''
                index += 1
                ''' Adding the SSE of each centroid '''
                total_sum_of_squared_errors += sum_of_squared_errors
            ''' Appending the total SSE to a list '''
            sse_list.append(total_sum_of_squared_errors)

        ''' Finding out the best SSE for k clusters '''
        best_sse = min(sse_list)
        best_sse_list.append(best_sse)

        ''' To find out the total number of data points in each cluster for best centroids for that particular value of k '''
        best_centroids_index = sse_list.index(min(sse_list))
        best_centroids = centroids_list[best_centroids_index]
        best_centroids_dict = {}
        for row in range(0, len(original_data_numpy_array)):
            data_point = original_data_numpy_array[row]
            mahalanobis_distance_list = []
            for centroid_index in range(0, len(best_centroids)):
                mahalanobis_distance_list.append(
                    calculate_mahalanobis_distance(data_point, best_centroids[centroid_index], inverse_covariance))

            min_mahal_index = mahalanobis_distance_list.index(min(mahalanobis_distance_list))
            data = np.ndarray.tolist(data_point)
            belong_to_cluster = best_centroids[min_mahal_index]

            cluster_key = np.array2string(belong_to_cluster)

            if cluster_key not in best_centroids_dict:
                best_centroids_dict[cluster_key] = []
            best_centroids_dict[cluster_key].append(data)

        ''' Finding the center of mass for each k cluster '''
        print("=====================================================================================================================================\n")
        center_of_mass = []
        for key,values in best_centroids_dict.items():
            print("The number of data points for centroid ",key,"=",len(values))
            center_of_mass.append(key)

        print("The center of mass =",center_of_mass)

        ''' Calculating the total time taken for each cluster '''

        end_time = time.time()
        total_time = end_time - start_time
        total_time_list.append(total_time)

        print("The best sse for", k_clusters, "clusters is:",round(best_sse,2))
        print("Total time to complete 500 iterations for",k_clusters, "clusters =",round(total_time,2),"seconds")
        print("\n=====================================================================================================================================")



    ''' Initializing the cluster list for plotting '''

    cluster_list = [2,3,4,5,6,7,8,9,10,11,12]
    

    ''' Plotting the time taken for each cluster vs number of clusters '''
    plt.title("Time taken to run (seconds) vs Total number of clusters")
    plt.xlabel('Total number of clusters')
    plt.xlim(2, 12)
    plt.ylabel('Total time taken (seconds)')
    plt.grid()
    plt.plot(cluster_list, total_time_list)
    plt.show()

    ''' Plotting the time taken for each cluster vs SSE for that cluster '''
    plt.title("SSE vs Total number of clusters ")
    plt.xlabel('Total number of clusters')
    plt.xlim(2, 12)
    plt.ylabel('SSE')
    plt.grid()
    plt.plot(cluster_list, best_sse_list)
    plt.show()

main()