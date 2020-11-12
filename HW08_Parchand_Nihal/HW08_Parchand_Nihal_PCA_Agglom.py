"""
file: HW08_Kunjilikattil_Rohit_Parchand_Nihal_PCA_Agglom.py
language: python3.7
author: rk4447@cs.rit.edu Rohit Kunjilikattil
author: np9603@cs.rit.edu Nihal Surendra Parchand
date: 09/20/2019
"""

''' Importing libraries'''

import pandas as pd
import numpy as np
import math
import heapq
import plotly
from numpy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.cluster import KMeans
import plotly.figure_factory as ff
import plotly.graph_objects as go
import operator

def PCA(original_data):
    """
    This function is used to calculate the Principal Component Analysis of the data by using
    cross-correlation matrix to calculate the eigen vectors that explains the data and then projecting
    the original data onto these vectors
    :param original_data: Original data on which agglomeration is performed (Guest records)
    :return: Final projected matrix
    """

    ''' Deleting the ID column '''
    del original_data[original_data.columns[0]]

    "Calculate cross correlation coefficient of all attributes"
    print("Cross-correlation matrix")
    print()
    cross_correlation_matrix = original_data.corr().abs()
    ''' Printing the cross-correlation matrix in the range [-1,1] '''
    print(np.round(original_data.corr(),2).to_string())
    print()

    "Calculate two most strongly cross-related attributes"

    ''' The following code considers all attributes and sorts it according to its most correlated attributes. 
    sorted_correlation_pairs is a series of the most correlated attributes in descending order '''
    correlation_pairs = cross_correlation_matrix.unstack()
    correlation_pairs = correlation_pairs.replace([1.000], 0)
    sorted_correlation_pairs = correlation_pairs.sort_values(kind='quicksort' , ascending=False )

    ''' Initializing the maximum correlation to worst '''
    max_correlation = 0

    ''' Iterating through the correlation pairs to find the most correlated attribute pair '''
    for attribute_tuple,corr_coef in sorted_correlation_pairs.iteritems():
        if corr_coef > max_correlation:
            max_correlation = corr_coef
            first_attribute = attribute_tuple[0].strip()
            second_attribute = attribute_tuple[1].strip()

    print("The two attributes that are most strongly cross-correlated with each other:",first_attribute,"and",second_attribute,"->",max_correlation)

    "Calculate the most strongly cross-related attribute to Fish"

    for attribute_tuple,corr_coef in sorted_correlation_pairs.iteritems():
        if attribute_tuple[0] == "  Fish" or attribute_tuple[1] == "  Fish":
            print("Which attribute is fish most strongly cross-correlated with? ",attribute_tuple[0],"and", attribute_tuple[1],"->",corr_coef)
            break


    "Calculate the most strongly cross-related attribute to Meat"

    for attribute_tuple,corr_coef in sorted_correlation_pairs.iteritems():
        if attribute_tuple[0] == "  Meat" or attribute_tuple[1] == "  Meat":
            print("Which attribute is meat most strongly cross-correlated with? ",attribute_tuple[0],"and", attribute_tuple[1],"->",corr_coef)
            break

    "Calculate the most strongly cross-related attribute to Beans"

    for attribute_tuple,corr_coef in sorted_correlation_pairs.iteritems():
        if attribute_tuple[0]==" Beans" or attribute_tuple[1]==" Beans":
            print("Which attribute is beans most strongly cross-correlated with? ",attribute_tuple[0],"and", attribute_tuple[1],"->",corr_coef)
            break

    "Calculate the least and the second least cross-related attribute to every other attribute"

    column_name_list = cross_correlation_matrix.columns.to_list()
    lowest_sum = math.inf
    col_sum = []
    ''' The following code is used to calculate the sum of correlation of entire column in cross-correlation matrix.
    We subtract 1 from the total sum to take care of the diagonal elements '''
    for c in column_name_list:
        sum_of_column_corr = cross_correlation_matrix[c].sum()-1
        col_sum.append(sum_of_column_corr)
        if sum_of_column_corr < lowest_sum:
            lowest_sum = sum_of_column_corr

    ''' Calculate the first and second minimum cross correlated attributes '''
    min_index = col_sum.index(min(col_sum))
    second_min_index = col_sum.index(heapq.nsmallest(2,col_sum)[-1])

    print("The least cross-correlated attribute is ", column_name_list[min_index])

    print("The second least cross-correlated attribute is ", column_name_list[second_min_index])

    "Calculate the covariance matrix of all the attributes"

    centred_df = original_data.values - original_data.values.mean()
    covariance_matrix = np.cov(centred_df.transpose())

    ''' Calculating the eigen values and eigen vectors using the covariance matrix '''
    eigen_values, eigen_vectors = eig(covariance_matrix)

    print()
    print("------------EIGEN VALUES----------------")
    eigen_values[::-1].sort()
    print(eigen_values)
    print()
    print("------------EIGEN VECTORS----------------")
    for i in range(5):
        print("Eigenvector " + str(i+1))
        print(np.round(eigen_vectors[i],decimals=1))
        print()

    ''' Storing first 3 eigen vectors in eigen pairs list'''
    first_3_eig_vectors = [(eigen_vectors[i]) for i in range(3)]
    first_3_eig_vectors = np.asarray(first_3_eig_vectors)

    ''' Taking the transpose of the eigen vectors. [3x20] -> [20x3] '''
    new_projection = first_3_eig_vectors.transpose()

    ''' Taking the dot product of the center points with the transpose of first 3 eigen vectors to get 850x3 matrix '''
    new_matrix = centred_df.dot(new_projection)

    "3D plot of projected points"
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    x_vals = new_matrix[:, 0:1]
    y_vals = new_matrix[:, 1:2]
    z_vals = new_matrix[:, 2:3]
    ax.scatter(x_vals,y_vals,z_vals, marker='o')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.show()


    ''' K-means clustering '''

    fig_2 = plt.figure(figsize=(4,4))
    ax_2 = fig_2.add_subplot(111, projection='3d')
    k_means_clustering = KMeans(n_clusters=6, random_state=0).fit_predict(new_matrix)

    k_means_ = KMeans(n_clusters=6, random_state=0).fit(new_matrix)
    print("The k-means cluster centers are: ",k_means_.cluster_centers_)
    print()
    ax_2.scatter(new_matrix[:,0],new_matrix[:,1],new_matrix[:,2],c=k_means_clustering,marker='o',s=15)
    ax_2.set_xlabel('PCA 1')
    ax_2.set_ylabel('PCA 2')
    ax_2.set_zlabel('PCA 3')
    plt.show()

    return new_matrix


def calc_center_of_mass(cluster_1,cluster_2):
    """
    This function is used to calculate the center of mass. It is calculated by taking the sum of all values of each
    cluster and dividing it by the total number of points in that cluster.
    :param cluster_1: Cluster 1
    :param cluster_2: Cluster 2
    :return: center of mass for cluster 1 and cluster 2.
    """

    sum_1 = 0
    sum_2 = 0

    for values in cluster_1:
        sum_1 += values

    for values in cluster_2:
        sum_2 += values

    center_of_mass_1 = sum_1/len(cluster_1)
    center_of_mass_2 = sum_2/len(cluster_2)

    return center_of_mass_1,center_of_mass_2


def calculate_centre(data):
    """
    This function is used to calculate the center of each cluster passed to it.
    :param data:
    :return:
    """
    length = len(data)
    sum=0
    for v in data:
        sum+=v

    return np.round(sum/length,2)

def calc_euclidean_distance(center_1, center_2):
    """
    This function calculates the euclidean distance between two centers.
    :param center_1: First center point
    :param center_2: Second center point
    :return: Euclidean distance between two points
    """

    center_1 = center_1[0]
    center_2 = center_2[0]
    return math.sqrt(math.pow(center_1 - center_2, 2))

def agglomerative_clustering(cluster,data,projected_matrix):
    """
    This function is used to perform agglomerative clustering on the data.
    :param cluster: Number of clusters for agglomeration
    :param data: Original data
    :param projected_matrix: The projected matrix returned from PCA.
    """


    data_length = len(data)
    ''' Initialize a cluster dictionary to store the initial clusters and their ids '''
    cluster_dictionary = {}
    data_after_PCA = projected_matrix

    ''' Initializing each record as its own cluster '''
    for index in range(0,data_length):
        cluster_dictionary[index] = data_after_PCA[index]

    ''' Initializing dictionaries to store information required for further analysis '''
    merge_dictionary = {}
    centre_dictionary = {}

    ''' This list initializes the 850 records to its own prototype '''
    final_cluster_label = []
    for i in range(0,data_length):
        final_cluster_label.append(i)
    print("Initial cluster labels")
    print(final_cluster_label)

    ''' For each point in dictionary we calculate the center and store it in a dictionary '''
    for key, value in cluster_dictionary.items():
        centre = calculate_centre(value)
        centre_dictionary[key] = [centre]

    ''' This appends the initial size as 1 for every record '''
    for iterator in range(data_length):
            centre_dictionary[iterator].append(1)

    ''' By performing agglomerative clustering, we will loop till we get one cluster'''
    while(len(centre_dictionary)!=cluster):
        ''' Initializing the minimum distance as infinity '''
        best_min_distance = math.inf

        ''' Calculating euclidean distance between every point except itself '''
        for key_1 in centre_dictionary.keys():
            for key_2 in centre_dictionary.keys():
                if key_1!=key_2:
                    dist = calc_euclidean_distance(centre_dictionary[key_1],centre_dictionary[key_2])
                    ''' Comparing current distance to best distance and updating the best distance and best centers '''
                    if dist < best_min_distance:
                        best_min_distance = dist
                        best_centres = [key_1,key_2]

        # print(best_centres)
        ''' This stores the keys as a tuple of two centers and the minimum size of both centers '''
        merge_dictionary[(best_centres[0],best_centres[1])]=min((centre_dictionary[best_centres[0]][1],centre_dictionary[best_centres[1]][1]))

        ''' The following code calculates the new center of mass and replace the previous center of mass with the newly calculated
         center of mass.
         The second line adds the size of both centers.'''
        centre_dictionary[best_centres[0]][0] = (centre_dictionary[best_centres[0]][0]+centre_dictionary[best_centres[1]][0])/2
        centre_dictionary[best_centres[0]][1] = centre_dictionary[best_centres[0]][1]+centre_dictionary[best_centres[1]][1]
        ''' Popping the bigger center from the dictionary '''
        centre_dictionary.pop(best_centres[1])

        ''' Finding the smaller and bigger center '''
        minimum_label = min(best_centres[0],best_centres[1])
        max_label = max(best_centres[0],best_centres[1])

        ''' Assigning the new cluster based on the minimum of the two values. '''
        for idx, item in enumerate(final_cluster_label):
            if item == max_label:
                final_cluster_label[idx] = minimum_label

    ''' This maintains a list of all smaller cluster sizes during each merge '''
    smaller_cluster_size_list=[]
    for value in merge_dictionary.values():
        smaller_cluster_size_list.append(value)

    print()
    print("Smaller cluster size list")
    print(smaller_cluster_size_list)
    print()
    print("Last 20 merges")
    for x in list(reversed(list(merge_dictionary)))[0:20]:
        print(merge_dictionary[x])
    print(final_cluster_label)

    ''' Printing out the 6 clusters and their cluster sizes '''
    final_cluster_label_set = set(final_cluster_label)
    unique_final_cluster_label_list = list(final_cluster_label_set)

    count_dict = {}
    for value in unique_final_cluster_label_list:
        count_dict[value] = final_cluster_label.count(value)

    sorted_count_list = sorted(count_dict.items(), key=operator.itemgetter(1))

    for key,value in sorted_count_list:
        print("Size of cluster",key,"is",value)

    ''' Plotting the dendogram based on original data '''
    names = [x for x in range(0,850)]
    fig = ff.create_dendrogram(data, labels=names)
    fig['layout'].update({'width': 1400, 'height': 600})
    fig.show()

def main():

    with open("HW_PCA_SHOPPING_CART_v892.csv",'r') as file:
        original_data = pd.read_csv(file,index_col=False)

    projected_matrix = PCA(original_data)
    original_data_values = original_data.values
    agglomerative_clustering(1,original_data_values,projected_matrix)
    agglomerative_clustering(6,original_data_values,projected_matrix)


main()


