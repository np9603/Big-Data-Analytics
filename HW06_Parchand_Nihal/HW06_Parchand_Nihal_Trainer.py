"""
file: HW06_Parchand_Nihal_Trainer.py
language: python3.7
author: np9603@cs.rit.edu Nihal Surendra Parchand
date: 11/2/2019
"""

''' Importing libraries'''

import math
import numpy as np
import pandas as pd


def find_entropy(less_than_threshold,more_than_threshold):
    """
    This function is used to calculate the weighted entropy for two different dataframes (less than threshold values and
    more than threshold values)
    :param less_than_threshold: Dataframe containing records that are less than the threshold value
    :param more_than_threshold: Dataframe containing records that are more than the threshold value
    :return: The total_weighted_entropy
    """

    ''' Storing total number of records '''
    total_records = len(less_than_threshold) + len(more_than_threshold)

    ''' Calculating the probability '''
    less_than_probability = len(less_than_threshold) / total_records
    more_than_probability = len(more_than_threshold) / total_records

    ''' Converting the dataframe to numpy arrays '''
    less_than_threshold_values = less_than_threshold.values
    more_than_threshold_values = more_than_threshold.values

    ''' Storing the target attribute values (Muffin or Cupcake) for threshold values '''
    target_for_less_than = less_than_threshold_values[:, -1]
    target_for_more_than = more_than_threshold_values[:, -1]

    ''' Finding the counts of muffin and cupcake for values lower than and greater than threshold value '''
    recipe_type, less_than_cupcake_muffin_count = np.unique(target_for_less_than, return_counts=True)
    recipe_type, more_than_cupcake_muffin_count = np.unique(target_for_more_than, return_counts=True)

    # print(recipe_type, more_than_cupcake_muffin_count, len(more_than_cupcake_muffin_count))
    ''' To ensure there are at least 5 records in each node '''
    if less_than_cupcake_muffin_count.sum() < 5 or more_than_cupcake_muffin_count.sum() < 5:
        ''' Return horrible badness '''
        return math.inf
    else:
        ''' Find the entropies for less than threshold values and more than threshold values '''
        less_than_entropy = sum((less_than_cupcake_muffin_count / less_than_cupcake_muffin_count.sum()) * - np.log2(
            less_than_cupcake_muffin_count / less_than_cupcake_muffin_count.sum()))
        more_than_entropy = sum((more_than_cupcake_muffin_count / more_than_cupcake_muffin_count.sum()) * - np.log2(
            more_than_cupcake_muffin_count / more_than_cupcake_muffin_count.sum()))

        ''' Calculate the total weighted entropy '''
        total_weighted_entropy = less_than_probability * less_than_entropy + more_than_probability * more_than_entropy

    return total_weighted_entropy

def find_best_attribute_threshold_entropy(original_training_data):
    """
    This function returns the best_split_index,best_attribute,best_threshold,best_minimum_entropy
    :param original_training_data: The original data in the form of a dataframe
    :return: best_split_index,best_attribute,best_threshold,best_minimum_entropy
    """

    ''' Initialize the values to horrible badness/infinity  '''
    best_minimum_entropy = math.inf
    best_threshold = math.inf
    best_attribute = ''
    best_split_index = math.inf

    ''' Storing column names in a list '''
    column_name_list = original_training_data.columns.values.tolist()

    ''' Storing columns names in a list except the target attribute '''
    attribute_columns_list = column_name_list[:-1]

    ''' Converting dataframe values to a numpy array for faster access and calculations '''
    original_training_data_values = original_training_data.values

    ''' Iterating through the dataframe column by column. attribute column list is [Flour,Sugar,Oils,Proteins]'''
    for columnindex in range(0,len(attribute_columns_list)):

        ''' Storing all threshold values for each column in a separate dataframe. : means select all rows [:,columnindex] '''
        threshold_values = original_training_data_values[:,columnindex]

        ''' For every possible threshold we have to check if it gives us the best split and minimum entropy '''
        for threshold in threshold_values:
            ''' Checking if the value is within the range (0-10) '''
            if 0 < threshold <= 10:
                ''' Splitting the data according to the threshold value. less_than_threshold contains all records where the 
                 threshold value is less than the row value of that particular column. Similarly for more_than_threshold '''
                less_than_threshold = original_training_data[threshold_values <= threshold]
                more_than_threshold = original_training_data[threshold_values > threshold]

                ''' Calling the total weighted entropy by passing in two dataframes'''
                total_weighted_entropy = find_entropy(less_than_threshold,more_than_threshold)

            ''' Check if current entropy is less/better than best minimum entropy and if it is, then update the best minimum
            entropy to current entropy and store the split index, threshold, and attribute. '''
            if total_weighted_entropy < best_minimum_entropy:
                best_minimum_entropy = total_weighted_entropy
                best_threshold = threshold
                best_attribute = attribute_columns_list[columnindex]
                best_split_index = columnindex

    return best_split_index,best_attribute,best_threshold,best_minimum_entropy

def classification(original_training_data):
    """
    This function returns the majority class between cupcake and muffin
    :param original_training_data: The original data in the form of a dataframe
    :return: majority_class [Muffin or CupCake]
    """

    ''' Storing the dataframe as numpy array '''
    original_training_data_values = original_training_data.values

    ''' Storing the values of target attribute for finding out the counts of each recipetype'''
    target_column = original_training_data_values[:, -1]

    ''' Recipe_type stores the unique values of target attribute in the form of a list [Muffin Cupcake] 
    cupcake_muffin_count stores the count of muffin and cupcakes in the form of a list [451 451]'''
    recipe_type, cupcake_muffin_count = np.unique(target_column, return_counts=True)

    ''' cupcake_muffin_count.argmax() returns the index of the highest value. In this case, it will return the index of 
     muffin or cupcake count. '''
    majority_class = recipe_type[cupcake_muffin_count.argmax()]

    return majority_class


def check_if_stopping_criterion_is_met(original_training_data_values):
    """
    This function is used as a check for the decision tree code and returns True if it satisfies any of the stopping
    criterion. Else returns False. Stopping criterion are:
    1. If there are less than 23 data points
    2. If call_depth is > 10
    3. If one of the nodes has more than 90% of the data
    :param original_training_data_values: Numpy array of original training data
    :return: Boolean value (True or False) based on if the stopping criterion is met or not
    """
    if len(original_training_data_values)<23:
        return True
    else:
        target_column = original_training_data_values[:, -1]
        recipe_type, cupcake_muffin_count = np.unique(target_column, return_counts=True)
        cupcake_ratio = cupcake_muffin_count[0] / (cupcake_muffin_count.sum())
        muffin_ratio = cupcake_muffin_count[1] / (cupcake_muffin_count.sum())

        if cupcake_ratio >= 0.9 or muffin_ratio >= 0.9:
            return True
        else:
            return False

def decision_tree(original_training_data,call_depth):
    """
    This is the decision tree function which takes in original dataframe and then finds the best attribute, best threshold to split data, best minimum entropy.
    After finding the first best split/threshold, it splits the original data based on the best threshold and then the same decision tree function is recursively
    called on the split data until the stopping criterion is met. The stopping criterion considered for this program are:
    1. If there are less than 23 data points
    2. If call_depth is > 10
    3. If one of the nodes has more than 90% of the data

    :param original_training_data: The original data in the form of a dataframe
    :param call_depth: It is the depth of the decision tree and it should not exceed 10
    :return: The resulting tree in the form of dictionary where key is best split and threshold and value is either another dictionary
    or leaf node (Muffin or Cupcake)
    """

    ''' Checking the stopping criterion. If yes then it returns the majority class (Muffin or CupCake) '''
    if check_if_stopping_criterion_is_met(original_training_data.values) or call_depth > 10:
        majority = classification(original_training_data)
        return majority

    else:
        ''' Each time we split the data and go deeper, we increment the depth of the tree '''
        call_depth += 1

        ''' Finding the best attribute, best threshold to split data, best minimum entropy '''
        best_split_index, best_attribute, best_threshold, best_minimum_entropy = find_best_attribute_threshold_entropy(original_training_data)
        original_training_data_values = original_training_data.values

        best_split_values = original_training_data_values[:,best_split_index]

        less_than_threshold = original_training_data[best_split_values <= best_threshold]
        more_than_threshold = original_training_data[best_split_values > best_threshold]

        ''' Initializing a variable called as condition which stores the format of the key for the resulting decision tree dictionary '''
        condition = original_training_data.columns[best_split_index] + " <= " + str(best_threshold)

        ''' Initializing a dictionary where key is condition and value is a list. This is the basic data structure in which the
         resulting decision tree is stored '''
        sub_tree = {condition: []}

        ''' Calling the decision tree recursively '''
        left_tree = decision_tree(less_than_threshold, call_depth)
        right_tree = decision_tree(more_than_threshold, call_depth)

        ''' For removing edge cases where on either split, the resulting decision tree gives the same result '''
        if left_tree == right_tree:
            sub_tree = left_tree
        else:
            ''' Appending the smaller trees in the final decision tree '''
            sub_tree[condition].append(left_tree)
            sub_tree[condition].append(right_tree)

        return sub_tree


def main():
    """
    This function is used to read training data, build decision tree, generate a classifier program which will then read a test data file and generate another
    csv file which stores the result for the classification results on the test data
    """

    ''' Reading the training data file '''
    original_training_data = pd.read_csv("DT_Data_CakeVsMuffin_v012_TRAIN.csv")

    ''' Storing the final decision tree '''
    final_tree = decision_tree(original_training_data,0)

    ''' Printing the final decision tree '''
    print("This is the resulting decision tree: \n")
    print(final_tree)

    ''' Iterating through the dictionary by using the key values '''
    for key in final_tree.keys():
        ''' Parent = Flour <= 5.1636'''
        parent = key
        ''' left_child = [{'Oils <= 3.1265': [{'Flour <= 2.7291': [{'Proteins <= 2.6527': ['Muffin', 'CupCake']}, 'Muffin']}, 'CupCake']}'''
        left_child = final_tree[parent][0]
        ''' right_child = {'Oils <= 7.7793': ['Muffin', {'Flour <= 8.2225': ['CupCake', 'Muffin']}]}]'''
        right_child = final_tree[parent][1]

    ''' Writing a file which generates code for classification '''
    file = open('HW06_Parchand_Nihal_Classifier.py','w+')
    file.write("'''Importing libraries''' "
               "\n\nimport pandas as pd \n\ndef main():"
               "\n\tdata_df = pd.read_csv('DT_Data_CakeVsMuffin_v012_TEST.csv')"
               "\n\tresult = []"
               "\n\tfor row in range(0,len(data_df)):"
               "\n\t\tFlour = data_df.loc[row][0]"
               "\n\t\tSugar = data_df.loc[row][1]"
               "\n\t\tOils = data_df.loc[row][2]"
               "\n\t\tProteins = data_df.loc[row][3]"
               "\n\t\tif {}:\n".format(parent))

    ''' Iterating through the left_tree '''
    for key in left_child.keys():
        file.write("\t\t\tif {}:\n".format(key))

        ''' Iterating through the inner left_tree '''
        for inner_key in left_child[key][0].keys():
            file.write("\t\t\t\tif {}:\n".format(inner_key))

            for inner_inner_key in ((left_child[key][0])[inner_key])[0]:
                file.write("\t\t\t\t\tif {}:\n".format(inner_inner_key))
                file.write("\t\t\t\t\t\tresult.append(0)\n")
                file.write("\t\t\t\t\telse:\n".format(inner_inner_key))
                file.write("\t\t\t\t\t\tresult.append(1)\n")

        file.write("\t\t\t\telse:\n")
        file.write("\t\t\t\t\tresult.append(0)\n")
        file.write("\t\t\telse:\n")
        file.write("\t\t\t\tresult.append(1)\n")
        file.write("\t\telse:\n")

    ''' Iterating through the right_tree '''
    for key in right_child.keys():
        file.write("\t\t\tif {}:\n".format(key))
        file.write("\t\t\t\tresult.append(0)\n")
        for inner_key in right_child[key][1].keys():
            file.write("\t\t\telif {}:\n".format(inner_key))
            file.write("\t\t\t\tresult.append(1)\n")
            file.write("\t\t\telse:\n")
            file.write("\t\t\t\tresult.append(0)\n\n")

    ''' Writing the results of classifier to a csv file '''
    file.write(
        "\twith open('HW06_Parchand_Nihal_MyClassifications.csv', 'w+') as file2:\n"
        "\t\tfor value in result:\n"
        "\t\t\tfile2.write(str(value))\n"
        "\t\t\tfile2.write('\\n')\n\n"
        "main()")

main()