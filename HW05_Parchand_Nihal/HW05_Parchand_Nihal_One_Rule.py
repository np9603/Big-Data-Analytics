"""
file: HW05_Parchand_Nihal_One_Rule.py
language: python3.7
author: np9603@cs.rit.edu Nihal Surendra Parchand
date: 09/20/2019
"""

''' Importing libraries'''

import math
import numpy as np
import pandas as pd
import re


def clean(original_data):
    """
    This function is used to clean the original dataframe
    :param original_data: The original uncleaned data in the form of a dataframe
    :return: Cleaned dataframe
    """

    ''' Storing the original data into a duplicate dataframe for cleaning'''
    clean_data_df = original_data

    ''' Deleting the UID, favorite color, musical instrument columns as mentioned in the email '''
    clean_data_df = clean_data_df.drop(columns=['UID', 'Of a dozen colors, what is your Favorite '
                                                       'Namespace colors?', 'Music Program?'])

    ''' Deleting columns which have same answers for each row '''
    clean_data_df = clean_data_df.drop(columns=['Polydactyl?  (6 fingers?)',
                                                'Who invented the first compiler?',
                                                'Youngest Nobel Lauriet', 'Log2(1024)','Peanut Allergy','Corn Allergy'])


    ''' Cleaning column 'How many times a week do you exercise?' '''
    clean_data_df.loc[clean_data_df['How many times a week do you exercise?'] == '0 Exercise? Never touch the stuff.',
                      'How many times a week do you exercise?'] = '0'

    ''' Cleaning column 'How many times a week do you wash your hair?'  There were some misinterpreted values like 4-Mar
    and 6-May which actually meant 3_to_4 and 5_to_6 respectively. '''
    clean_data_df.loc[clean_data_df['How many times a week do you wash your hair?'] == '4-Mar',
                      'How many times a week do you wash your hair?'] = '3_to_4'
    clean_data_df.loc[clean_data_df['How many times a week do you wash your hair?'] == '6-May',
                      'How many times a week do you wash your hair?'] = '5_to_6'

    ''' Cleaning column 'Allergies?' '''
    clean_data_df.loc[clean_data_df['Allergies?'] == ' Yes, hey fever and typical allergies?','Allergies?'] = 'Fever and typical allergies'
    clean_data_df.loc[clean_data_df['Allergies?'] == ' No, never touch the stuff.','Allergies?'] = 'No'

    ''' Cleaning column 'Linear Math?' Converting the boolean values to string values using astype(str)'''
    clean_data_df['Linear Math?'] = clean_data_df['Linear Math?'].astype(str)


    ''' Cleaning column 'Longest Hair Length?' Replaced similar meaning answers to keep it uniform throughout the column'''
    clean_data_df.loc[clean_data_df['Longest Hair Length?'] == 'Six to 10 inches (15cm to 25cm)','Longest Hair Length?'] = 'SixToTen'
    clean_data_df.loc[clean_data_df['Longest Hair Length?'] == 'Less then six inches (15 cm).','Longest Hair Length?'] = 'SixUnder'

    ''' Cleaning column 'Of the dozen colors, what is your eye color?' '''
    clean_data_df.loc[clean_data_df['Of the dozen colors, what is your eye color?'] == 'Cyan (light blue)','Of the dozen colors, what is your eye color?'] = 'Cyan'

    ''' Cleaning column 'How many NameSpace colors?' Strip() method used for removing leading and trailing white spaces '''
    clean_data_df['How many NameSpace colors?'] = clean_data_df['How many NameSpace colors?'].astype(str)
    clean_data_df['How many NameSpace colors?'] = clean_data_df['How many NameSpace colors?'].str.strip()


    ''' Cleaning column 'Star Wars Question' Converting all answers to lowercase '''
    clean_data_df['Star Wars Question'] = clean_data_df['Star Wars Question'].str.lower()

    ''' Cleaning column 'Winnie The Pooh' '''
    clean_data_df['Winnie The Pooh'] = clean_data_df['Winnie The Pooh'].str.lower()
    # Used regex to replace all occurrences of (anything before)+pooh to pooh
    clean_data_df['Winnie The Pooh'] = clean_data_df['Winnie The Pooh'].replace(to_replace =['.*pooh'], value ='pooh', regex = True)
    clean_data_df.loc[clean_data_df['Winnie The Pooh'] == 'poo','Winnie The Pooh'] = 'pooh'
    clean_data_df.loc[clean_data_df['Winnie The Pooh'] == 'pooh!','Winnie The Pooh'] = 'pooh'

    ''' Cleaning column 'School Club?' '''
    clean_data_df['School Club?'] = clean_data_df['School Club?'].str.lower()


    ''' Cleaning column 'Name of Computer' '''
    clean_data_df['Name of Computer'] = clean_data_df['Name of Computer'].str.lower()
    clean_data_df.loc[clean_data_df['Name of Computer'] == 'super computer deep thought','Name of Computer'] = 'deep thought'
    # Replaced '_' with N/A because these were the missing values
    clean_data_df.loc[clean_data_df['Name of Computer'] == '_','Name of Computer'] = 'N/A'


    ''' Cleaning column 'Combination to Air Shield' '''
    clean_data_df.loc[clean_data_df['Combination to Air Shield'] == '1-2-3-4-5', 'Combination to Air Shield'] = '12345'
    clean_data_df.loc[clean_data_df['Combination to Air Shield'] == '_', 'Combination to Air Shield'] = 'N/A'


    ''' Cleaning column 'Airspeed of Swallow' '''
    clean_data_df['Airspeed of Swallow'] = clean_data_df['Airspeed of Swallow'].str.lower()
    # Used regex to replace all occurrences of (anything before)+african or european+(anything after) to african or european swallow?
    clean_data_df['Airspeed of Swallow'] = clean_data_df['Airspeed of Swallow'].replace(to_replace =['.*african or european.*'], value ='african or european swallow?', regex = True)
    clean_data_df.loc[clean_data_df['Airspeed of Swallow'] == 'african vs european', 'Airspeed of Swallow'] = 'african or european swallow?'

    # Replacing all different answers (10/11mph/24/24mph) to 24mph
    clean_data_df['Airspeed of Swallow'] = clean_data_df['Airspeed of Swallow'].replace(to_replace =['.*24.*'], value ='24mph', regex = True)
    clean_data_df['Airspeed of Swallow'] = clean_data_df['Airspeed of Swallow'].replace(to_replace=['.*(10|11).*'], value='24mph', regex=True)

    ''' Cleaning column 'Favorite flavor of ice cream?' '''

    clean_data_df['Favorite flavor of ice cream?'] = clean_data_df['Favorite flavor of ice cream?'].str.lower()
    clean_data_df.loc[clean_data_df['Favorite flavor of ice cream?'] == 'cookies n creme', 'Favorite flavor of ice cream?'] = 'cookies and cream'
    clean_data_df.loc[clean_data_df['Favorite flavor of ice cream?'] == 'coffee coffee buzzbuzz', 'Favorite flavor of ice cream?'] = 'coffee'

    ''' Cleaning column 'Most Favorite Pizza Topping?' '''
    clean_data_df['Most Favorite Pizza Topping?'] = clean_data_df['Most Favorite Pizza Topping?'].str.lower()

    clean_data_df['Most Favorite Pizza Topping?'] = clean_data_df['Most Favorite Pizza Topping?'].replace(to_replace=['.*cheese.*'], value='cheese', regex=True)
    clean_data_df['Most Favorite Pizza Topping?'] = clean_data_df['Most Favorite Pizza Topping?'].replace(to_replace=['.*peppers.*'], value='peppers', regex=True)
    clean_data_df.loc[clean_data_df['Most Favorite Pizza Topping?'] == 'ham (or or canadian bacon)', 'Most Favorite Pizza Topping?'] = 'ham'

    ''' Cleaning column 'Least Favorite Pizza Topping?' '''
    clean_data_df['Least Favorite Pizza Topping?'] = clean_data_df['Least Favorite Pizza Topping?'].str.lower()

    # Removing leading and trailing white spaces
    clean_data_df['Least Favorite Pizza Topping?'] = clean_data_df['Least Favorite Pizza Topping?'].str.strip()

    clean_data_df['Least Favorite Pizza Topping?'] = clean_data_df['Least Favorite Pizza Topping?'].replace(to_replace=['.*cheese.*'], value='cheese', regex=True)
    clean_data_df['Least Favorite Pizza Topping?'] = clean_data_df['Least Favorite Pizza Topping?'].replace(to_replace=['.*peppers.*'], value='peppers', regex=True)
    clean_data_df['Least Favorite Pizza Topping?'] = clean_data_df['Least Favorite Pizza Topping?'].replace(to_replace=['.*anchovies.*'], value='anchovies', regex=True)
    clean_data_df['Least Favorite Pizza Topping?'] = clean_data_df['Least Favorite Pizza Topping?'].replace(to_replace=['.*pineapples.*'], value='pineapples', regex=True)

    ''' Cleaning column 'Avg HRS SLEEP' Converting misinterpreted values to their corresponding actual values '''
    clean_data_df.loc[clean_data_df['Avg HRS SLEEP'] == '8-Jun', 'Avg HRS SLEEP'] = '6-8'
    clean_data_df.loc[clean_data_df['Avg HRS SLEEP'] == '8-Jul', 'Avg HRS SLEEP'] = '7-8'
    clean_data_df.loc[clean_data_df['Avg HRS SLEEP'] == '7-Jun', 'Avg HRS SLEEP'] = '6-7'
    clean_data_df.loc[clean_data_df['Avg HRS SLEEP'] == '7:00', 'Avg HRS SLEEP'] = '7'


    ''' Cleaning column 'Breakfast Drink' Converted to lowercase for combining similar answers '''
    clean_data_df['Breakfast Drink'] = clean_data_df['Breakfast Drink'].str.lower()

    ''' Cleaning column 'Usual night sleep' '''
    clean_data_df.loc[clean_data_df['Usual night sleep'] == '8-Jun', 'Usual night sleep'] = '6-8'
    clean_data_df.loc[clean_data_df['Usual night sleep'] == '8-Jul', 'Usual night sleep'] = '7-8'
    clean_data_df.loc[clean_data_df['Usual night sleep'] == '7-Jun', 'Usual night sleep'] = '6-7'
    clean_data_df.loc[clean_data_df['Usual night sleep'] == '7 and a half', 'Usual night sleep'] = '7.5'
    clean_data_df.loc[clean_data_df['Usual night sleep'] == '8:00', 'Usual night sleep'] = '8'


    ''' Cleaning column 'Median Sleep' '''
    clean_data_df.loc[clean_data_df['Median Sleep'] == '~7', 'Median Sleep'] = '7'
    clean_data_df.loc[clean_data_df['Median Sleep'] == '7:30', 'Median Sleep'] = '7.5'

    # Median sleep hours can not be 0. Hence replaced with N/A
    clean_data_df.loc[clean_data_df['Median Sleep'] == '0', 'Median Sleep'] = 'N/A'

    # Someone misinterpreted the question and wrote 28. I think he/she might have forgotten to divide by 7. (28/7=4)
    clean_data_df.loc[clean_data_df['Median Sleep'] == '28', 'Median Sleep'] = '4'

    ''' Cleaning column 'Elevator Usage?' '''
    clean_data_df['Elevator Usage?'] = clean_data_df['Elevator Usage?'].str.strip()

    ''' Cleaning column 'Right or left Footed?  Going up Stairs?' '''
    clean_data_df.loc[clean_data_df['Right or left Footed?  Going up Stairs?'] == 'dont Êknow', 'Right or left Footed?  Going up Stairs?'] = 'dont know'


    ''' Cleaning column 'Can you tie a mens tie?' '''
    clean_data_df.loc[clean_data_df['Can you tie a mens tie?'] == 'No_I_CanKnot', 'Can you tie a mens tie?'] = 'No'
    clean_data_df.loc[clean_data_df['Can you tie a mens tie?'] == 'Ties? Never touch the stuff.', 'Can you tie a mens tie?'] = 'No'
    clean_data_df['Can you tie a mens tie?'] = clean_data_df['Can you tie a mens tie?'].replace(to_replace=['.*Half Windsor.*'], value='Half Windsor', regex=True)
    clean_data_df['Can you tie a mens tie?'] = clean_data_df['Can you tie a mens tie?'].replace(to_replace=['.*Full Windsor.*'], value='Full Windsor', regex=True)

    ''' Cleaning column 'French Braid Hair?' '''
    clean_data_df['French Braid Hair?'] = clean_data_df['French Braid Hair?'].str.strip()
    clean_data_df.loc[clean_data_df['French Braid Hair?'] == 'Hair? Never touch the stuff.', 'French Braid Hair?'] = 'No'
    clean_data_df.loc[clean_data_df['French Braid Hair?'] == 'of course. Everybody can do that!', 'French Braid Hair?'] = 'Yes'

    ''' Cleaning column 'Snack Food Preference?' '''
    clean_data_df['Snack Food Preference?'] = clean_data_df['Snack Food Preference?'].str.lower()
    clean_data_df['Snack Food Preference?'] = clean_data_df['Snack Food Preference?'].replace(to_replace=['.*tortilla.*'], value='tortilla chips', regex=True)

    ''' Cleaning column 'Eye lens issues?' '''
    clean_data_df['Eye lens issues?'] = clean_data_df['Eye lens issues?'].str.strip()


    ''' Cleaning column 'Toilet Paper Roll Out Over Front' '''
    clean_data_df.loc[clean_data_df['Toilet Paper Roll Out Over Front'] == 'TRUE', 'Toilet Paper Roll Out Over Front'] = 'out over the front'
    clean_data_df.loc[clean_data_df['Toilet Paper Roll Out Over Front'] == 'FALSE', 'Toilet Paper Roll Out Over Front'] = 'out over the back'

    ''' For printing the list of columns '''
    # print(list(clean_data_df.columns))

    ''' For printing the unique values of a column '''
    # print(clean_data_df['Airspeed of Swallow'].unique())

    ''' For printing the whole dataframe '''
    # print(clean_data_df.to_string())

    ''' Writing/Storing the clean data in a new csv file '''
    clean_data_df.to_csv(r'Clean_data.csv', index=None, header=True)

    ''' Returning the clean dataframe '''
    return clean_data_df

def find_best_attribute(clean_data_df):
    """
    This function is used to find the best attribute which best selects the target variable
    :param clean_data_df: The clean data frame
    :return: Dictionary of best attribute which has values set as 1 if there are more true values and 0 if there are more false values for the target variable
    """

    ''' Storing the column names in a list '''
    column_list = clean_data_df.columns.to_list()

    ''' Printing the column list'''
    # print(column_list)

    ''' Target attribute for this assignment is Most Favorite Pizza Topping?==peppers '''
    target_attribute = 'Most Favorite Pizza Topping?'

    ''' Initializing the minimum misclassification rate to maximum and then finding the best misclassification rate '''
    minimum_misclassification_rate = math.inf

    ''' Initializing the best attribute '''
    best_attribute = ''

    ''' Iterating through the column list '''
    for column in column_list:
        ''' Ignore the column if it is equal to the target attribute as the misclassification rate will always result to 0'''
        if column == target_attribute:
            pass
        else:
            ''' Storing unique values of each column in a list '''
            unique_values_in_column = clean_data_df[column].unique()

            ''' Initializing two dictionaries with keys as the unique values of each column with values as 0'''
            unique_values_dict_true = dict.fromkeys(unique_values_in_column, 0)
            unique_values_dict_false = dict.fromkeys(unique_values_in_column, 0)

            ''' Printing out the unique values in each column'''
            # print(unique_values_in_column)

            ''' Iterating through each record '''
            for value in range(0, len(clean_data_df[column])):
                ''' Checking the value for each record in each column against the corresponding index for peppers in the target attribute
                If it is equal to peppers then increase the value of the key by 1 in true dictionary'''
                if clean_data_df[target_attribute][value] == 'peppers':
                    unique_values_dict_true[clean_data_df[column][value]] += 1
                else:
                    ''' If it is not equal to peppers then increase the value of the key by 1 in false dictionary'''
                    unique_values_dict_false[clean_data_df[column][value]] += 1

            ''' Initializing the missed values '''
            missed_values = 0

            ''' Iterating through the two dictionaries and comparing their key value pairs to find the minimum value for each key '''
            for key in unique_values_dict_true:
                if key in unique_values_dict_false:
                    if unique_values_dict_true.get(key) <= unique_values_dict_false.get(key):
                        missed_values += int(unique_values_dict_true.get(key))
                    else:
                        missed_values += int(unique_values_dict_false.get(key))

            ''' Finding the misclassification rate by dividing the missed values by the number of observations '''
            misclassification_rate = missed_values / len(clean_data_df)

            ''' Printing the column, its missed values and its misclassification rate '''
            # print(column, missed_values, misclassification_rate)

            ''' Finding the minimum misclassification rate and best attribute '''
            if misclassification_rate < minimum_misclassification_rate:
                minimum_misclassification_rate = misclassification_rate
                best_attribute = column

    ''' Printing out the best attribute and its misclassification rate '''
    print("Best attribute is " + str(best_attribute) + " with Misclassification rate =",
          str(minimum_misclassification_rate))

    ''' Building One-Rule based on the best attribute '''
    ''' Initializing the lists which result to true and false '''
    one_rule_true = []
    one_rule_false = []

    ''' Initializing the dictionary for storing the unique values of best attribute and maximum of true values and false values '''
    one_rule_dict = {}

    ''' Iterating through the column list '''
    for column in column_list:
        if column == best_attribute:
            unique_values_in_column = clean_data_df[column].unique()
            unique_values_dict_true = dict.fromkeys(unique_values_in_column, 0)
            unique_values_dict_false = dict.fromkeys(unique_values_in_column, 0)

            for value in range(0, len(clean_data_df[column])):
                if clean_data_df[target_attribute][value] == 'peppers':
                    unique_values_dict_true[clean_data_df[column][value]] += 1
                else:
                    unique_values_dict_false[clean_data_df[column][value]] += 1

            for key in unique_values_dict_true:
                if key in unique_values_dict_false:
                    ''' For each unique value, if true values are more then append 1, else append 0'''
                    if unique_values_dict_true.get(key) > unique_values_dict_false.get(key):
                        one_rule_true.append(key)
                        one_rule_dict[key] = 1
                    else:
                        one_rule_false.append(key)
                        one_rule_dict[key] = 0

    ''' Printing out the One-Rule '''
    print("One rule : ")
    print("if (",end="")
    for value in one_rule_true:
        print(str(best_attribute) + "==" + str(value),end="")
    print("):")
    print("\t" + str(target_attribute) + " = true")
    print("else:")
    print("\t" + str(target_attribute) + " = false")

    ''' Printing out the list of unique values which have more true values '''
    # print(one_rule_true)

    ''' Printing out the list of unique values which have more false values '''
    # print(one_rule_false)

    ''' Return the dictionary containing key value pairs of unique values of columns and 0 or 1 based on maximum true or false values '''
    return one_rule_dict

def program_to_create_program(one_rule_dict):
    """
    This function is used to create a program which reads clean data and generates rules and then stores the result in a csv file
    :param one_rule_dict: Dictionary containing key value pairs of unique values of columns and 0 or 1 based on maximum true or false values
    """

    ''' Opening/Creating the file which creates another file which stores the result in a csv file format '''
    with open("HW05_Parchand_Nihal_Rule.py", 'w+') as file:
        file.write("'''Importing libraries''' \n\nimport pandas as pd\n\n\n"
                   "def main():\n\n"
                   "\tdata_df = pd.read_csv('Clean_data.csv')\n\n"
                   "\tresult = []\n"
                   "\tfor value in data_df['School Club?'].values:\n")

        ''' For every unique value in dictionary write its rule to append 1 or 0'''
        for key in one_rule_dict.keys():
            file.write("\t\tif value == '" + str(key) + "':\n\t\t\t"
                       "result.append(" + str(one_rule_dict[key]) +")\n\n")

        ''' Writing results to a csv file '''
        file.write("\twith open('HW05_Parchand_Nihal_Results.csv', 'w+') as f:\n"
                   "\t\tfor value in result:\n"
                   "\t\t\tf.write(str(value))\n"
                   "\t\t\tf.write('\\n')\n"
                   "main()")


def main():

    ''' Reading the data and storing it in a dataframe '''
    original_data = pd.read_csv("CS720_Obtuse_data_Anonymous_v037_TBK.csv",encoding='latin-1')

    ''' Passing the original data into the clean function to clean the dataframe and store it in clean_data_df'''
    clean_data_df = clean(original_data)

    ''' Passing the clean data for finding the best attribute and generating one rule accordingly '''
    one_rule_dict = find_best_attribute(clean_data_df)

    ''' Passing the resulting best attribute dictionary to the function which creates another program'''
    program_to_create_program(one_rule_dict)


main()


