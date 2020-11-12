'''Importing libraries''' 

import pandas as pd


def main():

	data_df = pd.read_csv('Clean_data.csv')

	result = []
	for value in data_df['School Club?'].values:
		if value == 'environment':
			result.append(0)

		if value == 'volunteering':
			result.append(0)

		if value == 'nerdheard':
			result.append(0)

		if value == 'none':
			result.append(0)

		if value == 'tech crew':
			result.append(0)

		if value == 'humane society':
			result.append(0)

		if value == 'food club':
			result.append(0)

		if value == 'un':
			result.append(0)

		if value == 'cs_club':
			result.append(0)

		if value == 'math':
			result.append(0)

		if value == 'anime':
			result.append(0)

		if value == 'chess':
			result.append(0)

		if value == 'robotics':
			result.append(0)

		if value == 'science':
			result.append(0)

		if value == 'radio':
			result.append(0)

		if value == 'gamedev':
			result.append(0)

		if value == 'library':
			result.append(0)

		if value == 'ncc':
			result.append(0)

		if value == 'trivia':
			result.append(0)

		if value == 'overly dramatic club':
			result.append(0)

		if value == 'socialservice':
			result.append(1)

		if value == 'debatable':
			result.append(0)

		if value == 'community':
			result.append(0)

		if value == 'spacecadet':
			result.append(0)

		if value == 'deca':
			result.append(0)

		if value == 'music':
			result.append(0)

		if value == 'art':
			result.append(0)

	with open('HW05_Parchand_Nihal_Results.csv', 'w+') as f:
		for value in result:
			f.write(str(value))
			f.write('\n')
main()