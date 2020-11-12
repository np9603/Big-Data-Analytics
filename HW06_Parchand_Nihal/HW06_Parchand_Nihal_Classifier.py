'''Importing libraries''' 

import pandas as pd 

def main():
	data_df = pd.read_csv('DT_Data_CakeVsMuffin_v012_TEST.csv')
	result = []
	for row in range(0,len(data_df)):
		Flour = data_df.loc[row][0]
		Sugar = data_df.loc[row][1]
		Oils = data_df.loc[row][2]
		Proteins = data_df.loc[row][3]
		if Flour <= 5.1636:
			if Oils <= 3.1265:
				if Flour <= 2.7291:
					if Proteins <= 2.6527:
						result.append(0)
					else:
						result.append(1)
				else:
					result.append(0)
			else:
				result.append(1)
		else:
			if Oils <= 7.7793:
				result.append(0)
			elif Flour <= 8.2225:
				result.append(1)
			else:
				result.append(0)

	with open('HW06_Parchand_Nihal_MyClassifications.csv', 'w+') as file2:
		for value in result:
			file2.write(str(value))
			file2.write('\n')

main()