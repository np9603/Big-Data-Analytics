'''
Name - NIHAL SURENDRA PARCHAND
Email - np9603@rit.edu
Version - Python 3.7
'''

# Importing Matplotlib library for plotting the values
import matplotlib.pyplot as plt


def calc_stopping_time(speed):
  """ This function is used to calculate the stopping time for speeds (0-120 mph)
  :param speed: speed is the speed of car from 0 - 120 mph
  :return: Stopping time of car ( i.e. maximum of 4 or the calculated time in seconds )
  """

  # alpha is the experimentally defined constant
  alpha = float(0.0085)

  # Calculate car stopping time in seconds using formula reaction_time_car = alpha * (speed in mph ^ 2)
  # If the resulting stopping time < 4 then return 4 as it is the maximum between 4 and the calculated stopping time
  # Otherwise return the calculated stopping time
  if alpha * (speed**2) > 4:
    return alpha * (speed**2)
  else:
    return 4


def calc_car_packing_density(speed):
  """
  This function is used to calculate car packing density for speeds (0-120 mph)
  :param speed: speed is the speed of car from 0 - 120 mph
  :return: Car packing density or the total distance maintained by two consequent cars to avoid collision
  """

  # Store the value of reaction time of car in a variable
  reaction_time_car = float(calc_stopping_time(speed))

  # Calculate car packing density by converting speed from mph to fps by multiplying it with 1.46667 and the
  # reaction time of car at that speed and then adding the length of car (i.e. 12 feet). This is the total distance
  # two consequent cars should maintain at a particular speed to avoid collisions.
  car_packing_density = speed * 1.46667 * reaction_time_car + 12

  return car_packing_density


def flux():
  """
  This function is used to calculate the total flux or cars per hour to find out the road efficiency as a function of
  speed
  """

  # Store the result of flux or cars per hour in a list which is later used for plotting a graph for data visualization
  number_of_cars_per_hour_list = []

  # Store the speeds in a list which is later used for plotting a graph for data visualization
  speed_list = []

  # A for loop for iterating through speeds 1-120 mph ( I skipped 0 mph because it does not make any sense for a car to
  # have speed 0 mph )

  for speed in range(1, 121):

    # Calculating total time required to travel the total distance ( car length + safety gap ) for every mph
    total_travel_time = calc_car_packing_density(speed) / (speed * 1.46667)

    # Calculating total number of cars that would travel in an hour
    number_of_cars_per_hour = int(3600/total_travel_time)

    # Appending the answers to the list
    number_of_cars_per_hour_list.append(number_of_cars_per_hour)
    speed_list.append(speed)

  # Plotting the speed vs number of cars per hour graph using Matplotlib plot function and specifying some parameters
  # like linewidth and markersize to make it more readable and visually appealing
  plt.plot(speed_list, number_of_cars_per_hour_list, linestyle='-', marker='o', linewidth=0.2, markersize=2)

  # Calculate the maximum number of cars per hour and store it in a variable for later reference
  maximum_number_of_cars_per_hour = max(number_of_cars_per_hour_list)

  # Finding the index of maximum number of cars per hour to find the corresponding most efficient speed
  maximum_number_of_cars_per_hour_index = number_of_cars_per_hour_list.index(maximum_number_of_cars_per_hour)

  # Finding the most efficient speed and storing it in the variable
  maximum_speed = speed_list[maximum_number_of_cars_per_hour_index]

  # Printing out the results in standard output
  print('The most efficient road speed in mph = ' + str(maximum_speed) + 'mph')
  print('The best efficiency = ' + str(maximum_number_of_cars_per_hour) + ' cars per hour')

  # For annotating the results found using the annotate function of Matplotlib library
  plt.annotate(str(maximum_number_of_cars_per_hour) + ' cars per hour at speed ' + str(maximum_speed) + ' mph' ,
               xy=(maximum_speed, maximum_number_of_cars_per_hour),
               xytext=(maximum_speed+30, maximum_number_of_cars_per_hour),
               arrowprops=dict(arrowstyle="-|>",facecolor='black'),
              )

  # Labelling the X-axis
  plt.xlabel("Speed (mph)")

  # Labelling the Y-axis
  plt.ylabel("Flux or Cars per hr")

  # Labelling the title of the plot
  plt.title("Road Efficiency as a Function of Speed")

  # For displaying the background grid instead of plain white background which makes the plot readable
  plt.grid()

  # Calling the show() function to display the plot
  plt.show()

# Calling the flux() function to run the program
flux()
