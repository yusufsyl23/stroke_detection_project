# Python3 code demonstrate clip() function

# importing the numpy
import numpy as np

in_array = 6
print ("Input array : ", in_array)

out_array = np.clip(in_array, a_min = 2, a_max = 6)
print ("Output array : ", out_array)