import numpy as np

x = [[ 1., 0., 0.], [ 0., 1., 2.]]

# print(np.ndim(x))       # 2
# # print(x.ndim)
# print(np.shape(x))      # (2, 3)
# print(np.size(x))       # 6     the total number of elements of the array.
# # print(np.itemsize(x))

# print(np.data(x))

a = np.arange(15).reshape(3, 5)

print(a)
print(x)

print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.itemsize)

