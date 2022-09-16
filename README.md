### python Path lib
```
import subprocess
from pathlib import Path
from typing import List

from h2hlib import MetabitProject, Package


def _get_proto_file_names() -> List[str]:
    result = []
    for proto_file in Path("interface").rglob("*.proto"):
        result.append(str(proto_file)[: -len(".proto")])
    return result
```
## np
multidimensional array
python array:  The type is specified at object creation time by using a type code, which is a single character.
```
import numpy as np
a = np.array([0, 0.5, 1.0, 1.5, 2.0])

type(a)
#numpy.ndarray

a = np.arange(2, 20, 2)
# array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])

a = np.arange(8, dtype=np.float)
# array([0., 1., 2., 3., 4., 5., 6., 7.])

a[5:] # slice is the same
a.sum()
a.std()
a.cumsum() # he cumulative sum of all elements (starting at index position 0).

# (vectorized) mathmatical operations
2 * a # note this is different than a normal python list which will get expaned by the multiplier
# array([ 0., 2., 4., 6., 8., 10., 12., 14.])
a ** 2 # This calculates element-wise the square values.
2 ** a # This interprets the elements of the ndarray as the powers
a ** a # This calculates the power of every element to itself.

# Universal functions
np.exp(a) # Calculates the exponential values element-wise. e to the power of each elements in a
np.sqrt(a)

# Note: Applying the universal function np.sqrt() to a Python float object is much slower than the same operation # with the math.sqrt() function
# np.sqrt(2.5) do not use
math.sqrt(2.5)
```
Multiple Dimensions
```
b = np.array([a, a * 2])
# array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
#        [ 0.,  2.,  4.,  6.,  8., 10., 12., 14.]])
b[0]
# array([0., 1., 2., 3., 4., 5., 6., 7.])
b[0, 2]
# 2.0
b[:, 1]
#  array([1., 2.]) thats how you get a column
b.sum(axis = 0) # sum along the first axis; i.e., column-wise.
# array([ 0.,  3.,  6.,  9., 12., 15., 18., 21.])

#left join 重复？
