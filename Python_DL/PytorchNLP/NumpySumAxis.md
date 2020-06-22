# Understand Numpy Sum Axis

Reference: [https://medium.com/@aerinykim/numpy-sum-axis-intuition-6eb94926a5d1]

## For 2-D array
The way to understand the <b>"axis"</b> of numy sum is that it <b><i>collapses</i></b> the specified axis.
So when it collapses the axis 0 (<b>the row</b>), it becomes just one row (int sums colums-wise)

## Example
``
x = np.array([[0, 1], [2, 3]])

x.shape = (2, 2)

np.sum(x, axis=0)
= [x[0][0] + x[1][0], x[0][1]+x[1][1]] = [[2, 4]]

np.sum(x, axis=1)
= [x[0][0] + x[0][1], x[1][0] + x[1][1]] = [1, 5]
``

## For 3-D array (or N-D array)
``
x = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

x.shape = (2, 2, 3)

np.sum(x, axis=0)
= [x[0][0][0] + x[1][0][0], x[0][0][1]+x[1][0][1], etc.] = [[6, 8, 10],[12, 14, 16]]

np.sum(x, axis=1) (remove axi 2)
= [x[0][0][0] + x[0][1][0], etc.] = [[3, 5, 7],[15, 17, 19]]

np.sum(x, axis=2) (remove axis 1)
= [x[0][0][0] + x[0][0][1] + ..., etc] = [[3, 12], [21 ,30]]
``
