# TensorFlow: approximation of a quadratic polynomial
This test model was created to approximate the quadratic polynomial f(x) = 2x² + x. 

### Model
The model is a generic second degree polynomial f(x) = ax² + bx + c.

### Training data
The training data for this example is conformed by function evaluations (it does not have noise):

`x = [0, 1, 2, 3, 4]  y = [0, 3, 10, 21, 36]`

### Results
The final results after 5000 iterations is:

`a: 2.00039, b: 0.998191, c: 0.00141833, loss: 2.84776e-06`

This results are very close to the real values: 

`a: 2, b: 1, c: 0`

### Note
The parameter passed to GradientDescentOptimizer could cause the algorithm to diverge. Try 0.01 in this code and see what happens.
