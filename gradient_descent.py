from numpy import *

def predict_y(x, b, m):
     y = m*x + b
     return y

def predict_x(y, b, m):
     x = y/m - b
     return x

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0] # first column in the array
        y = points[i, 1] # seconds column in the array
        totalError += (y - (m * x + b)) **2 # sum of squared errors
    return totalError / float(len(points)) # average sum of squares for this step

def step_gradient(b_current, m_current, points, learningRate):
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        # calulate the b_gradient
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))

        # calulate the m_gradient
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    # use the gradients (nudge up or down with gradient multiplied by learningRate)
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)

    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]

def run():
    points = genfromtxt('data.csv', delimiter=',')

    ## hyperparameters
    learning_rate = 0.0001
    # y = mx + b (slope formula)
    # y-intercept - start with 0, will learn over time
    initial_b = 0
    # slope
    initial_m = 0
    # since such a small dataset we start with 1000.  on larger datasets
    # we'd want 100,000 or 100,000,000
    num_iterations = 1000

    # let's get the ideal (optimal) b and m
    [b, m] = gradient_descent_runner(
        points,
        initial_b,
        initial_m,
        learning_rate,
        num_iterations)

    error_for_line = compute_error_for_line_given_points(b, m, points)

    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, error_for_line)

    print("\nx predictions on the line\n")
    x = predict_x(5, b, m)
    print("Predict x: given y={}, x={}".format(5, x))
    x = predict_x(-5, b, m)
    print("Predict x: given y={}, x={}".format(-5, x))
    x = predict_x(2, b, m)
    print("Predict x: given y={}, x={}".format(2, x))

    print("\ny predictions on the line\n")
    y = predict_y(5, b, m)
    print("Predict y: given x={}, y={}".format(5, y))
    y = predict_y(-5, b, m)
    print("Predict y: given x={}, y={}".format(-5, y))
    y = predict_y(2, b, m)
    print("Predict y: given x={}, y={}".format(2, y))


if __name__ == '__main__':
    run()
