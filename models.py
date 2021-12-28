import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        # return the dot product of weights and features using the pre-made nn method

        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        # calculate the value as a scalar, from the run function
        # return 1 if the result is positive and -1 if it is negative

        val = nn.as_scalar(self.run(x))
        if val >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        # we want to loop over the batched data and train until we have 100% accuracy
        # this is done using the update method of class nn.Parameter, which takes a multiplier and a direction
        # it operates by adding or subtracting the feature vector x to the weights when incorrectly classifying
        # in time, this converges to the correct split.

        loop = True

        while loop:

            loop = False
            for x, y in dataset.iterate_once(1):
                # if our predicted x value is not what the output value is we need to update

                predicted_class = self.get_prediction(x)
                correct_class = nn.as_scalar(y)

                if predicted_class != correct_class:

                    loop = True
                    nn.Parameter.update(self.w, x, correct_class)


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # initialize learning rate - note it is negative not positive as per Discourse query
        self.learning_rate = -0.01

        # initialize number of perceptrons in the hidden layer
        self.hidden_size_1 = 100

        # initialize weights, biases of hidden layer input
        self.hidden_weights_1 = nn.Parameter(1, self.hidden_size_1)
        self.hidden_biases_1 = nn.Parameter(1, self.hidden_size_1)

        # initialize weights, biases of hidden layer output
        self.hidden_weights_2 = nn.Parameter(self.hidden_size_1, 1)
        self.hidden_biases_2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # compute: f(x) = ReLU(x * weights_1 + biases_1) * weights_2 + biases_2

        # compute hidden layer calculations
        x_weights_1 = nn.Linear(x, self.hidden_weights_1)
        out_1 = nn.ReLU(nn.AddBias(x_weights_1, self.hidden_biases_1))

        # complete final prediction calculations
        x_weights_2 = nn.Linear(out_1, self.hidden_weights_2)
        return nn.AddBias(x_weights_2, self.hidden_biases_2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        # compute the loss with the predicted output of x and the correct output y, using square loss
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # train the neural network until the loss function returns less than 0.02
        while not nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:

            # train on an iteration of the data-set batch
            for x, y in dataset.iterate_once(1):

                # get loss for current
                loss = self.get_loss(x, y)

                # compute the gradient of the loss
                grad = nn.gradients(loss, [self.hidden_weights_1, self.hidden_biases_1,
                                           self.hidden_weights_2, self.hidden_biases_2])

                # update the weights and biases using gradients
                self.hidden_weights_1.update(grad[0], self.learning_rate)
                self.hidden_biases_1.update(grad[1], self.learning_rate)

                self.hidden_weights_2.update(grad[2], self.learning_rate)
                self.hidden_biases_2.update(grad[3], self.learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # initialize learning rate - note it is negative not positive as per Discourse query
        self.learning_rate = -0.01

        # initialize number of perceptrons in the first hidden layer
        self.hidden_size_1 = 100

        # initialize weights, biases between feature input and hidden layer 1
        self.hidden_weights_1 = nn.Parameter(784, self.hidden_size_1)
        self.hidden_biases_1 = nn.Parameter(1, self.hidden_size_1)

        # initialize weights, biases between hidden layer 1 and output
        self.hidden_weights_2 = nn.Parameter(self.hidden_size_1, 10)
        self.hidden_biases_2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # compute: f(x) = ReLU(x * weights_1 + biases_1) * weights_2 + biases_2

        # compute hidden layer calculations
        x_weights_1 = nn.Linear(x, self.hidden_weights_1)
        out_1 = nn.ReLU(nn.AddBias(x_weights_1, self.hidden_biases_1))

        # complete final prediction calculations
        x_weights_2 = nn.Linear(out_1, self.hidden_weights_2)
        return nn.AddBias(x_weights_2, self.hidden_biases_2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        # return the loss given by the soft-max loss function
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # train until the validation accuracy is at least 97.5%
        while dataset.get_validation_accuracy() < 0.975:

            # train on an iteration of the data-set batch
            for x, y in dataset.iterate_once(1):

                # get loss for current
                loss = self.get_loss(x, y)

                # compute the gradient of the loss
                grad = nn.gradients(loss, [self.hidden_weights_1, self.hidden_biases_1,
                                           self.hidden_weights_2, self.hidden_biases_2])

                # update the weights and biases using gradients
                self.hidden_weights_1.update(grad[0], self.learning_rate)
                self.hidden_biases_1.update(grad[1], self.learning_rate)

                self.hidden_weights_2.update(grad[2], self.learning_rate)
                self.hidden_biases_2.update(grad[3], self.learning_rate)
