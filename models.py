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
        return nn.DotProduct(self.get_weights(),x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        val = nn.as_scalar(self.run(x))
        if val < 0 :
            return -1
        else:
            return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        is_outdated = True
        while is_outdated:
            is_outdated = False
            for x,y in dataset.iterate_once(1):
                if (nn.as_scalar(y)) != self.get_prediction(x):
                    self.get_weights().update(x, nn.as_scalar(y))
                    is_outdated = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        h1 = 10
        h2 = 90
        self.w1 = nn.Parameter(1,h1)
        self.w2 = nn.Parameter(h1,h2)
        self.w3 = nn.Parameter(h2,1)
        self.b1 = nn.Parameter(1,h1)
        self.b2 = nn.Parameter(1,h2)
        self.b3 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        s1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        s2 = nn.AddBias(nn.Linear(nn.ReLU(s1), self.w2), self.b2)
        s3 = nn.AddBias(nn.Linear(nn.ReLU(s2), self.w3), self.b3)
        return s3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        multiplier = -0.02
        is_outdated = True
        while is_outdated:
            is_outdated = False
            for x, y in dataset.iterate_once(40):
                loss = self.get_loss(x, y)
                if (nn.as_scalar(loss) >= 0.001):
                    grad_w1, grad_w2, grad_w3, grad_b1, grad_b2, grad_b3 = nn.gradients(loss, [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                    self.w1.update(grad_w1, multiplier)
                    self.w2.update(grad_w2, multiplier)
                    self.w3.update(grad_w3, multiplier)
                    self.b1.update(grad_b1, multiplier)
                    self.b2.update(grad_b2, multiplier)
                    self.b3.update(grad_b3, multiplier)
                    is_outdated = True

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
        h1 = 300
        # h2 = 50
        self.w1 = nn.Parameter(784,h1)
        # self.w2 = nn.Parameter(h1,h2)
        self.w2 = nn.Parameter(h1,10)
        self.b1 = nn.Parameter(1,h1)
        # self.b2 = nn.Parameter(1,h2)
        self.b2 = nn.Parameter(1,10)
        "*** YOUR CODE HERE ***"

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
        s1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        s2 = nn.AddBias(nn.Linear(nn.ReLU(s1), self.w2), self.b2)
        # s3 = nn.AddBias(nn.Linear(nn.ReLU(s2), self.w3), self.b3)
        return s2
        "*** YOUR CODE HERE ***"

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
        return nn.SoftmaxLoss(self.run(x), y)
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        multiplier = -0.03
        is_outdated = True
        while is_outdated:
            is_outdated = False
            for x, y in dataset.iterate_once(5):
                loss = self.get_loss(x, y)
                val_accuracy = dataset.get_validation_accuracy()
                if val_accuracy < 0.974:
                    grad_w1, grad_w2, grad_b1, grad_b2 = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
                    self.w1.update(grad_w1, multiplier)
                    self.w2.update(grad_w2, multiplier)
                    # self.w3.update(grad_w3, multiplier)
                    self.b1.update(grad_b1, multiplier)
                    self.b2.update(grad_b2, multiplier)
                    # self.b3.update(grad_b3, multiplier)
                    is_outdated = True
                    # if val_accuracy >= 0.8 and val_accuracy < 0.9 and multiplier == -0.04:
                    #     multiplier = -0.04
                    # if val_accuracy >= 0.9 and val_accuracy < 0.96 and multiplier == -0.03:
                    #     multiplier = -0.03
                    # if val_accuracy >= 0.96 and val_accuracy < 0.973 and multiplier == -0.02:
                    #     multiplier = -0.01
        "*** YOUR CODE HERE ***"

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        h1 = 75
        h2 = 75
        self.wi = nn.Parameter(self.num_chars, h1)
        self.wh = nn.Parameter(h1, h2)
        self.wo = nn.Parameter(h2, len(self.languages))
        self.b1 = nn.Parameter(1, h1)
        self.b2 = nn.Parameter(1, h2)
        self.b3 = nn.Parameter(1, len(self.languages))
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        z = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.wi), self.b1)), self.wh), self.b2)
        for x in list(xs[1:]):
            add_wi_wh = nn.Add(nn.Linear(x,self.wi), nn.Linear(z, self.wh))
            add_bias_wi_wh = nn.AddBias(add_wi_wh, self.b1)
            z = nn.AddBias(nn.Linear(nn.ReLU(add_bias_wi_wh), self.wh), self.b2)
        z = nn.AddBias(nn.Linear(z, self.wo), self.b3)
        return z
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        multiplier = -0.03
        is_outdated = True
        while is_outdated:
            is_outdated = False
            for x, y in dataset.iterate_once(50):
                loss = self.get_loss(x, y)
                val_accuracy = dataset.get_validation_accuracy()
                if val_accuracy < 0.84:
                    grad_wi, grad_wh, grad_wo, grad_b1, grad_b2, grad_b3 = nn.gradients(loss, [self.wi, self.wh, self.wo, self.b1, self.b2, self.b3])
                    self.wi.update(grad_wi, multiplier)
                    self.wh.update(grad_wh, multiplier)
                    self.wo.update(grad_wo, multiplier)
                    self.b1.update(grad_b1, multiplier)
                    self.b2.update(grad_b2, multiplier)
                    self.b3.update(grad_b3, multiplier)
                    if val_accuracy >= 0.7 and val_accuracy < 0.75 and multiplier == -0.025:
                        multiplier = -0.02
                    if val_accuracy >= 0.75 and val_accuracy < 0.8 and multiplier == -0.02:
                        multiplier = -0.01
                    if val_accuracy >= 0.8 and val_accuracy < 0.83 and multiplier == -0.01:
                        multiplier = -0.005
                    if val_accuracy >= 0.83 and val_accuracy < 0.84 and multiplier == -0.008:
                        multiplier = -0.001
                    is_outdated = True
        "*** YOUR CODE HERE ***"
