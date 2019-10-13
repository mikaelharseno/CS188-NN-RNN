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
        return nn.DotProduct(self.w,x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        res = nn.as_scalar(self.run(x))
        if res >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        multiplier = 1
        bad = True
        while bad:
            correctall = True
            for x, y in dataset.iterate_once(batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y)*multiplier)
                    correctall = False
            if correctall == True:
                bad = False

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        inputdim = 1
        h1 = 120
        h2 = 80
        outputdim = 1
        self.w1 = nn.Parameter(inputdim, h1)
        self.b1 = nn.Parameter(1, h1)
        self.w2 = nn.Parameter(h1, h2)
        self.b2 = nn.Parameter(1, h2)
        self.w3 = nn.Parameter(h2, outputdim)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x,self.w1),self.b1))
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1,self.w2),self.b2))
        out = nn.Linear(h2,self.w3)
        return out

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
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 50
        multiplier = 0.05
        for x, y in dataset.iterate_forever(batch_size):
            grads = nn.gradients(self.get_loss(x,y),[self.w1,self.b1,self.w2,self.b2,self.w3])
            self.w1.update(grads[0], -1*multiplier)
            self.b1.update(grads[1], -1*multiplier)
            self.w2.update(grads[2], -1*multiplier)
            self.b2.update(grads[3], -1*multiplier)
            self.w3.update(grads[4], -1*multiplier)
            loss = nn.as_scalar(self.get_loss(x,y))
            print(loss)
            if loss < 0.0001:
                break
        

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
        inputdim = 784
        h1 = 400
        h2 = 320
        h3 = 200
        h4 = 200
        outputdim = 10
        self.w1 = nn.Parameter(inputdim, h1)
        self.b1 = nn.Parameter(1, h1)
        self.w2 = nn.Parameter(h1, h2)
        self.b2 = nn.Parameter(1, h2)
        self.w3 = nn.Parameter(h2, h3)
        self.b3 = nn.Parameter(1, h3)
        """self.w4 = nn.Parameter(h3, h4)
        self.b4 = nn.Parameter(1, h4)"""
        self.w5 = nn.Parameter(h3, outputdim)
        self.b5 = nn.Parameter(1, outputdim)

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
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x,self.w1),self.b1))
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1,self.w2),self.b2))
        h3 = nn.ReLU(nn.AddBias(nn.Linear(h2,self.w3),self.b3))
        "h4 = nn.ReLU(nn.AddBias(nn.Linear(h3,self.w4),self.b4))"
        out = nn.AddBias(nn.Linear(h3,self.w5),self.b5)
        return out

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
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        h1 = 270
        h2 = 200
        batch_size = 150
        m = 4
        acc 95.4 epoch 5
        acc 96.1 epoch 6 end

        h1 = 270
        h2 = 200
        batch_size = 150
        m = 6
        acc 96.3 epoch 5
        acc 96.8 epoch 6 end

        h1 = 270
        h2 = 200
        batch_size = 150
        m = 9
        acc 96.8 epoch 5
        acc 97.1 epoch 6 end

        h1 = 270
        h2 = 200
        batch_size = 150
        m = 12
        acc 95.8 epoch 5
        
        h1 = 270
        h2 = 200
        batch_size = 150
        m = 15
        hancur

        h1 = 270
        h2 = 200
        h3 = 130
        batch_size = 150
        m = 4
        acc 94.8 epoch 3

        h1 = 270
        h2 = 200
        h3 = 130
        batch_size = 150
        m = 6
        acc 95.6 epoch 3

        h1 = 270
        h2 = 200
        h3 = 130
        batch_size = 150
        m = 8
        acc 95.6 epoch 3

        h1 = 270
        h2 = 200
        h3 = 130
        batch_size = 150
        m = 11
        acc 93.5 epoch 3

        h1 = 350
        h2 = 240
        h3 = 200
        batch_size = 150
        m = 4
        acc 94.8 epoch 3

        h1 = 350
        h2 = 240
        h3 = 200
        batch_size = 150
        m = 8
        acc 96.4 epoch 3
        acc 97.1 epoch 5

        h1 = 350
        h2 = 240
        h3 = 200
        batch_size = 150
        m = 12
        acc 96.6 epoch 3
        acc 97.3 epoch 5
        acc 97.4 epoch 6 end

        h1 = 400
        h2 = 320
        h3 = 210
        batch_size = 150
        m = 12
        acc 96.6 epoch 3
        acc 97.5 epoch 5

        h1 = 400
        h2 = 320
        h3 = 210
        h4 = 180
        batch_size = 150
        m = 15
        acc 97.4 epoch 5
        
        h1 = 400
        h2 = 320
        h3 = 270
        h4 = 200
        batch_size = 150
        m = 9
        acc 97.4 epoch 5
        
        h1 = 400
        h2 = 320
        h3 = 270
        h4 = 200
        batch_size = 150
        m = 12
        acc 97.2 epoch 5 PASS

        h1 = 400
        h2 = 320
        h3 = 270
        h4 = 200
        batch_size = 600
        m = 12
        acc 95.9 epoch 5

        h1 = 400
        h2 = 320
        h3 = 200
        batch_size = 500
        m = 12
        acc 95.5 epoch 5

        h1 = 400
        h2 = 320
        h3 = 200
        batch_size = 500
        m = 17
        acc 96.4 epoch 5

        h1 = 400
        h2 = 320
        h3 = 200
        batch_size = 500
        m = 24
        acc 96.1 epoch 5
        acc 97.5 terminate 11 epochs
        acc 96.7 epoch 5
        10:56:28 (epoch 8)

        h1 = 400
        h2 = 320
        h3 = 200
        batch_size = 300
        m = 15
        acc 96.1 epoch 5
        11:10:20
        
        """
        "*** YOUR CODE HERE ***"
        batch_size = 300
        m = 20
        multiplier = 0.03*m
        count = 0
        lastvalidacc = 2
        i = 0
        for x, y in dataset.iterate_forever(batch_size):
            i = i + 1
            grads = nn.gradients(self.get_loss(x,y),[self.w1,self.b1,self.w2,self.b2,self.w3,self.b3,self.w5,self.b5])
            self.w1.update(grads[0], -1*multiplier)
            self.b1.update(grads[1], -1*multiplier)
            self.w2.update(grads[2], -1*multiplier)
            self.b2.update(grads[3], -1*multiplier)
            self.w3.update(grads[4], -1*multiplier)
            self.b3.update(grads[5], -1*multiplier)
            """self.w4.update(grads[6], -1*multiplier)
            self.b4.update(grads[7], -1*multiplier)"""
            self.w5.update(grads[6], -1*multiplier)
            self.b5.update(grads[7], -1*multiplier)
            validacc = dataset.get_validation_accuracy()
            if 0.81 < validacc < 0.87:
                multiplier = 0.03*m
            if 0.87 < validacc < 0.91:
                multiplier = 0.014*m
            if 0.91 < validacc < 0.973:
                multiplier = 0.012*m
            if validacc > 0.973 and lastvalidacc > validacc:
                multiplier = 0.007*m
                count += 1
                if count > 8:
                    break
            lastvalidacc = validacc

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
        "*** YOUR CODE HERE ***"
        inputdim = self.num_chars
        hiddendim = 50
        outputdim = len(self.languages)
        h1 = 130
        h2 = 80
        self.winput = nn.Parameter(inputdim, h1)
        "self.binput = nn.Parameter(1, h1)"
        self.whidden = nn.Parameter(hiddendim, h1)
        "self.bhidden = nn.Parameter(1,h1)"
        self.b1 = nn.Parameter(1, h1)
        self.w2 = nn.Parameter(h1, h2)
        self.b2 = nn.Parameter(1, h2)
        self.w3 = nn.Parameter(h2, hiddendim)
        self.b3 = nn.Parameter(1, hiddendim)
        hinf1 = 27
        self.winf1 = nn.Parameter(hiddendim, hinf1)
        self.binf1 = nn.Parameter(1, hinf1)
        self.winf2 = nn.Parameter(hinf1, outputdim)
        self.binf2 = nn.Parameter(1, outputdim)

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
        "*** YOUR CODE HERE ***"
        i = 0
        for x in xs:
            i += 1
            if i == 1:
                h1 = nn.ReLU(nn.AddBias(nn.Linear(x,self.winput),self.b1))
                h2 = nn.ReLU(nn.AddBias(nn.Linear(h1,self.w2),self.b2))
                h = nn.AddBias(nn.Linear(h2,self.w3),self.b3)
            else:
                h1 = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(x,self.winput),nn.Linear(h,self.whidden)),self.b1))
                h2 = nn.ReLU(nn.AddBias(nn.Linear(h1,self.w2),self.b2))
                h = nn.AddBias(nn.Linear(h2,self.w3),self.b3)
        hinf1 = nn.ReLU(nn.AddBias(nn.Linear(h,self.winf1),self.binf1))
        out = nn.AddBias(nn.Linear(hinf1,self.winf2),self.binf2)
        return out

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
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 150
        m = 9
        multiplier = 0.04*m
        count = 0
        lastvalidacc = 2
        i = 0
        for x, y in dataset.iterate_forever(batch_size):
            i = i + 1
            grads = nn.gradients(self.get_loss(x,y),[self.winput,self.whidden,self.b1,self.w2,self.b2,self.w3,self.b3,self.winf1,self.binf1,self.winf2,self.binf2])
            self.winput.update(grads[0], -1*multiplier)
            self.whidden.update(grads[1], -1*multiplier)
            self.b1.update(grads[2], -1*multiplier)
            self.w2.update(grads[3], -1*multiplier)
            self.b2.update(grads[4], -1*multiplier)
            self.w3.update(grads[5], -1*multiplier)
            self.b3.update(grads[6], -1*multiplier)
            self.winf1.update(grads[7], -1*multiplier)
            self.binf1.update(grads[8], -1*multiplier)
            self.winf2.update(grads[9], -1*multiplier)
            self.binf2.update(grads[10], -1*multiplier)
            validacc = dataset.get_validation_accuracy()
            if 0.70 < validacc < 0.75:
                multiplier = 0.03*m
            if 0.70 < validacc < 0.83:
                multiplier = 0.014*m
            if 0.83 < validacc and lastvalidacc > validacc:
                multiplier = 0.007*m
                count += 1
                if count > 8:
                    break
            lastvalidacc = validacc
