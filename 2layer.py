__author__ = "Carl-Philip Majgaard"
import numpy as np
from data import Data
from scipy import special
import pickle
import sys
import numexpr as ne
import enchant

class Settings:
    input_dimension = 128 # input layer dimensionality
    output_dimension = 26  # output layer dimensionality
    eta = 15  # Gradient descent learning rate
    lastloss = 999999 #not for editing

def sigmoid(p):
    return special.expit(p)

def dsigmoid(y):
    return ne.evaluate("y * (1.0 - y)")

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):

    num_examples = len(X)  # training set size
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = ne.evaluate("exp(z3)")
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    with np.errstate(divide='ignore'):
        corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    result = predict(model, X)
    incorrects = np.sum(np.not_equal(y, result))
    print "######"
    print "Losses"
    print float(incorrects)/num_examples
    print "Ratio is: %f" %((1. / num_examples * data_loss)/ (float(incorrects)/num_examples),)
    loss = 1. / num_examples * data_loss

    return loss

#Bold Driver learning rate optimization
#Returns true if our loss is higher than last iteration
#Cuts learning rate in half
def bold_driver(model, X, y):
    num_examples = len(X)  # training set size
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = ne.evaluate("exp(z3)")
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    with np.errstate(divide='ignore'):
        corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)


    loss = 1. / num_examples * data_loss

    if loss > Settings.lastloss + 10.e-10:
        Settings.eta = Settings.eta/2
        Settings.lastloss = loss
        return True
    else:
        Settings.eta = Settings.eta * 1.05
        Settings.lastloss = loss
        return False


def predict(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = ne.evaluate("exp(z3)")
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)



def build_nn(X, y, nn_hdim, num_passes=20000, print_loss=False):
    num_examples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(Settings.input_dimension, nn_hdim) / np.sqrt(Settings.input_dimension)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_hdim))
    W3 = np.random.randn(nn_hdim, Settings.output_dimension) / np.sqrt(nn_hdim)
    b3 = np.zeros((1, Settings.output_dimension))
    model = {}

    # Gradient descent for batches
    for i in range(0, num_passes):

        # Forward propagate
        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        a2 = sigmoid(z2)
        z3 = a2.dot(W3) + b3
        exp_scores = np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagate
        delta4 = probs
        delta4[range(num_examples), y] -= 1
        dW3 = (a2.T).dot(delta4)
        db3 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = delta4.dot(W3.T) * dsigmoid(a2)
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * dsigmoid(a1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Gradient descent update
        eta = Settings.eta
        W1 += ne.evaluate("-(eta / num_examples) * dW1")
        b1 += ne.evaluate("-(eta / num_examples) * db1")
        W2 += ne.evaluate("-(eta / num_examples) * dW2")
        b2 += ne.evaluate("-(eta / num_examples) * db2")
        W3 += ne.evaluate("-(eta / num_examples) * dW3")
        b3 += ne.evaluate("-(eta / num_examples) * db3")

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

        if bold_driver(model, X, y) == True:

            W1 += ne.evaluate("(eta / num_examples) * dW1")
            b1 += ne.evaluate("(eta / num_examples) * db1")
            W2 += ne.evaluate("(eta / num_examples) * dW2")
            b2 += ne.evaluate("(eta / num_examples) * db2")
            W3 += ne.evaluate("(eta / num_examples) * dW3")
            b3 += ne.evaluate("(eta / num_examples) * db3")
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}



        # Print our losses every so often
        if print_loss == True and i % 100 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))
            # print W1
    return model

def confusion_matrix(truecats, classcats ):
    '''Takes in two Nx1 matrices of zero-index numeric categories and
    computes the confusion matrix. The rows represent true
    categories, and the columns represent the classifier output.

    '''
    unique1, mapping = np.unique( np.array(truecats.T), return_inverse=True)
    unique2, mapping = np.unique( np.array(classcats.T), return_inverse=True)

    unique1 = unique1.tolist()
    unique2 = unique2.tolist()

    unique1 += unique2
    unique = np.unique(np.array(unique1)).tolist()

    confmatrix = np.matrix(np.zeros((len(unique), len(unique))))
    print confmatrix.shape
    print truecats.shape[0]
    for i in range(truecats.shape[0]):
        confmatrix[truecats[i,0],classcats[i,0]] += 1

    return confmatrix

def confusion_matrix_str(cmtx):
    '''Takes in a confusion matrix and returns a string suitable for printing.'''

    s = '%10s|' %("Classif.->")
    for i in range(cmtx.shape[1]):
        s += "%9dC" %(i,)
    s += "\n"
    for i in range(cmtx.shape[0]):
        s += "%9dT|" %(i,)
        for j in range(cmtx.shape[1]):
            s += "%10d" %(cmtx[i,j],)
        s+="\n"
    return s

def shorttest_chars():

    #LETTER DATASET
    print "Loading Dataset"
    letters = Data("orderedletters.csv")
    n_samples = letters.getRawNumRows()
    print n_samples

    print "Extracting Data"
    letterMatrix = letters.getData(letters.getHeaders())

    print "Splitting into Data and Classes"
    letterData = np.asarray(letterMatrix[:,1:])
    rawLetterCodes = letterMatrix[:,0]

    letterCodes = np.asarray(rawLetterCodes.ravel().astype(int))[0]
    print letterCodes

    if(len(sys.argv) > 1 and sys.argv[1] == "load"):
        print "Loading pickle"
        model = pickle.load( open( "shortmodel.p", "rb" ) )
    else:
        print "Building Model"
        model = build_nn(letterData[(n_samples-1000):], letterCodes[(n_samples-1000):], 100, print_loss=True, num_passes= 1000)
        pickle.dump( model, open( "shortmodel.p", "wb" ) )

    result = predict(model, letterData[:(n_samples-1000)])

    codesforconf = np.matrix(letterCodes[:(n_samples-1000)].reshape((n_samples-1000, 1)))
    classforconf = np.matrix(result.reshape((n_samples-1000, 1)))

    confm = confusion_matrix(codesforconf, classforconf)
    print(confusion_matrix_str(confm))

def test_chars():

    #LETTER DATASET
    print "Loading Dataset"
    letters = Data("orderedletters.csv")
    n_samples = letters.getRawNumRows()
    print n_samples

    print "Extracting Data"
    letterMatrix = letters.getData(letters.getHeaders())

    print "Splitting into Data and Classes"
    letterData = np.asarray(letterMatrix[:,1:])
    print letterData
    rawLetterCodes = letterMatrix[:,0]

    letterCodes = np.asarray(rawLetterCodes.ravel().astype(int))[0]
    print letterCodes

    if(len(sys.argv) > 1 and sys.argv[1] == "load"):
        print "Loading pickle"
        model = pickle.load( open( "nnsave.p", "rb" ) )
    else:
        print "Building Model"
        model = build_nn(letterData, letterCodes, 100, print_loss=True, num_passes=14000)
        pickle.dump( model, open( "nnsave.p", "wb" ) )

    result = predict(model, letterData)

    codesforconf = np.matrix(letterCodes.reshape((n_samples, 1)))
    classforconf = np.matrix(result.reshape((n_samples, 1)))

    confm = confusion_matrix(codesforconf, classforconf)
    print(confusion_matrix_str(confm))


def test_sentence():
    # Sentence DATASET
    # All characters loaded here have not been seen by the
    # network and do not exist in letters.csv
    print "Loading Dataset"
    letters = Data("quickbrown.csv")
    n_samples = letters.getRawNumRows()

    print "Extracting Data"
    letterMatrix = letters.getData(letters.getHeaders())

    print "Splitting into Data and Classes"
    letterData = np.asarray(letterMatrix[:,1:])
    rawLetterCodes = letterMatrix[:,0]

    letterCodes = np.asarray(rawLetterCodes.ravel().astype(int))[0]
    print letterCodes

    print "Loading pickle"
    model = pickle.load( open( "nnsave.p", "rb" ) )

    result = predict(model, letterData)

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    dict = {}

    for idx, letter in enumerate(alphabet):
        dict[idx] = letter

    words = ""

    for letter in result:
        words += dict[letter]

    print words

def main():
    #shorttest_chars()
    test_chars()
    test_sentence()


if __name__ == "__main__":
    main()
