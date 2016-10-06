###Simple ANN
This is a simple 2-hidden-layer neural net which can be used for MNIST.
It has been trained and tested on the full set of alphabetic characters as well, with success. 

#### 2layer.py
This is the network itself.
It requires a whole bunch of packages, so I recommend setting up
a virtual environment to install them in.
Python Pip has all of the packages, so use that.

#### Running the program:
Adjust filenames in the main function to reflect dataset filenames

```
$ python 2layer.py
``` 

to build a new model

```
$ python 2layer.py load
``` 

to skip building and test with the latest model
