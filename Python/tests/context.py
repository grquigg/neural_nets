import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural.nn import NeuralNetwork
from neural.utils import softmax, sigmoid, relu