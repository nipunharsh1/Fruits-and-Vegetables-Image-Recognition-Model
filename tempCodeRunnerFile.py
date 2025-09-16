# Test TensorFlow/Keras
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Test Keras (built into TensorFlow)
from tensorflow import keras
print(f"Keras version: {keras.__version__}")

# Test other packages
import numpy as np
from PIL import Image
import flask
print("All packages installed successfully!")
