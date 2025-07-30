import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Import the MNIST dataset. MNIST stands for Modified National Institute of Standards and Technology database.
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Load the dataset.
x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize the data by dividing by 255.0, since pixel values are between 0 and 255.

# Next, build a sequential machine learning model.
# A sequential model is a linear stack of layers where each layer has exactly one input tensor and one output tensor.
model = tf.keras.models.Sequential([ 
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten the input image of shape 28x28 into a 1D array of 784 elements.
    tf.keras.layers.Dense(128, activation='relu'), # Dense layer with 128 neurons and ReLU activation function.
    tf.keras.layers.Dropout(0.2), # Dropout layer to prevent overfitting.
    tf.keras.layers.Dense(10, activation='softmax') # Dense layer with 10 neurons and softmax activation function.
]) # This is a classic neural network architecture for image classification.

# Train the model.
# Returns an array of logits, or raw predictions that a classification model generates.
predictions = model(x_train[:1]).numpy()
print(predictions)

# The tf.nn.softmax function converts these logits to probabilities for each class.
tf.nn.softmax(predictions).numpy()
# Separating tf.nn.softmax from the previous line is a common way to improve numerical stability for loss calculations.

# Define a loss function for training using losses.SparseCategoricalCrossentropy.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())
# The untrained model gives probabilities close to random (1/10 for each class), so the loss is close to -tf.math.log(1/10) ~= 2.3.

# Before training, the model needs to be compiled.
# When compiling, you need to specify the optimizer, loss function, and metrics.
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Use the Model.fit method to adjust the model parameters and minimize the loss.
model.fit(x_train, y_train, epochs=5)
# 1. model.fit() is the core method in training.
# 2. It takes x_train and y_train, which are the normalized input pixels values and corresponding labels respectively.
# 3. epochs=5 means that the model will process the entire training dataset 5 times.
# 4. This method returns a training history object containing matrices for each epoch.
# When you run this code, note the loss and accuracy calculations for each epoch.

# Check the model using a validation set or test set.
model.evaluate(x_test, y_test, verbose=2) # Setting verbose to 2 allows for detailed output and a progress bar.
# Based on the output, the accuracy of the model is about 97.78%.

# To return a probability, you can wrap the trained model and attach the softmax to it.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
probability_model(x_test[:5])