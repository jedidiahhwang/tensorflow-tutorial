import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print (tf.__version__) # Confirm the version of TensorFlow

# This project will analyze fashion data (Fashion MNIST)from 70,000 grayscale images in 10 categories at low resolution (28x28 pixels).
# Load the data from Fashion MNST.
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # Recall this is Python's tuple unpacking syntax to assign multiple variables at once.
# fashion_mnist.load_data() will return two tuples, being assigned respectively in regards to the comma separated tuples on the left.

# The dataset has 10 labels with corresponding classes of clothing. It is sorted in an array.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Start to explore the data below.
# The shape property will return the number of images in the set, with its associated pixel dimensions of 28x28.
print(train_images.shape) # Recall this is the unpackaged tuple in line 12.

# This shows the length of the training set, which is 60,000 (recall that 60,000 are training and 10,000 for testing).
print(len(train_labels))

# Print more about the data and training data.
print(train_labels) # Note the '...' is because this is 60000 labels to print.
print(test_images.shape)
print(len(test_labels))

plt.figure() # Creates a new plot window for visualization -- a blank canvas.
plt.imshow(train_images[0]) # imshow() interprets the array values in the first training image and renders as an image.
plt.colorbar() # Color bar on the right hand of the window.
plt.grid(False) # Removes the grid lines that normally appear.
plt.show() # Display the figure on the screen and actually renders the visualization.

# Normalize the values by dividing by 255.
train_images = train_images / 255
test_images = test_images / 255

# Verify that the data is in the correct format by displaying the first 25 images from the training set.
plt.figure(figsize=(10, 10)) # Create a 10x10 sized window.
for i in range(25): # Loop through the first 25 images in the array.
    plt.subplot(5, 5, i + 1) # Create a 5x5 sub layout, and i + 1 is the position where the image will be placed (remember index starts at 0).
    plt.xticks([]) # Remove the x-axis ticks with an empty array.
    plt.yticks([]) # Remove the y-axis ticks with an empty array.
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary) # cmap=plt.cm.binary sets the color to black and white.
    plt.xlabel(class_names[train_labels[i]]) # Adds respective labels for each item by getting the numeric label i from train_labels and grabbing the associated item from class_names.
plt.show()

# Prior to this, we have been exploring the test data provided to use. Now we will build the model.
# Setup the layers, which represent basic building blocks of a neural network from the data fed to them.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Transforms the format of the images to a 2D array of 28x28.
    tf.keras.layers.Dense(128, activation = 'relu'), # Two Dense layers are created. The first with 128 nodes. The second with 10.
    # Recall that 'relu' means Rectified Linear Unit and represents the kind of activation function. The sigmoid function is another popular alternative.
    tf.keras.layers.Dense(10)
])

# Next, compile the model.
# Optimizer is how the model is updated based on the data it sees and its loss function.
# Loss function measures the accuracy of the model during training.
# Metrics is used to monitor the training and testing steps, using 'accuracy' as its testing metric.
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

# Next we need to train and feed the model.
# 1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
# 2. The model learns to associate images and labels.
# 3. You ask the model to make predictions about a test set—in this example, the test_images array.
# 4. Verify that the predictions match the labels from the test_labels array.
model.fit(train_images, train_labels, epochs = 10) # Should get about 91% accuracy.

# How accurate does the model perform on the test data?
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest Accuracy: ', test_acc)
# Since the test set is less than the training set accuracy, this is overfitting the model. This means the model is poorly suited for new data, and is used to the training data.

# With the model trained, try making some predictions.
# Attach a softmax layer to convert the model's linear outputs, or logits, to probabilities.
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
# Use the model to do some predictions.
predictions = probability_model.predict(test_images)
print(predictions[0]) # Access the return value from predict().
# The 10 numbers represent the "confidence" that the image corresponds to the 10 potential types of clothing. The higher the confidence number, the more likely it's going to be that article of clothing.
print(np.argmax(predictions[0]))
# It thinks that the item is index 9, or an  ankle boot.
print(test_labels[0]) # The first testing data is labeled as an ankle boot, or 9. The model prediction was correct.

# Continue to graph the full set of 10 class predictions.
def plot_image(i, predictions_array, true_label, img):
    '''
    Creates a single image visualization that displays a test image along with the model's prediction results, including confidence scores and color-coded accuracy indicators.
    '''
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array) # Find the highest probability image from the predictions_array.
    if predicted_label == true_label: # If the image is correct, set the color to blue.
        color = 'blue'
    else: # If the image is incorrect, set the color to red.
        color = 'red'

    # Creates a label that includes the predicted class name, the confidence percentage, the true class name, and the blue or red color.
    # The first {} formats the class name.
    # The second {:2.0f}% formats the confidence percentage.
    # The third {} formats the true class name.
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color = color)

def plot_value_array(i, predictions_array, true_label):
    '''
    Creates a bar chart visualization that displays the model's prediction probabilities for all 10 clothing classes, with color-coded bars to highlight the predicted class and true class.
    '''
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777") # Creates a bar chart object.
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red') # thisplot is an object created by plt.bar(), which takes arguments to make the plot object. We then set the appropriate properties using built-in methods like set_color().
    thisplot[true_label].set_color('blue')

# Use our functions to plot and predict the first test image.
i = 0
plt.figure(figsize = (6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Use our functions to plot and predict the 12th test image.
i = 12
plt.figure(figsize = (6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Do the same with several imgaes at the same time now.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize = (2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Now use the trained model against a piece of test data.
img = test_images[1]
print(img.shape)

# Set aside the image in its own batch.
img = (np.expand_dims(img, 0)) # TensorFow/Keras requires a 3D, batch format data input (batch_size, height, width).
print(img.shape)

# Predict the correct label for the test image.
predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation = 45)
plt.show()

# Check the label.
print(np.argmax(predictions_single[0]))