import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import backend as k
import matplotlib.pyplot as plt
import random

def euclidean_distance(vects):
  '''
    This method return the distance between f(Pi) and f(Pj).
    Formula: (D(Pi, Pj)^2)
    Argument:
      vects -- a tensor which consists of two vectors (a pair of images).
    Rrturns:
      a vector of scalars (a single number).
  '''
  i, j = vects
  # k.sum(x, axis, keepdims) will return the sum of x
  sum_square = k.sum(k.square(i - j), axis=1, keepdims=True)
  return k.sqrt(k.maximum(sum_square, k.epsilon()))


def eucl_dist_output_shape(shapes):
    '''
      Return a tuple as the output shape to the function.
    '''
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''
    Arguments:
      y_true -- a numpy array that is given labels of the pairs.
        0 = imgaes of the pair come from same class. 
        1 = images of the pair come from different classes.
      y_pred -- distance of a images pair by using Euclidean distance function.
    Returns:
      loss -- real number, value of the loss.
    '''
    margin = 0.8
    square_pred = k.square(y_pred)
    margin_square = k.square(k.maximum(margin - y_pred, 0))
    return k.mean((1 - y_true) * square_pred + y_true * margin_square)


def create_pairs(x, digit_indices, dataset):
  '''
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    Arguments:
      x -- a numpy array with images.
      digit_indices -- the arrays of specific labels which consist of indices of images.
      dataset -- the classes of specific labels.
    Returns:
      a combined numpy array of positive pairs as well as negative pair of images.
      a numpy array which consists of the labels of pairs.
  '''
  pairs = []
  labels = []
  # n = the minimum length of an array among all of the classes
  n = min([len(digit_indices[d]) for d in range(len(dataset))]) - 1

  for d in range(len(dataset)):
      for i in range(n):
          z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
          # store two images that is in the same class as the positive pair.
          pairs += [[x[z1], x[z2]]]
          # randomly generate number of the dataset as an axis 
          inc = random.randrange(1, len(dataset))
          # the negative pair should not consists of the same images
          dn = (d + inc) % len(dataset)
          z1, z2 = digit_indices[d][i], digit_indices[dn][i]
          pairs += [[x[z1], x[z2]]]
          labels += [0, 1]
  return np.array(pairs), np.array(labels)


def Create_Base_Network(input_shape):
  '''
    Create a sharable base network for two input datasets by using CNN.
    Argument:
      input_shape -- the shape of the input will pass into this method.
      4D tensor with shape: (batch, rows, cols, channels)
    Returns:
      a built model with the input shape as well as the constructed layers.
      shape = (batch, new_rows, new_cols, filters)
  '''
  ip = keras.Input(shape=input_shape)

  # The 'relu' parameter is used to replace all negative values by zero, relu is abbreviated as Rectified Linear Unit
  # padding="same" results in padding the input such that the output has the same length as the original input.
  layer1 = keras.layers.Conv2D(16, kernel_size=(3,3), padding='same', activation='relu')(ip)
  # extract the features of the image
  layer1 = keras.layers.MaxPooling2D(pool_size=(2,2))(layer1)
  layer1 = keras.layers.Dropout(0.1)(layer1)
  layer2 = keras.layers.Conv2D(32, kernel_size=(2,2), padding='same', activation='relu')(layer1)
  layer2 = keras.layers.MaxPooling2D(pool_size=(2,2))(layer2)
  layer3 = keras.layers.Conv2D(64, kernel_size=(2,2), padding='same', activation='relu')(layer2)
  layer3 = keras.layers.MaxPooling2D(pool_size=(2,2))(layer3)
  layer3 = keras.layers.Flatten()(layer3)
  layer3 = keras.layers.Dense(64, activation='relu')(layer3)
  layer3 = keras.layers.Dropout(0.2)(layer3)
  output = keras.layers.Dense(128, activation='relu')(layer3)
  return keras.Model(ip, output)


def compute_accuracy(y_true, y_pred):
  '''
    Compute classification accuracy with a fixed threshold on distances.
    Used during evaluation.
    Arguments:
      y_true -- a numpy array that is given labels of the pairs.
      y_pred -- distance of a images pair by using Euclidean distance function.
    Returns:
      return a number represents the mean of distances of pairs.
  '''
  # if the number of distance between two images is greater than 0.5 (the edge)
  # then the pair is negative and replace the distance of the images as the label 1
  pred = y_pred.ravel() > 0.5
  return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
  '''
    Compute classification accuracy with a fixed threshold on distances.
    Used during training process.
    Arguments:
      y_true -- a numpy array that is given labels of the pairs.
      y_pred -- distance of a images pair by using Euclidean distance function.
    Returns:
      return a number of accuracy that the pair labels compare to the distance of a pair.
  '''
  # k.cast() casts a tensor to a different dtype and returns it.
  # k.equal() Element-wise equality between two tensors. and returns a bool tensor.
  return k.mean(k.equal(y_true, k.cast(y_pred > 0.5, y_true.dtype)))

def history_displayed(history, train_monitor, test_monitor):
  '''
    This function returns the historgram of the monitor (the taget).
    Arguments:
      hisotry -- the history of the training results.
      train_monitor -- the target that will be used as the training in the plot. 
      test_monitor -- the target that will be used as the testing  in the plot.
  '''
  plt.plot(history.history[train_monitor])
  plt.plot(history.history[test_monitor])
  plt.title('Model {}'.format(train_monitor))
  plt.ylabel(train_monitor)
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

def loss_testing_displayed(test_var_1, test_var_2):
  '''
    This function is used to evaluate the performance of the loss function.
    Arguemts:
      test_var_1 -- the vector which is indicated as the first input image.
      test_var_2 -- the vector which is indicated as the second input image.
  '''
  test_var_1 = tf.constant(test_var_1, shape=(3,3), dtype='float')
  test_var_2 = tf.constant(test_var_2, shape=(3,3), dtype='float')
  # Implement the method of measuring distance between to images
  # the images here are used as a pair
  dis = euclidean_distance([test_var_1,test_var_2])
  # return the number of distance
  # provide the cusomted labels for the pair
  pos_result = contrastive_loss(0, dis)
  neg_result = contrastive_loss(1, dis)
  with tf.Session() as sess:
    loss_pos_val = sess.run(pos_result)
    loss_neg_val = sess.run(neg_result)
    print('Contrastive loss of Positive Pair computed with tensorflow{:10.4f}'.format(loss_pos_val))
    print('Contrastive loss of Negative Pair computed with tensorflow', loss_neg_val)

def Load_Dataset():
  '''
    This method is used to download the Fashion Mnist dataset from keras library,
    and classifying the data in order to match the criteria of the assignment.
    Returns:
      the processed two datasets which will be used in the following procedure.
  '''
  # Create new dataset in order to store different classes of images and labels
  Set1_imgs, Set1_labels = [], []
  Set2_imgs, Set2_labels = [], []
  
  # Download the dataset from keras.fashion and give names for the downloaded data
  (train_imgs, train_labels), (test_imgs, test_labels) = keras.datasets.fashion_mnist.load_data()

  # Combine both images and labels of training data and testing data
  imgs = np.concatenate((train_imgs, test_imgs))
  labels = np.concatenate((train_labels, test_labels))
  print('Original intact dataset of images:', imgs.shape)
  print('Original intact dataset of labels:', labels.shape)
  
  # Create new dataset in order to store different classes of images and labels
  Set1_imgs, Set1_labels = [], []
  Set2_imgs, Set2_labels = [], []
    
  # Classify both images and labels with different classes
  for data in range(0, len(imgs)):
    if labels[data] in labels1:
      Set1_imgs.append(imgs[data])
      Set1_labels.append(labels[data])
    else:
      Set2_imgs.append(imgs[data])
      Set2_labels.append(labels[data])           
  
  # Since the initial type of the arrays are python arrays
  # Convert into numpy arrays
  Set1_imgs = np.array(Set1_imgs)
  Set1_labels = np.array(Set1_labels)
  Set2_imgs = np.array(Set2_imgs)
  Set2_labels = np.array(Set2_labels)
  
  # Normalize data dimensions so that they are of approximately the same scale
  Set1_imgs = Set1_imgs.astype('float32')
  Set2_imgs = Set2_imgs.astype('float32')
  Set1_imgs /= 255.
  Set2_imgs /= 255.
  
  return Set1_imgs, Set1_labels, Set2_imgs, Set2_labels

def images_displayed(class_names, images, labels):
  '''
    This method is to display the first 25 images.
    Arguments:
      class_names -- a set of classes.
      images -- a set of images.
      labels -- a set of labels.
  '''
  plt.figure(figsize=(10,10))
  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels[i]])
  plt.show()

def pairs_displayed(images_pair):
  '''
    Display the pairs of each class.
    Arguments:
      images_pair -- the pair of images which will be plotted.
  '''
  fig, ax = plt.subplots(nrows=10, ncols=4,figsize=(40, 40))
  idx = 0
  for row in range(10):
      idx = random.randrange(0,len(images_pair),2)
      ax[row,0].imshow(images_pair[idx][0],cmap = 'gray')
      ax[row,1].imshow(images_pair[idx][1],cmap = 'gray')
      idx+=1
      ax[row,2].imshow(images_pair[idx][0],cmap = 'gray')
      ax[row,3].imshow(images_pair[idx][1],cmap = 'gray')
  plt.show()

def accuracy_displayed(input_pairs, pairs_labels, set_name):
  '''
    This method displays the outcome of evaluating accuracy of each dataset.
    Arguments:
      input_pairs -- an array which consists of sets of pairs.
      pairs_labels -- an array which consists of labels of paris.
      set_name -- a string that represents the dataset. 
  '''
  y_pred = model.predict(input_pairs)
  acc = compute_accuracy(pairs_labels, y_pred)
  print('* Accuracy on %s: %0.2f%%' % (set_name ,100 * acc))
  return acc
  

  
  
  
if __name__ == '__main__':
  
  # define the names of classes
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  # Set 1 is the images with labels for classes "top", "trouser", "pullover", "coat", "sandal", "andke boot"
  labels1 = [0, 1, 2, 4, 5, 9]

  # Set 2 is the images with labels for classes "dress", "sneaker", "bag", "shirt"
  labels2 = [3, 6, 7, 8]
  
  # Loaded the keras fashion mnist dataset
  Set1_imgs, Set1_labels, Set2_imgs, Set2_labels = Load_Dataset()
  
  # Verification of the classified datasets
  print('Below is the first set of dataset with six labels')
  images_displayed(class_names, Set1_imgs, Set1_labels)
  
  print('Below is the first set of dataset with six labels')
  images_displayed(class_names, Set2_imgs, Set2_labels)
  
  # Distribute training and testing datasets with 80/20 percent
  train_imgs, test_imgs, train_labels, test_labels = train_test_split(Set1_imgs, Set1_labels, test_size=0.2, random_state=87)
  
  # verification of allocating the set1 with 80% and 20% as training and testing data
  print('The shape of training dataset:', train_imgs.shape)
  train_amount = train_imgs.size / (train_imgs.size + test_imgs.size)
  print('The allocation of traning data:', 100 * train_amount, '%')
  
  print('The shape of testing dataset:', test_imgs.shape)
  test_amount = test_imgs.size / (train_imgs.size + test_imgs.size)
  print('The allocation of testing data:', 100 * test_amount, '%')
  
  # the rows and columnsof the images are at position 1 and 2
  img_rows, img_cols = train_imgs.shape[1:]
  input_shape = (img_rows, img_cols, 1)
  batch_size = 256
  epochs = 20
  
  # create positive and negative pairs of training dataset
  digit_indices = [np.where(train_labels == i)[0] for i in labels1]
  tr_pairs, tr_y = create_pairs(train_imgs, digit_indices, labels1)

  # create positive and negative pairs of testing dataset
  digit_indices = [np.where(test_labels == i)[0] for i in labels1]
  te_pairs, te_y = create_pairs(test_imgs, digit_indices, labels1)
  
  # Reshape the input arrays to 4D (batch_size, rows, columns, channels)
  tr_pairs = tr_pairs.reshape(tr_pairs.shape[0], 2, img_rows, img_cols, 1)
  te_pairs = te_pairs.reshape(te_pairs.shape[0], 2, img_rows, img_cols, 1)

  # Create positive and negative pairs of test 3
  digit_indices = [np.where(Set2_labels == i)[0] for i in labels2]
  test3_pairs, test3_y = create_pairs(Set2_imgs, digit_indices, labels2)
  test3_pairs = test3_pairs.reshape(test3_pairs.shape[0], 2, img_rows, img_cols, 1)
  
  # Testing with pairs from the set of images with labels ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"] union ["dress", "sneaker", "bag", "shirt"]
  test2_pairs = np.concatenate((te_pairs, test3_pairs))
  test2_y = np.concatenate((te_y, test3_y))

  
  print('set 1')
  print(te_pairs.shape, te_pairs.dtype)
  print(te_y.shape, te_y.dtype)
  print('set 2')
  print(test2_pairs.shape, test2_pairs.dtype)
  print(test2_y.shape, test2_y.dtype)
  print('set 3')
  print(test3_pairs.shape, test3_pairs.dtype)
  print(test3_y.shape, test3_y.dtype)
  
  
  # Testing the accuracy of loss function
  test_var_1 = [0, 0, 0]
  test_var_2 = [1, 1, 1]
  loss_testing_displayed(test_var_1, test_var_2)
  
  train_accs, test1_accs, test2_accs, test3_accs = [], [], [], []
  
  for row in range(5):
    print('Iteration {}'.format(row+1))
    
    # Implement the CNN with both inputs
    base_network = Create_Base_Network(input_shape)
    base_network.summary()

    # Generate two input shape with Keras.Input()
    input_a = keras.Input(shape=input_shape)
    input_b = keras.Input(shape=input_shape)

    # Put two inputs into the same weight of NN
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Create the distance method by implementing Lambda()
    distance = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    model = keras.Model([input_a, input_b], distance)
    # Distribute training and testing datasets with 80/20 percent
    train_pairs, valid_pairs, train_labels, valid_labels = train_test_split(tr_pairs, tr_y, train_size=0.8, random_state=87)

    # Display the structure of the CNN
    model.summary()

    # Compile the model
    rmsprop = keras.optimizers.RMSprop(learning_rate=1e-3, decay=1e-3/epochs)
    # Configures the model for training.
    model.compile(loss=contrastive_loss, optimizer=rmsprop, metrics=[accuracy])

    # Using EarlyStopping method to stop testing when the validation of accuracy is not decreased.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='auto', verbose=0, patience=10)
    callbacks_list = [early_stop]
    
    # Trains the model for a fixed number of epochs (iterations on a dataset).
    history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, validation_data=([valid_pairs[:, 0], valid_pairs[:, 1]], valid_labels))

    #Plot training & validation accuracy values
    history_displayed(history, 'accuracy', 'val_accuracy')

    # Plot training & validation loss values
    history_displayed(history, 'loss', 'val_loss')

    # Testing with pairs from the set of images with labels ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"]
    train_acc = accuracy_displayed([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, 'Training Dataset')
    train_accs.append(train_acc)
    
    test1_acc = accuracy_displayed([te_pairs[:, 0], te_pairs[:, 1]], te_y, 'Test Dataset 1')
    test1_accs.append(test1_acc)

    ##### Evaluate the model of test 2 #####
    test2_acc = accuracy_displayed([test2_pairs[:, 0], test2_pairs[:, 1]], test2_y, 'Test Dataset 2')
    test2_accs.append(test2_acc)

    ##### Evaluate the model of test 3 #####
    test3_acc = accuracy_displayed([test3_pairs[:, 0], test3_pairs[:, 1]], test3_y, 'Test Dataset 3')
    test3_accs.append(test3_acc)
    
    k.clear_session()
  
  x = np.arange(1,6)
  plt.plot(x, train_accs, 'o-', label='train_acc')
  plt.plot(x, test1_accs, 'v-', label='Acc of Set1')
  plt.plot(x, test2_accs, 's-', label='Acc of Set1 ∪ Set2')
  plt.plot(x, test3_accs, 'D-', label='Acc of Set2')  
  plt.title('TESTING GENERALISATION CAPABILITY')
  plt.ylabel('Accuracy')
  plt.xlabel('Iteration')
  plt.legend(loc='upper left')
  plt.show()