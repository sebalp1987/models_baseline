import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_image(image_path, size):
    """
    Load an image as a Numpy array.
    :param image_path:  Path of the image
    :param size:        Target size
    :return             Image array, normalized between 0 and 1
    """
    image = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(image_path, target_size=size)) / 255.
    return image


def process_predictions(class_probabilities, class_readable_labels, k=2):
    """
    Process a batch of predictions from our estimator.
    :param class_probabilities:     Prediction results returned by the Keras classifier for a batch of data
    :param class_readable_labels:   List of readable-class labels, for display
    :param k:                       Number of top predictions to consider
    :return                         Readable labels and probabilities for the predicted classes
    """
    topk_labels, topk_probabilities = [], []
    for i in range(len(class_probabilities)):
        # Getting the top-k predictions:
        topk_classes = sorted(np.argpartition(class_probabilities[i], -k)[-k:])

        # Getting the corresponding labels and probabilities:
        topk_labels.append([class_readable_labels[predicted] for predicted in topk_classes])
        topk_probabilities.append(class_probabilities[i][topk_classes])

    return topk_labels, topk_probabilities


def display_predictions(images, topk_labels, topk_probabilities):
    """
    Plot a batch of predictions.
    :param images:                  Batch of input images
    :param topk_labels:             String labels of predicted classes
    :param topk_probabilities:      Probabilities for each class
    """
    num_images = len(images)
    num_images_sqrt = np.sqrt(num_images)
    plot_cols = plot_rows = int(np.ceil(num_images_sqrt))

    figure = plt.figure(figsize=(13, 10))
    grid_spec = gridspec.GridSpec(plot_cols, plot_rows)

    for i in range(num_images):
        img, pred_labels, pred_proba = images[i], topk_labels[i], topk_probabilities[i]
        # Shortening the labels to better fit in the plot:
        pred_labels = [label.split(',')[0][:20] for label in pred_labels]

        grid_spec_i = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=grid_spec[i],
                                                       hspace=0.1)

        # Drawing the input image:
        ax_img = figure.add_subplot(grid_spec_i[:2])
        ax_img.axis('off')
        ax_img.imshow(img)
        ax_img.autoscale(tight=True)

        # Plotting a bar chart for the predictions:
        ax_pred = figure.add_subplot(grid_spec_i[2])
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.spines['bottom'].set_visible(False)
        ax_pred.spines['left'].set_visible(False)
        y_pos = np.arange(len(pred_labels))
        ax_pred.barh(y_pos, pred_proba, align='center')
        ax_pred.set_yticks(y_pos)
        ax_pred.set_yticklabels(pred_labels)
        ax_pred.invert_yaxis()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import os
    from resources import STRING
    from config.config import global_params

    input_shape = global_params.get('input_shape')
    valid = os.path.join(STRING.data_input_path, STRING.data_input_path, 'val')
    test_images = np.asarray([load_image(file) for file in valid])
    print('Test Images: {}'.format(test_images.shape))

    image_batch_low_quality = tf.image.resize(test_images, input_shape)
    model = tf.keras.models.load_model(STRING.root_path + '/training/model.h5')
    predictions = model.predict(image_batch_low_quality)
    print('Predicted class probabilities: {}'.format(predictions.shape))

    class_readable_labels = ['NORMAL', 'PNEUMONIA']
    top5_labels, top5_probabilities = process_predictions(predictions, class_readable_labels)

    display_predictions(test_images, top5_labels, top5_probabilities)