import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
import math

GRASS = 0
COW = 1
SKY = 2
FARM = 3

CLASSES = [GRASS, COW, SKY]


def build_samples(filename, class_value=FARM):
    image = np.asarray(Image.open(filename))
    samples = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    predictions = np.full(samples.shape[0], fill_value = class_value)
    return samples, predictions

def confusion_matrix(predictions, f_X):
    conf_matrix = np.zeros((len(CLASSES), len(CLASSES)))
    for prediction, truth in zip(predictions, f_X):
        conf_matrix[int(truth)][int(prediction)] += 1

    return conf_matrix

def get_precision(conf_matrix):
    sum_columns = np.sum(conf_matrix, axis=0)
    diagonal = np.diagonal(conf_matrix)
    return np.mean(diagonal/sum_columns)


def svm_classify_image(image, X_train, f_X_train, c=1.0, kernel='linear'):
    samples, _ = build_samples(image, class_value = FARM)

    svm_classifier = svm.SVC(C=c, kernel=kernel)
    svm_classifier.fit(X_train, f_X_train)
    predicted = svm_classifier.predict(samples)
    
    image_array = np.asarray(Image.open(image))    
    result_image = np.array(list(map(class_color, predicted))).reshape(image_array.shape)

    classified_img = Image.fromarray(result_image.astype(np.uint8))
    classified_img.save("classified_pixels.png")


def plot_matrix(matrix, filename):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Reds)

    for row, col in np.ndindex(matrix.shape):
        discr = matrix[row][col]
        ax.text(row, col, str(discr), va='center', ha='center')

    plt.xlabel("Estimacion")
    plt.ylabel("Real")
    plt.savefig(f'{filename}', bbox_inches='tight')
    plt.close()


def svm_get_precision(X_train, f_X_train, X_test, f_X_test, c, kernel='linear', gamma='0.01'):
    print('before fit')
    svm_classifier = svm.SVC(C=c, kernel=kernel)
    svm_classifier.fit(X_train, f_X_train)
    print('after fit')
    
    test_predicted = svm_classifier.predict(X_test)

    test_confusion_matrix = confusion_matrix(test_predicted, f_X_test)
    
    plot_matrix(test_confusion_matrix, f'test_kernel_{kernel}_c_{c}_g_{gamma}.png')
    
    test_precision = get_precision(test_confusion_matrix)

    return test_precision

def class_color(class_name):
    if class_name == GRASS:
        return (0, 255, 0)
    if class_name == COW:
        return (179, 104, 0)
    if class_name == SKY:
        return (0, 204, 255)

def plot_radial_results():

    # format of data is c_value, kernel=linear, test_precision, train_precision
    data = np.genfromtxt('radial_precision.csv', delimiter=',')

    gamma_values = []
    precisions = []
    for d in data:
        precisions.append(float(d[2]))
        gamma_values.append(float(d[0]))
    # plt.legend()

    # plt.legend()

    points = np.arange(len(precisions))
    width = 0.8

    low = min(precisions)
    high = max(precisions)
    plt.ylim([0.94, 1.0])
    plt.bar(points, precisions, width, label='gammas', color='blue')

    plt.title("Precisión vs. Gamma en kernel radial")
    plt.xlabel("Gamma")
    plt.ylabel("Precisión")
    # plt.legend()

    # plt.ytick(ticks=np.arange)
    
    plt.xticks(ticks=np.arange(len(gamma_values)), labels=gamma_values)

    plt.savefig("radial_precisions.png")
    plt.close()


def plot_kernel_results():


    # format of data is c_value, kernel=linear, test_precision, train_precision
    data = np.genfromtxt('kernel_precision.csv', delimiter=',')
    # kernels = ['linear', 'polinomial', 'radial', 'sigmoide']
    kernels = ['linear', 'polinomial', 'radial']
    precisions = []
    for d in data:
        precisions.append(float(d[2]))

    # plt.legend()

    points = np.arange(len(precisions))
    width = 0.8

    low = min(precisions)
    high = max(precisions)
    plt.ylim([0.98, 1.0])
    plt.bar(points, precisions, width, label='kernels', color='blue')

    plt.title("Precisión vs. Kernel")
    plt.xlabel("Kernel")
    plt.ylabel("Precisión")
    # plt.legend()

    # plt.ytick(ticks=np.arange)
    
    plt.xticks(ticks=np.arange(len(kernels)), labels=kernels)

    plt.savefig("kernel_precisions.png")
    plt.close()



def plot_c_results():
    # format of data is c_value, kernel=linear, test_precision, train_precision
    data = np.genfromtxt('c_precision_old.csv', delimiter=',')
    # kernels = ['linear', 'polinomial', 'radial', 'sigmoide']
    c_values = []
    precisions = []
    for d in data:
        precisions.append(float(d[3]))
        c_values.append(float(d[1]))
    # plt.legend()

    points = np.arange(len(precisions))
    width = 0.8

    # low = min(precisions)
    # high = max(precisions)
    plt.ylim([0.98, 1.0])
    plt.bar(points, precisions, width, label='valores c', color='blue')

    plt.title("Precisión vs. C")
    plt.xlabel("Valor C")
    plt.ylabel("Precisión")
    # plt.legend()

    # plt.ytick(ticks=np.arange)
    
    plt.xticks(ticks=np.arange(len(c_values)), labels=c_values)

    plt.savefig("c_precisions.png")
    plt.close()

