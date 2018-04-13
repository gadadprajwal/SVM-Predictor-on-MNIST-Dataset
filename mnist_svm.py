import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from random import random,shuffle,sample
import numpy as np
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot


def extract_testing_data(mnist):
    return mnist.test.images,mnist.test.labels

def extract_training_data(mnist, num):
    image_train,label_train = [],[]
    train_labels = []
    num = int(num / 10)
    for i in range(0,10):
        for j in range(len(mnist.train.images)):
            if mnist.train.labels[j][i] == 1:
                image_train.append(mnist.train.images[j])
                label_train.append(mnist.train.labels[j])
            if len(image_train) == num*(i+1):
                break
    train = list(zip(image_train,label_train))
    shuffle(train)
    image_train, label_train = zip(*train)
    for label in label_train:
        temp = 0
        for i in range(len(label)):
            if label[i] == 1:
                temp =i
                break
        train_labels.append(temp)
    label_train = train_labels
    return image_train, label_train


def get_model(train_images, train_labels):
    clf = LinearSVC(max_iter = 5000)
    model = clf.fit(train_images, train_labels)
    return model

def accuracy_using_svmpredict(model, test_images, test_labels):
    accuracy = 0
    for i in range(len(test_images)):
        if test_labels[i][model.predict([test_images[i]])] == 1:
            accuracy += 1
    return accuracy/len(test_images)*100


def accuracy_without_svmpredict(model, test_images, test_labels):
    accuracy = 0
    for image,labels in zip(test_images,test_labels):
        prediction = np.dot(model.coef_, image) + model.intercept_
        if prediction.argmax() == labels.argmax():
            accuracy += 1
    return accuracy/len(test_images) * 100




def pca(train_images,dimension,get_reduced=False, eigen_vectors = False):
    pca = PCA(n_components = dimension, svd_solver = 'auto')
    train_images_reduced = pca.fit_transform(train_images)
    new_train_images = pca.inverse_transform(train_images_reduced)
    square_error = 0
    for i in range(len(train_images)):
        temp = 0
        for j in range(len(train_images[0])):
            temp += (new_train_images[i][j] - train_images[i][j])*(new_train_images[i][j] - train_images[i][j])
        square_error += temp
    if eigen_vectors:
        return pca.components_
    if not get_reduced:
        return square_error/len(train_images)
    else:
        return new_train_images

def error_plot(train_images):
    X = np.logspace(np.log10(1.0),np.log10(500.0),num=50)
    Y = []
    for x in X:
        Y.append(pca(train_images,int(x)))
    plot.plot(X,Y)
    plot_name="libsvm_error_dimension.png"
    plot.savefig(plot_name)

def accuracy_plot(train_images, train_labels, test_images, test_labels):
    X = [2,5,10,20,30,50,70,100,150,200,250,300,400,748]
    Y = []
    for x in X:
        new_train_images = pca(train_images, x, get_reduced = True)
        reduced_model = get_model(new_train_images, train_labels)
        Y.append(accuracy_using_svmpredict(reduced_model, test_images, test_labels))
    plot.plot(X,Y)
    plot_name = "libsvm_accuracy_dimension.png"
    plot.savefig(plot_name)

def get_nn_model(train_images, train_labels):
    clf = MLPClassifier(max_iter = 200)
    model = clf.fit(train_images, train_labels)
    return model

def accuracy_using_nnpredict(model, test_images,test_labels):
    accuracy = 0
    for i in range(len(test_images)):
        if test_labels[i][model.predict([test_images[i]])] == 1:
            accuracy += 1
    return accuracy/len(test_images)*100

def nn_accuracy(train_images, train_labels, test_images, test_labels):
    X = [2,5,10,20,30,50,70,100,150,200,250,300,400,748]
    Y = []
    for x in X:
        new_train_images = pca(train_images, x, get_reduced = True)
        reduced_model = get_nn_model(new_train_images, train_labels)
        Y.append(accuracy_using_nnpredict(reduced_model, test_images, test_labels))
    plot.plot(X,Y)
    plot_name = 'nn_accuracy_dimension.png'
    plot.savefig(plot_name)

def visualize_eigen_vectors(train_images):
    eigen_vectors = pca(train_images, 500, eigen_vectors = True)[:10]
    for i in range(len(eigen_vectors)):
        temp = []
        for j in range(0,len(eigen_vectors[i]),28):
            temp.append(eigen_vectors[i][j:j+28])
        plt = plot.subplot(2,5,i+1)
        plot.axis('off')
        plot.imshow(temp,'gray_r')
    plot.savefig('eigen_vectors.png')


def main():
    print("Importing MNIST Dataset")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_images,train_labels = extract_training_data(mnist,1000)
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    test_images, test_labels = extract_testing_data(mnist)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)
    print("a.Training MINST dataset on SVM and accuracy using svmpredict\nb.Accuracy calculations without using svmpredict\nc.PCA to reduce dimensionality to 50\nd.SVM Accuracy with lower dimension space  and visualization of eigen vectors\ne.SVM plot Accuracy vs. Dimension\nf.Neural Network Accuracy with lower dimension space\ng.NN Accuracy plot vs dimensions\n")
    model = get_model(train_images, train_labels)
    choice = input("***Enter your choice***")
    if choice == 'a':
        print("Accuracy using SVM on 1000 training data",accuracy_using_svmpredict(model,test_images, test_labels))
    elif choice == 'b':
        print("Accuracy using SVM without using svmpredict",accuracy_without_svmpredict(model, test_images, test_labels))
    elif choice == 'c':
        error_plot(train_images)
        print("plotted mean-square-error vs dimension")
    elif choice == 'd':
        new_train_images = pca(train_images, 50, get_reduced = True)
        reduced_model = get_model(new_train_images, train_labels)
        print("Accuracy of SVM with reduced dimensions",accuracy_using_svmpredict(reduced_model, test_images, test_labels))
        visualize_eigen_vectors(train_images)
        print("Plotted top 10 eigen vectors")
    elif choice == 'e':
        accuracy_plot(train_images, train_labels, test_images, test_labels)
        print("Plotted accuracy vs dimensions")
    elif choice == 'f':
        new_train_images = pca(train_images, 50, get_reduced = True)
        reduced_model = get_nn_model(new_train_images, train_labels)
        print("Accuracy using NN with reduced dimensions",accuracy_using_nnpredict(reduced_model, test_images, test_labels))
    elif choice == 'g':
        nn_accuracy(train_images, train_labels, test_images, test_labels)
        print("Plotted accuracy vs dimesions using NN")


if __name__=="__main__":
    main()
