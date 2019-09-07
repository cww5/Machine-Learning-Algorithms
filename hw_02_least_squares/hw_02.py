import logging
import argparse
import glob
import sys
import os
from random import random as rand
import time

def dot_product(refw, refx, ncols):
    '''Compute the dot product between two vectors.
    Input: refw=list(vector), refx=list(vector)
    Output: dp=the dot product (scalar value)'''
    dp = 0
    for j in range(ncols):
        dp += refw[j] * refx[j]
    return dp

def data_reader(data_file):
    '''Read data
    Input: data_file=string (file path to the data from pwd)
    Output: data=the content of the data formatted for this script'''
    f = open(data_file)
    data = []
    i = 0
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        data.append([])
        for j in range(len(line)):
            #data[i][j] = line[j]
            data[i].append(line[j])
        data[i].append(1.0) #data[i][len(line)-1] = 1
        #Why does this append 1 to the end?
        i += 1
    f.close()
    return data

def labels_reader(labels_file):
    '''Read labels
    Input: data_file=string (file path to the data from pwd)
    Output: label_lines=the labels content formatted for this script
            max_label=the maximum value of the labels used for initializing class_ array later'''
    f = open(labels_file)
    label_lines = []
    max_label = 0
    c = 0
    for line in f.readlines():
        c += 1
        a = [int(x) for x in line.split()]
        label_lines.append(a)
        if a[1] > max_label:
            max_label = a[1]
    f.close()
    return label_lines, max_label

def class_maker(label_lines, max_label):
    '''Identifies the classes of the input labels
    Input: label_lines=list of labels from labels_reader
           max_label=the label holding the maximum value
    Output: class_=list of classes
            class_size=the count of classes [count(0), count(1)]'''
    class_ = {}
    #class_ = []
    class_size = [0,0]
    #for i in range(max_label+1):
    #    class_.append(0)
    for line in label_lines:
        class_[line[1]] = line[0] #class[label] = class 
        class_size[line[0]] = class_size[line[0]] + 1
        if class_[line[1]] == 0: class_[line[1]] = -1 #is class is 0, reassign to -1
    return class_, class_size        

def w_maker():
    '''Initialize w vector
    Output: w (the vector)'''
    w = []
    for j in range(cols):
        #w.append(0.02 * rand() - 0.01)
        w.append(0.0002 * rand() - 0.0001)
        #w[j] = 0.002 * rand(1) - 0.001
    logging.debug("original w: {}".format(w))
    return w

def gradient_descent(class_, data, w):
    '''Compute Gradient Descent Iteration until stopping difference in errors is < stopping condition
    Input: class_=the content of the classes from class_maker
           data=the data read in from data_reader
           w=the w vector created from w_maker
    Output: delta=the final change in error that stopped the while loop
            error=the final error value'''
    eta = 0.0001
    errors = []
    idx = 0
    delta = 1
    while abs(delta) > 0.001: #stopping_condition
        #compute dell f
        dellf = [0 for j in range(cols)]
        for i in range(rows):
            #if (class_[i] != None):
            if (i in class_): #only process for i values in the class_ dict
                dp = dot_product(w, data[i], cols)
                for j in range(cols):
                    dellf[j] += (class_[i] - dp) * data[i][j]

        #update w
        for j in range(cols):
            w[j] = w[j] + eta * dellf[j]

        error = 0
        #compute error
        for i in range(rows):
            #if (class_[i] != None): ##if defined(class_[i])
            if (i in class_):
                error += (class_[i] - dot_product(w, data[i], cols)) ** 2
        errors.append(error)
        logging.debug("error: {}".format(error))
        if idx == 0:
            idx += 1
            continue
        delta = errors[idx] - errors[idx-1]
        logging.debug("{} - {} = {}".format(errors[idx], errors[idx-1], delta))
        idx += 1
    return delta, error

def calculate_distance(w):
    '''Calculates the distance from w to the plane
    Input: w=the w vector'''
    normw = 0
    for j in range(cols-1): #code has cols-1
        normw += w[j]**2
        logging.debug("w{} = {}".format(j, w[j]))
    logging.info("normw: {}".format(normw))
    normw = normw ** 0.5
    logging.info("||w|| = {}".format(normw))
    d_origin = abs(w[-1] / normw)
    logging.info("distance to origin = {}".format(d_origin))

def predictions(class_, data, w):
    '''Predictions for the data using w
    Input: class_=the content of the classes from class_maker
           data=the data read in from data_reader
           w=the w vector created from w_maker'''
    for i in range(rows):
        #if (class_[i] != None):
        if (i not in class_):
            dp = dot_product(w, data[i], cols)
            if dp > 0:
                logging.info("Class: 1 || Label: {}".format(i))
            else:
                logging.info("Class: 0 || Label: {}".format(i))

def parse_options():
    parser = argparse.ArgumentParser(description="Implement Gradient Descent for Minizing Least Square Loss")
    parser.add_argument('data_file', help='path to the data file')
    parser.add_argument('labels_file', help='path to the labels file')
    parser.add_argument('-vb', '--verbose', action='store_true',help='set verbose logs')
    ret_args = parser.parse_args() 
    fmt_str = "%(asctime)s|%(levelname)s|Line %(lineno)d\t: %(message)s"
    if ret_args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt_str)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt_str)
    return ret_args

if __name__ == "__main__":
    args = parse_options()
    data_filepath, labels_filepath = args.data_file, args.labels_file
    logging.debug("data file path: {}".format(data_filepath))
    logging.debug("labels file path: {}".format(labels_filepath))
    data_content = data_reader(data_filepath)
    global rows, ref, cols
    rows = len(data_content)
    ref = data_content[0]
    cols = len(ref)
    label_content, maximum_label = labels_reader(labels_filepath)
    classes, class_sizes = class_maker(label_content, maximum_label)
    w_vector = w_maker()
    delta_err, error_val = gradient_descent(classes, data_content, w_vector)
    logging.info("delta = {}".format(delta_err))
    logging.info("error = {}".format(error_val))
    calculate_distance(w_vector)
    predictions(classes, data_content, w_vector)



