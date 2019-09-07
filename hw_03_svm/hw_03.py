#Connor Watson
#HW 03 - SVM with Hinge Loss
#10/7/2018

import argparse
import glob
import sys
import os
from random import random as rand
import time

def dot_product(refw, refx):
    '''Compute the dot product between two vectors.
    Input: refw=list(vector), refx=list(vector)
    Output: dp=the dot product (scalar value)'''
    dp = 0
    for j in range(cols):
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
    for line in f.readlines():
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
    Output: y=list of classes
            class_size=the count of classes [count(0), count(1)]'''
    y = {}
    class_size = [0,0]
    #for i in range(max_label+1):
    #    y.append(0)
    for line in label_lines:
        y[line[1]] = line[0]
        class_size[line[0]] = class_size[line[0]] + 1
        if y[line[1]] == 0: y[line[1]] = -1 #if class is 0, reassign to -1
    return y, class_size        

def w_maker():
    '''Initialize w vector
    Output: w (the vector)'''
    w = []
    for j in range(cols):
        #w.append(0.02 * rand() - 0.01)
        w.append(0.0002 * rand() - 0.0001)
        #w[j] = 0.002 * rand(1) - 0.001
    print("original w: {}".format(w))
    return w

def gradient_descent(y, x, w):
    '''Compute Gradient Descent Iteration until stopping difference in errors is < stopping condition
    Input: y=the content of the classes from class_maker
           data=the data read in from data_reader
           w=the w vector created from w_maker
    Output: delta=the final change in error that stopped the while loop
            error=the final error value'''
    
    eta = 0.001
    idx = 0
    delta = 1
    prev_obj, obj = 100, 50
    while abs(prev_obj - obj) > .001: #stopping_condition
        #compute dell f
        dellf = [0 for j in range(cols)]
        for i in range(rows):
            if (i in y):
                dp = dot_product(w, x[i])
                if y[i] * ( dp ) < 1:
                    for j in range(cols):
                        #derivative of error function
                        dellf[j] += x[i][j] * y[i]

        #update w
        for j in range(cols):
            w[j] = w[j] + eta * dellf[j]
            
        error = 0
        #compute error
        for i in range(rows):
            if (i in y):
                #objective
                magw = sum([elem**2 for elem in w]) ** 0.5
                r = dot_product(w, x[i])
                error += max([0,(1 - (y[i] * r)/magw)])
        prev_obj = obj
        obj = error
        print("Objective: {}".format(obj))
        if idx == 0:
            idx += 1
            continue
        #print("{} - {} = {}".format(errors[idx], errors[idx-1], delta))
        idx += 1
        #time.sleep(0.25)
    return obj

def calculate_distance(w):
    '''Calculates the distance from w to the plane
    Input: w=the w vector'''
    normw = 0
    for j in range(cols-1): #code has cols-1
        normw += w[j]**2
    print("w = {}".format(str(w)))
    print("normw: {}".format(normw))
    normw = normw ** 0.5
    print("||w|| = {}".format(normw))
    d_origin = abs(w[-1] / normw)
    print("distance to origin = {}".format(d_origin))

def predictions(y, x, w):
    '''Predictions for the data using w
    Input: class_=the content of the classes from class_maker
           data=the data read in from data_reader
           w=the w vector created from w_maker'''
    for i in range(rows):
        if (i not in y):
            dp = dot_product(w, x[i])
            if dp > 0:
                print("Class: 1 || Label: {}".format(i))
            else:
                print("Class: 0 || Label: {}".format(i))

def parse_options():
    parser = argparse.ArgumentParser(description="Implement Gradient Descent for Minizing Least Square Loss")
    parser.add_argument('data_file', help='path to the data file')
    parser.add_argument('labels_file', help='path to the labels file')
    ret_args = parser.parse_args()
    fmt_str = "%(asctime)s|%(levelname)s|Line %(lineno)d\t: %(message)s"
    return ret_args

if __name__ == "__main__":
    args = parse_options()
    data_filepath, labels_filepath = args.data_file, args.labels_file
    print("data file path: {}".format(data_filepath))
    print("labels file path: {}".format(labels_filepath))
    data_content = data_reader(data_filepath)
    global rows, ref, cols
    rows = len(data_content)
    ref = data_content[0]
    cols = len(ref)
    label_content, maximum_label = labels_reader(labels_filepath)
    classes, class_sizes = class_maker(label_content, maximum_label)
    w_vector = w_maker()
    obj_val = gradient_descent(classes, data_content, w_vector)
    print("Objective = {}".format(obj_val))
    calculate_distance(w_vector)
    predictions(classes, data_content, w_vector)



