#Connor Watson

import argparse
import glob
import sys
import os 
global debug
debug = False

def data_reader(data_file):
    '''This function takes in a path to data and returns the formatted data, with num rows and cols'''
    f = open(data_file)
    lines = f.readlines()
    f.close()
    ret_data = []
    for line in lines:
        if line == '':
            continue
        nums = [float(num) for num in line.split()]
        ret_data.append(nums)
    num_rows, num_cols = len(ret_data), len(ret_data[0])
    print("Data Rows: {} x Cols: {}".format(num_rows, num_cols))
    return ret_data, num_rows, num_cols

def labels_reader(labels_file):
    '''This function takes in a path to labels and returns the formatted labels, with counts of the labels'''
    f = open(labels_file)
    train_labels = {}
    counts = {}
    lines = f.readlines()
    f.close()
    for line in lines:
        #Each line is formatted as such: [value(label) key(idk)]
        if line == '':
            continue
        lin = line.split()
        train_labels[int(lin[1])] = int(lin[0])
        if int(lin[0]) not in counts:
            counts[int(lin[0])] = 1
        else:
            counts[int(lin[0])] += 1
    print("Label Counts 0:{}, 1:{}".format(counts[0],counts[1]))
    if debug:
        print("____________actual labels____________")
        for key in train_labels:
            print("Label {}: Class {}".format(key, train_labels[key]))
        print("_____________________________________")
    return train_labels, counts

def compute_stats(in_data, rows, cols, in_labels, l_counts):
    '''This function computes the means and standard deviations'''
    m0, m1 = [], []
    s0, s1 = [], []
    for i in range(cols):
        m0.append(1)
        m1.append(1)
        s0.append(1)
        s1.append(1)
    for i in range(rows):
        if in_labels.get(i) != None and in_labels[i] == 0:
            for j in range(cols):
                m0[j] = m0[j] + in_data[i][j]
        if in_labels.get(i) != None and in_labels[i] == 1:
            for j in range(cols):
                m1[j] = m1[j] + in_data[i][j]
    for j in range(cols):
        m0[j] = m0[j]/l_counts[0]
        m1[j] = m1[j]/l_counts[1]
    #Calculate variance
    for i in range(rows):
        if in_labels.get(i) != None and in_labels[i] == 0:
            for j in range(cols):
                s0[j] += (in_data[i][j] - m0[j])**2
        if in_labels.get(i) != None and in_labels[i] == 1:
            for j in range(cols):
                s1[j] += (in_data[i][j] - m1[j])**2
    #Calculate std dev
    for j in range(cols):
        s0[j] = s0[j] ** 0.5
        s1[j] = s1[j] ** 0.5
    if debug:
        print(m0)
        print(m1)
        print(s0)
        print(s1)
    return m0, m1, s0, s1

def classify(in_data, rows, cols, in_labels, m0, m1, s0, s1):
    '''This function classifies the unlabeled points'''
    classifications = {}
    print("____________predictions____________")
    for i in range(rows):
        if in_labels.get(i) == None:
            d0,d1 = 0,0
            for j in range(cols):
                d0 = d0 + ((in_data[i][j] - m0[j])/s0[j]) ** 2
                d1 = d1 + ((in_data[i][j] - m1[j])/s1[j]) ** 2
            if (d0 < d1):
                print("Label {}: Class 0".format(i))
                classifications[i] = 0
            else:
                print("Label {}: Class 1".format(i))
                classifications[i] = 1
    print("____________________________________")
    return classifications

def accuracy(predicted_labels, true_labels):
    print("Any incorrect predictions will be printed:")
    correct = 0
    for i in predicted_labels:
        if predicted_labels[i] == true_labels[i]:
            correct += 1
        else:
            print("Label:{} | Prediction:{} | Actual:{}".format(i, predicted_labels[i],true_labels[i]))
    print("Accuracy: {}%".format(float(correct/len(predicted_labels)) * 100))

def parse_options():
    parser = argparse.ArgumentParser(description="Implement Naive Bayes Classifier")
    parser.add_argument('data_file', help='path to the data file')
    parser.add_argument('labels_file', help='path to the labels file')
    parser.add_argument('--all', help='path to the file of all labels')
    #Note that breast_cancer.labels contains all labels
    #but traininglabels #0-9 are the test data
    parser.add_argument('-vb', '--verbose', action='store_true',help='set verbose logs')
    ret_args = parser.parse_args() 
    if ret_args.verbose: debug = True
    else: debug = False
    return ret_args

if __name__ == "__main__":
    args = parse_options()
    data_filepath, labels_filepath = args.data_file, args.labels_file
    if debug:
        print("data file path: {}".format(data_filepath))
        print("labels file path: {}".format(labels_filepath))
    #Reading in the data - lists of 30 floats
    data, nrows, ncols = data_reader(data_filepath)
    #t_labels_d = {} #{ traininglabels.N : { means:[],std:[],label_counts:{},classifications:{} } } }
    
    #Reading in the labels
    training_labels, label_counts = labels_reader(labels_filepath)
    #Computing means
    means0, means1, devs0, devs1 = compute_stats(data, nrows, ncols, training_labels, label_counts)
    #Classify unlabeled points
    classified_labels = classify(data, nrows, ncols, training_labels, means0, means1, devs0, devs1)
    if args.all:
        all_labels, all_label_counts = labels_reader(args.all)
        accuracy(classified_labels, all_labels)
