#Connor Watson
#HW 07 - Bagging on the HW06 Decision Stump
#11/11/2018

import argparse
import sys
import os
from random import randint as rand

#this will store lists of all the predictions needed for the labels
preds = {}

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
            data[i].append(line[j])
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
    #max_label = 0
    for line in f.readlines():
        a = [int(x) for x in line.split()]
        label_lines.append(a)
        #if a[1] > max_label:
        #    max_label = a[1]
    f.close()
    return label_lines#, max_label

def class_maker(label_lines):
    '''Identifies the classes of the input labels
    Input: label_lines=list of labels from labels_reader
           max_label=the label holding the maximum value
    Output: y=list of classes
            class_size=the count of classes [count(0), count(1)]'''
    class_d = {}
    class_size = [0,0]
    for line in label_lines:
        class_d[line[1]] = line[0]
        class_size[line[0]] = class_size[line[0]] + 1
        #if class_d[line[1]] == 0: class_d[line[1]] = -1 #if class is 0, reassign to -1
    return class_d, class_size

def filter_training_data(data, labels):
    global preds 
    row_indeces = []  #these are the row indeces of labeled data
    total_pres = 0
    nrow = len(data)
    for i in range(nrow):
        if i not in labels:
            preds[i] = {0:0,1:0}
            total_pres += 1
        else:
            row_indeces.append(i)
    #print("{} labels need predicting".format(total_pres))
    #print("{} labeled data".format(len(labeled_data)))
    #print("{} original data".format(len(data)))
    return row_indeces
            
def bag_data(data, indeces, labels):
    '''Produces one bag of data for the classified training dataset.'''
    nrow, ncol = len(data), len(data[0])
    new_data = []
    new_labs = {}
    cur = 0
    while(len(new_data) < len(data)):
        row_idx = indeces[rand(0,len(indeces)-1)]
        if labels.get(row_idx) == None: #this is just in case there is an error
            print("Unexpected bagged data (unclassified) row {}".format(row_idx))
            continue
        new_data.append(data[row_idx])
        new_labs[cur] = labels[row_idx]
        cur += 1
    return new_data, new_labs
        
def gini_selection(data, labels):
    '''Traverse all columns and output column and threshold
    with lowest gini index.
    Input: data=matrix of data points
           labels=dictionary of labels
    Output: gini=lowest gini index
            c=had best split
            split=best split'''
    
    nrow, ncol = len(data), len(data[0])
    ginivals = [[0, 0] for j in range(ncol)]
    temp, c, s = 0, 0, 0
    #c=col with best split
    #s=best split
    #print(data)
    for j in range(ncol):
        #for column j, grab the element in each row
        listcol = [item[j] for item in data]
        keys = sorted( range( len(listcol) ), key=lambda col: listcol[col])
        listcol = sorted(listcol)  #sort the elements in the "column"
        #print(listcol)
        #print(keys)
        ginis = []
        prevrow = 0
        #find the value that gives the best gini split
        #of the data into a partition of 2 sets
        for k in range(1,nrow):
            #left partition size, right partition size
            lsize, rsize = k, (nrow - k)
            #proportion of -1 labels in left/right partition
            lp, rp = 0, 0
            for l in range(k):
                if (labels.get(keys[l]) == 0):
                    lp += 1
            for r in range(k, nrow):
                if (labels.get(keys[r]) == 0):
                    rp += 1
            #used to evaluate gini of a split
            gini = float((lsize / nrow) * (lp / lsize) * (1 - lp / lsize) + (rsize / nrow) * (rp / rsize) * (1 - rp / rsize))
            ginis.append(gini)
            if (ginis[k - 1] == float(min(ginis))):
                ginivals[j][0] = ginis[k - 1]
                ginivals[j][1] = k
            #print("col:{} | gini:{} | lp:{} | rp:{} | lsize:{} | rsize:{}".format(j, gini, lp, rp, lsize, rsize))
        if (j == 0):
            #for the leftmost column, get the ginival
            temp = ginivals[j][0]
        if (ginivals[j][0] <= temp):
            #update best (minimum) ginival
            temp = ginivals[j][0]
            c = j
            s = ginivals[j][1]
            if (s != 0):
                s = float((listcol[s] + listcol[s - 1]) / 2)
            ##print("col:{} | split:{} | gini:{}".format(c,s))
        #print("gini:{} | lp:{} | rp:{} | lsize:{} | rsize:{}".format(gini, lp, rp, lsize, rsize))

    left_count, right_count = 0, 0
    left_label, right_label = 0, 0
    for i in range(nrow):
        if labels.get(i) != None:
            if data[i][c] < s: #for all points left of the split
                if labels[i] == 0: #check if more 0 or 1 labels exist
                    left_count += 1 
                else:
                    right_count += 1

    if left_count > right_count:
        right_label = 1
    else:
        left_label = 1

    print("gini index: {}\ncolumn with best split: {}\nbest split: {}".format(temp,c,s))
    return c, s, left_label, right_label

def tally_predictions(col, split, data, labels, left, right):
    global preds
    nrow = len(data)
    for i in range(nrow):
        point = data[i][col]
        if labels.get(i) == None:
            #print("{} < {} : {}".format(point, split, point<split))
            if point < split:
                preds[i][left] += 1  #neg tally will be left classified
            else:
                preds[i][right] += 1 #left tally will be right classified

def print_predictions():
    global preds
    actual = {}
    for key in preds:
        if preds[key][0] > preds[key][1]:
            print("{} {}".format(key, 0))
            actual[key] = 0
        else:
            print("{} {}".format(key, 1))
            actual[key] = 1
    #return actual

def compare_predictions(ap, labels_path):
    f = open(labels_path)
    d = {}
    for line in f:
        l = line.split()
        d[int(l[1])] = int(l[0])
    f.close()
    num_wrong = 0
    num_correct = 0
    for key in ap:
        if ap[key] == d[key]:
            num_correct += 1
        else:
            num_wrong += 1
    print("error: {}/{} = {}".format(num_wrong, len(ap), 100 * num_wrong/len(ap)))
    
def parse_options():
    parser = argparse.ArgumentParser(description="Bagging on the HW06 Decision Stump")
    parser.add_argument("data_file", help="path to the data file")
    parser.add_argument("labels_file", help="path to the training labels file")
    parser.add_argument("--labs", help="path to the labels file")
    ret_args = parser.parse_args()
    return ret_args

if __name__ == "__main__":
    args = parse_options()
    data_filepath, labels_filepath = args.data_file, args.labels_file    
    #print("data file path: {}".format(data_filepath))
    #print("labels file path: {}".format(labels_filepath))
    data_content = data_reader(data_filepath)
    label_content = labels_reader(labels_filepath)
    classes, class_sizes = class_maker(label_content)
    training_indeces = filter_training_data(data_content, classes)
    
    for i in range(101):
        print("_______iteration:{}________".format(i))
        bag, bag_labs = bag_data(data_content, training_indeces, classes)
        best_col, best_split, leftlab, rightlab = gini_selection(bag, bag_labs)
        tally_predictions(best_col, best_split, data_content, classes, leftlab, rightlab)

    print_predictions()
    #if not args.labs:
    #    pass
    #else:
    #    compare_predictions(actual_preds, args.labs)
    
