#Connor Watson
#HW 05 - CART Decision Tree Algorithm
#10/28/2018

import argparse
import sys
import os

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
    class_d = {}
    class_size = [0,0]
    for line in label_lines:
        class_d[line[1]] = line[0]
        class_size[line[0]] = class_size[line[0]] + 1
    return class_d, class_size 

def gini_selection(data, labels):
    '''Traverse all columns and output column and threshold
    with lowest gini index.
    Input: data=matrix of data points
           labels=dictionary of labels
    Output: gini=lowest gini index
            c=had best split
            split=best split'''
            
    ginivals = [[0, 0] for j in range(cols)]
    temp, c, s = 0, 0, 0
    #c=col with best split
    #s=best split

    for j in range(cols):
        listcol = [item[j] for item in data]
        keys = sorted( range( len(listcol) ), key=lambda col: listcol[col])
        listcol = sorted(listcol)
        ginis = []
        prevrow = 0
        #find the value that gives the best gini split
        #of the data into a partition of 2 sets
        for k in range(1,rows):
            #left partition size, right partition size
            lsize, rsize = k, (rows - k)
            #proportion of -1 labels in left/right partition
            lp, rp = 0, 0
            for l in range(k):
                if (labels.get(keys[l]) == 0):
                    lp += 1
            for r in range(k, rows):
                if (labels.get(keys[r]) == 0):
                    rp += 1
            #used to evaluate gini of a split
            gini = (lsize / rows) * (lp / lsize) * (1 - lp / lsize) + (rsize / rows) * (rp / rsize) * (1 - rp / rsize)
            ginis.append(gini)
            if (ginis[k - 1] == float(min(ginis))):
                ginivals[j][0] = ginis[k - 1]
                ginivals[j][1] = k
        if (j == 0):
            #for the first column, get the ginival
            temp = ginivals[j][0]
        if (ginivals[j][0] <= temp):
            #update best ginival
            temp = ginivals[j][0]
            c = j
            s = ginivals[j][1]
            if (s != 0):
                s = (listcol[s] + listcol[s - 1]) / 2
    print("gini index: {}\ncolumn with best split: {}\nbest split: {}".format(temp,c,s))

def parse_options():
    parser = argparse.ArgumentParser(description="Implement CART Decision Tree Algorithm")
    parser.add_argument("data_file", help="path to the data file")
    parser.add_argument("labels_file", help="path to the labels file")
    ret_args = parser.parse_args()
    #fmt_str = "%(asctime)s|%(levelname)s|Line %(lineno)d\t: %(message)s"
    return ret_args

if __name__ == "__main__":
    args = parse_options()
    data_filepath, labels_filepath = args.data_file, args.labels_file
    print("data file path: {}".format(data_filepath))
    print("labels file path: {}".format(labels_filepath))
    data_content = data_reader(data_filepath)
    global rows, cols
    rows = len(data_content)
    cols = len(data_content[0])
    label_content, maximum_label = labels_reader(labels_filepath)
    classes, class_sizes = class_maker(label_content, maximum_label)
    gini_selection(data_content, classes)
    
