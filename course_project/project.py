import argparse
import sys
import os
from random import randint as rand
from math import floor

verbose = False
module_description = '''

Author: Connor Watson
Date: 11/29/2018

Term Project combining Feature Selection with Classification.

This is the term project we were assigned.
In the data set, each column can be a value in
{0,1,2}, and labels are either {0,1}.

The training data has dimensions 8000 x 29623, but we are
to select 15 features (and their neighbors), because
the rest are noise. 4000 rows are cases, 4000 are controls.

The test data has dimensions 2000 x 29623, but we don't know
the true labels. Only the instructor does.

Cross validation may be used to evaluate accuracy of the
proposed method, which should be about 63%.

The input via command line is:
training dataset
training labels
test dataset

The output:
1) predictions of the labels of the test dataset in the format
row label
2) total number of features
3) feature column numbers used for final prediction
--if all features were used, output "ALL"

The program should be able to save the predicted labels for
each row in a csv, as well as a separate csv containing the
features.

Restrictions:
Not allowed - numpy, scipy, any feature selection libraries
Allowed - C programs (svmlight, liblinear, fest, bmrm,), modules for
svm, logistic regression, naive bayes, linear regression,
dimensionality reduction

'''

#this will store lists of all the predictions needed for the labels
preds = {}

def construct_predictions_dictionary(data):
    '''This takes in a data set and formats the global preds variable
    to be ready to store the tallies of predictions.
    Input: data=list of data
    Output: formats preds as such:
        {key:value} = {(row_index):{0:(tally_0), 1:(tally_1)}}'''
    global preds 
    nrow = len(data)
    for i in range(0, nrow):
        preds[i] = {0:0,1:0}    

def data_reader(data_file):
    '''Read data
    Input: data_file=string (file path to the data from pwd)
    Output: data=the content of the data as a list of lists, where
            each sublist is a row of data'''
    f = open(data_file)
    data = []
    i = 0
    while True:
        line = f.readline()
        if line == "":
            break
        line = [int(x) for x in line.split()]
        data.append([])
        for j in range(0, len(line)):
            data[i].append(line[j])
        i += 1
    f.close()
    if verbose:
        print("Data read")
    return data

def labels_reader(labels_file):
    '''Read labels
    Input: data_file=string (file path to the data from pwd)
    Output: label_lines=the labels content formatted for this script
            max_label=the maximum value of the labels used for initializing class_ array later'''
    f = open(labels_file)
    label_lines = []
    for line in f.readlines():
        a = [int(x) for x in line.split()]
        label_lines.append(a)
    f.close()
    if verbose:
        print("Labels read")
    return label_lines

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
    if verbose:
        print("Labels Dictionary Made")
    return class_d, class_size

def get_neighbors(chi_cols, num_cols):
    '''Helper function for chi_squared_test
    Input: chi_cols=list of 15 cols extracted from chi squared test
           num_cols=total number of columns from the data
    Output: all column indeces and their neighbors' indeces'''
    new_cols = []
    for col in chi_cols:
        if col not in new_cols:
            new_cols.append(col)
        neighbors = []
        if col - 1 > 0:
            neighbors.append(col - 1)
        if col + 1 < num_cols:
            neighbors.append(col + 1)
        for nei in neighbors:
            if nei not in new_cols:
                new_cols.append(nei)
    new_cols = sorted(new_cols)
    return new_cols
        
def chi_squared_test(data, labels, num_features):
    '''Chi Squared Feature Selection
    Input: data=list of rows, where each row is a list of values
           labels=dictionary of labels and row values
           num_features=int number of features desired
    Output: col_nums=list of indeces for the columns that "pass the test" '''
    rows = len(data)
    cols = len(data[0])
    chi2_column_values = []
    for j in range(0, cols):
        #Build contingency table per column (observed values)
        #Contingency Table structure:
        #Labels>>    0   1
        #Data:    0 |  |  |
        #         1 |  |  |
        #         2 |  |  |
        #https://stats.stackexchange.com/questions/7152/how-should-you-handle-cell-values-equal-to-zero-in-a-contingency-table/7160
        #To avoid structural zeroes, 1 represents no observed value for that data value : label pair
        observed_values = [[1,1],[1,1],[1,1]]
        for i in range(0, rows):
            data_point = data[i][j]
            label_val = labels[i]
            if label_val == 0:
                if data_point == 0:
                    observed_values[0][0] += 1
                elif data_point == 1:
                    observed_values[1][0] += 1
                elif data_point == 2:
                    observed_values[2][0] += 1
            elif label_val == 1:
                if data_point == 0:
                    observed_values[0][1] += 1
                elif data_point == 1:
                    observed_values[1][1] += 1
                elif data_point == 2:
                    observed_values[2][1] += 1
        row_totals = [1,1,1]
        col_totals = [1,1]
        for r in range(0, len(observed_values)):
            row = observed_values[r]
            row_totals[r] += sum(row)
            for c in range(0, len(row)):
                col_totals[c] += row[c]
        total = sum(col_totals) #sum(row_totals) is also acceptable
        #total represents the total amount of observations for all data points
        #regardless of label
        expected_values = [[1,1],[1,1],[1,1]]
        for r in range(0, len(row_totals)):
            row = row_totals[r]
            for c in range(0, len(col_totals)):
                col = col_totals[c]
                expected_values[r][c] = (float(row)*float(col)) / float(total)
        chi2_values = []
        for r in range(0, len(expected_values)):
            for c in range(0, len(expected_values[0])):
                obs, exp = observed_values[r][c], expected_values[r][c]
                chi2_value = ((float(obs) - float(exp))**2) / float(exp)
                chi2_values.append(chi2_value)                
        chi_squared = sum(chi2_values)
        chi2_column_values.append([j, chi_squared])
    sorted_by_chi2 = sorted(chi2_column_values, key=lambda tup: tup[1], reverse=True)
    col_nums = sorted_by_chi2[:num_features]
    desired_chi_cols = [pair[0] for pair in col_nums]
    desired_cols = get_neighbors(desired_chi_cols, cols)
    if verbose:
        print("chi squared complete")
    return desired_cols

def feature_extraction(data, features):
    '''Extracts data based on selected features
    Input: data=training data
           features=the selected 15 features and their neighbors
    Output: reduced_data=the data set based on the selected features'''
    reduced_data = []
    for i in range(0, len(data)):
        reduced_data.append([data[i][f_idx] for f_idx in features])
    if verbose:
        print("Reduced data to: {} x {}".format(len(reduced_data), len(reduced_data[0])))
    return reduced_data
        
def bag_data(data, labels):
    '''Produces one bag of data for the classified training dataset
    Input: data=list of data
           labels=dictionary of the row labels and indeces
    Output: bagged_data=a list equal in length to data, but is a bagged set of data
            bagged_labels=dictionary containing labels for each row in bagged_data'''
    nrow, ncol = len(data), len(data[0])
    indeces = list(range(0, nrow))
    bagged_data = []
    bagged_labs = {}
    cur = 0 #represents the current index of the row of bagged data
    while(len(bagged_data) < len(data)):
        row_idx = indeces[rand(0,len(indeces)-1)]
        if labels.get(row_idx) == None: #this is just in case there is an error
            print("Unexpected bagged data (unclassified) row {}".format(row_idx))
            continue
        bagged_data.append(data[row_idx])
        bagged_labs[cur] = labels[row_idx]
        cur += 1
    return bagged_data, bagged_labs
        
def gini_selection(data, labels):
    '''Traverse all columns and output column and threshold
    with lowest gini index.
    Input: data=matrix of data points
           labels=dictionary of labels
    Output: gini=lowest gini index
            c=had best split
            split=best split'''
    nrow, ncol = len(data), len(data[0])
    ginivals = [[0.0, 0.0] for j in range(int(ncol))]
    temp, c, s = 0, 0, 0
    #c=col with best split
    #s=best split
    #print(data)
    for j in range(ncol):
        #for column j, grab the element in each row
        listcol = [float(item[j]) for item in data]
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
            lsize, rsize, lp, rp = float(lsize), float(rsize), float(lp), float(rp)
            #used to evaluate gini of a split
            gini = float(   (lsize / float(nrow)) * (lp / lsize) * (1.0 - lp / lsize)    +    (rsize / float(nrow)) * (rp / rsize) * (1.0 - rp / rsize))
            ginis.append(gini)
            if (float(ginis[k - 1]) == float(min(ginis))):
                ginivals[j][0] = ginis[k - 1]
                ginivals[j][1] = k
            #print("col:{} | gini:{} | lp:{} | rp:{} | lsize:{} | rsize:{}".format(j, gini, lp, rp, lsize, rsize))
        if (j == 0):
            #for the leftmost column, get the ginival
            temp = ginivals[j][0]
        if (ginivals[j][0] < temp): ##this used to be <
            #update best (minimum) ginival
            temp = ginivals[j][0]
            c = j
            s = ginivals[j][1]
            if (s != 0):
                s = float( (float(listcol[s]) + float(listcol[s - 1])) / 2.0)
            #print("col:{} | split:{} | gini:{}".format(c,s, gini))
        #print("gini:{} | lp:{} | rp:{} | lsize:{} | rsize:{}".format(gini, lp, rp, lsize, rsize))
    if s == 0:
        s = 0.01
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
        
    if verbose:
        print("gini index: {}\ncolumn with best split: {}\nbest split: {}".format(temp,c,s))
    return c, s, left_label, right_label

def tally_predictions(col, split, data, labels, left, right):
    '''Tallies the predictions in preds according to the split found from gini selection.
    Input: col=integer column with best split
           split=float best split for the decision stump
           data=list of data rows
           labels=dictionary of labels for the data
           left=label for the left side of the stump
           right=label for the right side of the stump
    Output: Tallies total predictions in global variable preds'''
    global preds
    nrow = len(data)
    for i in range(0, nrow):
        point = data[i][col]
        if labels.get(i) == None:
            #print("{} < {} : {}".format(point, split, point<split))
            if point < split:
                preds[i][left] += 1  #neg tally will be left classified
            else:
                preds[i][right] += 1 #left tally will be right classified

def print_predictions(file_location, write_file = False, cv = True):
    '''Writes predictions and outputs them to the screen
    Input: file_location=string location of output file (with name)
           write_file=boolean True to represent we will write to the output file
           cv=boolean False when this is not running during cross validation
    Output: actual=dictionary of row labels with their predicted label'''
    global preds
    f = ""
    if write_file and not cv:
        f = open(file_location, "w")
    actual = {}
    keys = sorted(list(preds.keys()))
    for key in keys:
        if preds[key][0] > preds[key][1]:
            if not cv:
                print("{} {}".format(0, key))
            if write_file and not cv:
                f.write("{} {}\n".format(0, key))
            actual[key] = 0
        else:
            if not cv:
                print("{} {}".format(1, key))
            if write_file and not cv:
                f.write("{} {}\n".format(1, key))
            actual[key] = 1
    if write_file and not cv:
        f.close()
    return actual
    
def get_accuracy(preds_made, actual_labels):
    '''Computes accuracy of predictions during cross validation
    Input: preds_made=dictionary of predictions
           actual_labels=trainlabels
    Output: accuracy=float (num correct)/(total num rows) * 100  '''
    keys = sorted(list(preds_made.keys()))
    num_correct = 0
    for key in keys:
        if preds_made[key] == actual_labels[key]:
            num_correct += 1
    accuracy = (num_correct / len(keys)) * 100
    if verbose:
        print("After cross validation, accuracy is: {:.2f}%".format(accuracy))
    return accuracy

def cross_validate(data, all_labels):
    '''Performs cross validation of the train set by repeating the following 5 times:
    >>bag the data, find the stump, tally predictions
    Input: filtered_data=list of all the data after selecting the top features
           all_labels=dictionary of train labels
    Output: resets preds to be empty 
            filtered_data=list of rows with selected features
            features_and_neighbors=list of features'''
    global preds
    tot_rows = len(data)
    nrows = floor(tot_rows * 0.70)
    trainrows = data[:nrows]
    testrows = data[nrows:]
    #print(len(trainrows))
    #print(len(testrows))
    #print(tot_rows)
    
    top_features = 15 #there may actually be more because of the neighbors
    features_and_neighbors = chi_squared_test(trainrows, classes, top_features)
    filtered_data = feature_extraction(trainrows, features_and_neighbors)
    filtered_test = feature_extraction(testrows, features_and_neighbors)
    
    test_labels = {}
    c = 0
    for i in range(nrows, tot_rows):
        test_labels[c] = all_labels[i]
        preds[c] = {0:0,1:0}
        c += 1
    
    for i in range(51):
        if verbose:
            print("_______iteration:{}________".format(i))
        bag, bag_labs = bag_data(filtered_data, classes)
        best_col, best_split, leftlab, rightlab = gini_selection(bag, bag_labs)
        #pass an empty labels dictionary below so tally_predictions can predict
        tally_predictions(best_col, best_split, filtered_test, {}, leftlab, rightlab) 
        
    cv_preds = print_predictions("")
    acc = get_accuracy(cv_preds, test_labels)
    preds = {}
    
    new_dataset = feature_extraction(data, features_and_neighbors)
    return new_dataset, features_and_neighbors
    
def run_classifier(filtered_data, trainlabels, testdata, writing_file, writing_flag):
    '''Performs classification of the test set:
    >>bag the data, find the stump, tally predictions, output predictions
    Input: filtered_data=list of all the data after selecting the top features
           trainlabels=dictionary of train labels
           testdata=list of test data rows
           writing_file=full output file path
           writing_flag=True (write to a file) or False (don't write to a file)'''
    best_col, best_split, leftlab, rightlab = gini_selection(filtered_data, trainlabels)
    #pass an empty labels dictionary below so tally_predictions can predict
    tally_predictions(best_col, best_split, testdata, {}, leftlab, rightlab)
    actual_preds = print_predictions(writing_file, writing_flag, False)

def print_features(original_num_features, features):
    '''Prints the selected features (or all of them if all features are used)
    Input: original_num_features=integer number of columns in the traindata
           features=list of feature columns selected from chi squared test   '''
    if len(features) == original_num_features:
        print("ALL")
    else:
        print("List of features selected: {}".format(features))
        print("Num features: {}".format(len(features)))
            
def parse_options():
    global verbose
    parser = argparse.ArgumentParser(description = module_description)
    parser.add_argument("data_file", help="path to the data file")
    parser.add_argument("labels_file", help="path to the labels file")
    parser.add_argument("test_file", help="path to the test data file")
    parser.add_argument("--vb", action="store_true", help="print detailed information for debugging")
    parser.add_argument("--wp", action="store_true", help="write predictions to a csv file")
    parser.add_argument("--wf", help="path to csv file for predictions")
    ret_args = parser.parse_args()
    if ret_args.vb:
        verbose = True
    if ret_args.wp:
        if ret_args.wf == None:
            sys.exit("wp option entered, expected wf option as well")
    return ret_args

if __name__ == "__main__":
    args = parse_options()
    data_filepath, labels_filepath, test_filepath, write_flag = args.data_file, args.labels_file, args.test_file, args.wp
    output_file = ""
    if write_flag:  
        output_file = args.wf  
    data_content = data_reader(data_filepath)
    data_rows, data_cols = len(data_content), len(data_content[0])
    if verbose:
        print(data_rows, data_cols)
    
    label_content = labels_reader(labels_filepath)
    classes, class_sizes = class_maker(label_content)
        
    testdata_content = data_reader(test_filepath)
    test_rows, test_cols = len(testdata_content), len(testdata_content[0])
    if verbose:
        print(test_rows, test_cols)
        
    extracted_data, features_neighbors = cross_validate(data_content, classes)
    extracted_test_data = feature_extraction(testdata_content, features_neighbors)
    construct_predictions_dictionary(extracted_test_data)
    
    run_classifier(extracted_data, classes, extracted_test_data, output_file, write_flag)
    
    print_features(data_rows, features_neighbors)
