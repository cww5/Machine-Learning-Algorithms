#Connor Watson
#HW 08 - K Means Unsupervised Clustering
#11/17/2018

import argparse
import sys
import random

def data_reader(data_file):
    """Read data
    Input: data_file=string (file path to the data from pwd)
    Output: data=the content of the data formatted for this script"""
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

def distance(x_point, center_point):
    """Calculate Euclidean distance from data point to a centroid
       Input: x_point - a data point
              center_point - a centroid
       Output : dist - Euclidean distance from x to centroid"""
    dist = sum([(x - c)**2 for x, c in zip(x_point, center_point)]) ** 0.5
    return dist

def get_centroids(points, k):
    """Calculate centroids for the data points
    Input: points - the data read in from the file
           k - the amount of centroids
    Output: list of centers"""
    d = {}
    centers = []
    for i in range(0, rows):
        distances = []
        for j in range(0, len(k)):
            distances.append(distance(points[i], k[j]))
        idx = distances.index(min(distances))
        if idx not in d:
            d[idx] = [points[i]]
        else:
            d[idx].append(points[i])
    for key in d:
        centers.append([sum(dist)/len(dist) for dist in zip(*d[key])])
    return centers

def k_means(data, k):
    """Gets the final centroids for the data
    Input: data - the data read in from the file
           k - the amount of centroids
    Output: list of final centroids"""
    centroids = random.sample(data, k)
    while True:
        #Compute new centroids until convergence
        new_centroids = get_centroids(data, centroids)
        set_cents = set([tuple(a) for a in centroids])
        set_new_cents = set([tuple(a) for a in new_centroids])
        if  set_cents == set_new_cents:
            #Convergence - centroids don't change
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids

def classify(data, centroids):
    """Get labels for points according to the closest centroid
    Input: data - the data read in from the file
           centroids - the calculated centroids
    Output: for each row of data, output the centroid with the centroid index"""
    for i in range(0, rows):
        distances = []
        for j in range(0,len(centroids)):
            distances.append(distance(data[i], centroids[j]))
        idx = distances.index(min(distances))
        print('{} {}'.format(idx, i))
        
def parse_options():
    """This method parses the input args given by the user."""
    parser = argparse.ArgumentParser(description="K means unsupervised clustering algorithm")
    parser.add_argument("data_file", help="path to the data file")
    parser.add_argument("num_clusters", help="number of clusters to form")
    ret_args = parser.parse_args()
    return ret_args

if __name__ == "__main__":
    args = parse_options()
    data_filepath, k_clusters = args.data_file, int(args.num_clusters)    
    data_content = data_reader(data_filepath)
    global rows, columns
    rows = len(data_content)
    cols = len(data_content[0])
    centers = k_means(data_content, k_clusters)
    classify(data_content, centers)
