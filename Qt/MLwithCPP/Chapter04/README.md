# Chapter 04: Clustering

- <b>Clustering</b> is an <b>unsupervised machine learning</b> that is used for splitting the original dataset of objects into groups classifed by properties.
- Depending on the types of dimensions in this spsce, which can be both numerical and categorical, we choose the type of clustering algorithm and specific metric function
- The main difference between <b>clustering</b> and <b>classification</b> is an underfined set of target groups
- We can split cluster analysis into the following phases:
  - Selecting objects for clustering
  - Determining the set of object properties that we will use for the metric
  - Normalizing property values
  - Calculating the metric
  - Identifying distinct groups of objects based on metric values

## Technical Requirements
 * Modern C++ compiler with C++17 support
 * CMake build system version >= 3.8
 * ```Dlib``` library installation
 * ```Shogun-toolbox``` toolbox library installation
 * ```Shark-ML``` library installation

## Type of Clustering Algorithms
 * [Partition-based Clustering Algorithms](#partition-based-clustering-algorithms)
 * [Distance-based Clustering Algorithms](#distance-based-clustering-algorithms)
 * [Graph Theory-based Clustering Algorithms](#graph-theory-based-clustering-algorithms)
 * [Spectral Clustering Algorithms](#spectral-clustering-algorithms)
 * [Hierarchical Clustering Algorithms](#hierarchical-clustering-algorithms)
 * [Density-based Clustering Algorithms](#density-based-clustering-algorithms)
 * [Model-based Clustering Algorithms](#model-based-clustering-algorithms)

## Partition-based Clustering Algorithms
 - Use a similarity measure to combine objects into groups
 - Sometimes, several measures need to be tried with the same algorithm to choose the best one
 - Require either the number of desired clusters or a threshold that regulates the number of output clusters to be explicitly specified

## Distance-based Clustering Algorithms
 - The most known representatives of this family of methods are the k-means and k-medoids algorithms. 
 - With the k-means algorithm, the similarity is proportional to the distance to the cluster center of mass. ```The cluster center of mass``` is the average value of cluster objects' coordinates in the data space.

## Graph Theory-based Clustering Algorithms

## Spectral Clustering Algorithms

## Hierarchical Clustering Algorithms

## Density-based Clustering Algorithms

## Model-based Clustering Algorithms
