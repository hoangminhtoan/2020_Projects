# DataProcessing

## Technical Requirements
 * Modern C++ compiler with C++17 support
 * Cmake build system version >= 3.8
 * ```Dlib``` library installation
 * ```Shogun``` toolbox library installation
 * ```Shark-ML``` library installation
 * ```Eigen``` library installation
 * ```hdf5lib``` library installation
 * ```HighFive``` library installation
 * ```RapidJSON``` library installation
 * ```Fast-CPP-CSV-Paser``` library installation

## Table Contents
 * [Comma Sperated Values (CSV)](#comma-seperated-values)
 * [JSON](#json)





### Comma Sperated Values
 - The most popular format for representing structured data is called CSV. This format is just a text file with a two-dimensional table in it whereby values in a row are sperated with commas
 - The <b>advantages</b> of this file format:
   + A straightforward structure
   + Human-readable
   + Supported on a variety of computer platforms
 - The <b>disadvantages</b> of this file format:
   + A lack of support of multidimensional data and data with complex structuring
   + Slow parsing speed in comparison with binary format

### JSON
 - This is a file format with name-value pairs and arrays of such pairs
 - The <b>advantages</b> of this file format:
   + Human-readable
   + Supported on a variety of computer platforms
   + Possible to store hierarchical and nested data structures.
 - The <b>disadvantages</b> of this file format:
   + Slow parsing speed in comparison with binary format
   + Not very useful for representing numberical matrices