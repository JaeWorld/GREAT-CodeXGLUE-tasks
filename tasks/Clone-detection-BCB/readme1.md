Common Program Analysis
This project aims to find the common programs between two datasets, which can be used to evaluate code similarity algorithms. The program uses codeBERT model to encode the source code and cosine similarity as the similarity measure. The program then outputs a dictionary of similarity scores for each combination of source code pairs in the two datasets. It might takes several days for executing.

Requirements
This program requires the following packages:

torch
numpy
transformers
tqdm
The packages can be installed via pip:

python

  pip install torch
  pip install numpy
  pip install transformers
  pip install tqdm

Usage
To run the program, execute main.py file. The program takes two dataset names as input, and uses config.py to set the paths for the dataset files. The helper.py file contains several utility functions used in main.py.

The main function find_common_programs loads the dataset lists, randomly samples the specified number of programs, encodes the source code using codeBERT, calculates the cosine similarity, and outputs a dictionary of similarity scores.

The results are stored as a dictionary in a pickle file named as dataset1_dataset2.pkl in the directory specified in config.py.

Notes
The code uses logging module to output logs of the program run, and the logs are stored in a file named as commonAnalysis_yyyy-mm-dd.log.
The program skips certain dataset combinations as they are not required for analysis.
The source code pairs with exact similarity score of 1 are not removed. This can be modified in the find_common_programs function.
