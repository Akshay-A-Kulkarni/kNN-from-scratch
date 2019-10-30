# kNN-from-scratch

An implementation of the kNN Algorithm.

The original dataset is contained in the "spambase.data" file, courtesy of UC Irvine's Machine Learning Repository. It contains feature data detailing characteristics of several thousand emails. This feature data is divided into 57 columns such as word frequency and lengths of adjacent capital letters, and a 58th column documenting whether the email was either SPAM or HAM (i.e., it was not junk mail).

The file itself is unlabeled, however the names of the columns in order are provided in the "spambase.names" file included in this repository (and also courtesy of the UCI Machine Learning Repository).

Further documentation is included in the "spambase.documentation" file included in this repository.

The knn.py file enclosed in this repository pre-processes the data into a compatible format before running it through an implementation of kNN. The kNN Algorithm attempts to classify "new" test emails as either SPAM or HAM based on their Euclidean Distance from the "old' training emails based on a majority vote conducted by each test email's k-nearest-neighbors.
