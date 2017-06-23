# MLDS_Final_Project_Adaptive_Batch_Size_with_
NTU MLDS Course Final Project

Environment

Python: 2.7
Keras: 2.0.4

Usage

There are 3 datasets used in the project: cifar10, mnist, and our own datasets

if you want to use our own datasets, please download the dataset from here:https://drive.google.com/open?id=0BxsYkQxhzTmFZW1fVW14NVhaVlU

and unzip it under the datasets folder

there are five dynamic-batch-size-mode can do:

constant batch size: constant
linear: linear
exponential: exp
gaussian: gau
cosine: cos

and you can use different batch size by appending the batch size after the mode
e.g.: 
python fish_classifier_dynamic_batch.py --dataset mnist --batch_type linear 256

