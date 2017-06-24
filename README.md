# MLDS_Final_Project_Adaptive_Batch_Size_with_Epoch
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
training on fish dataset
	python fish_classifier_dynamic_batch.py --dataset fish --batch_type linear 256

training on mnist dataset  
	python fish_classifier_dynamic_batch.py --batch_type constant 32   

training on cifar-1- dataset
	python fish_classifier_dynamic_batch.py --batch_type gau 256  

