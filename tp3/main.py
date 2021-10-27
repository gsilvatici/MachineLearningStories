import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from src.svm_utils import svm_classify_image
from src.svm_utils import build_samples 
from src.svm_utils import plot_kernel_results
from src.svm_utils import plot_c_results
from src.svm_utils import plot_radial_results

from datetime import datetime, timedelta

GRASS = 0
COW = 1
SKY = 2
FARM = 3

grass_samples, grass_predictions = build_samples('./images/pasto.jpg', class_value = GRASS)
cow_samples, cow_predictions = build_samples('./images/vaca.jpg', class_value = COW)
sky_samples, sky_predictions = build_samples('./images/cielo.jpg', class_value = SKY)

X = np.append(np.append(grass_samples, cow_samples, axis=0), sky_samples, axis=0)
f_X = np.append(np.append(grass_predictions, cow_predictions, axis=0), sky_predictions, axis=0)

X_train, X_test, f_X_train, f_X_test = train_test_split(X, f_X, test_size=0.2, random_state=42)


from src.svm_utils import svm_get_precision 

def try_c_values():
    out_file = open('c_precision.csv', 'a')
    # values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


    out_file.write('kernel,c value,train precision,test precision\n')
    
    for c in values:
        test_precision = svm_get_precision(X_train, f_X_train, X_test, f_X_test, c, kernel='rbf', gamma=0.01)
        new_line = f'linear,{c},{test_precision}\n'
        out_file.write(new_line)
        print(new_line)

    out_file.close()

def try_sig():
    kernels = ['sigmoid']

    test_precision = svm_get_precision(X_train, f_X_train, X_test, f_X_test, c=1.0, kernel='sigmoid')  
    print("ke")
    new_line = f'{test_precision}\n'
    print(new_line)

    
def try_radial():
    # out_file = open('radial_precision.csv', 'a')
    gammas = [0.01]

    for g in gammas:    
        test_precision = svm_get_precision(X_train, f_X_train, X_test, f_X_test, c=1.0, kernel='rbf', gamma=g)  
        new_line = f'{g},1,{test_precision}\n'
        # out_file.write(new_line)
        print(new_line)

    # out_file.close()
    
def try_kernels():
    kernels = ['linear', 'poly', 'rbf']

    out_file = open('kernel_precision.csv', 'a')

    for k in kernels:    
        train_precision, test_precision = svm_get_precision(X_train, f_X_train, X_test, f_X_test, c=1.0, kernel=k)  
        new_line = f'{k},1,{train_precision},{test_precision}\n'
        out_file.write(f'{k},1,{train_precision},{test_precision}\n')
        print(new_line)

    out_file.close()

now = datetime.now()

# try_sig()

# plot_kernel_results()

# plot_radial_results()

try_c_values()

# svm_classify_image('./images/farm2.jpg', X_train, f_X_train)

# try_c_values()


try_radial()
        
later = datetime.now()
    
print(later - now)