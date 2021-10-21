import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from src.svm_utils import svm_classify_image
from src.svm_utils import build_samples 
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

print('something')

now = datetime.now()

svm_classify_image('./images/cow.jpg', X_train, f_X_train)


later = datetime.now()

print(later - now)
