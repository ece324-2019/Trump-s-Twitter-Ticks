import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

confusion = [[133,92,73],[135,245,105],[68,73,112]]
df_cm = pd.DataFrame(confusion, index = ['down','no change','up'],
                  columns = ['down', 'no change','up'])
plt.figure(figsize = (3,3))
plt.xlabel('actual')
plt.ylabel('predicted')
sn.heatmap(df_cm, annot=True, fmt='.2f')

plt.show()