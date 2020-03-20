import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt  # so we can add to plot
import numpy as np  # needed for math stuff

import seaborn as sns

dvaDat = pd.read_table("data_banknote_authentication.txt", sep=",",
                       names=["variance", "skewness", "curtosis", "entropy", "class"])

cols = dvaDat.columns
X = dvaDat.iloc[:, 0:3].values
y = dvaDat.iloc[:, 4].values

# missing value counts in each of these columns
numeric_data = dvaDat.select_dtypes(include=[np.number])
dvaDat = pd.read_table("data_banknote_authentication.txt", sep=",",
                       names=["variance", "skewness", "curtosis", "entropy", "class"])
corr = numeric_data.corr()
# print(corr)
sn.heatmap(corr, square=True)

# Basic correlogram
sn.pairplot(dvaDat, kind="reg")

plt.figure(figsize=(100,100))
plt.legend(loc='upper center')
plt.show()
