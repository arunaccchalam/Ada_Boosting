import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report,plot_confusion_matrix,accuracy_score

# data
df = pd.read_csv("mushrooms.csv")
df.head()

# EDA

sns.countplot(data=df,x='class')

df.describe()

df.describe().transpose()

plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=df.describe().transpose().reset_index().sort_values('unique'),x='index',y='unique')
plt.xticks(rotation=90);

# Train Test Split

X = df.drop('class',axis=1)

X = pd.get_dummies(X,drop_first=True)

y = df['class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

# Modeling

model = AdaBoostClassifier(n_estimators=1)

model.fit(X_train,y_train)

## Evaluation


predictions = model.predict(X_test)

predictions

print(classification_report(y_test,predictions))

model.feature_importances_

model.feature_importances_.argmax()

X.columns[22]

sns.countplot(data=df,x='odor',hue='class')

# Analyzing performance as more weak learners are added.

len(X.columns)

error_rates = []

for n in range(1,96):
    
    model = AdaBoostClassifier(n_estimators=n)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    err = 1 - accuracy_score(y_test,preds)
    
    error_rates.append(err)

plt.plot(range(1,96),error_rates)

model

model.feature_importances_

feats = pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=['Importance'])

feats

imp_feats = feats[feats['Importance']>0]

imp_feats

imp_feats = imp_feats.sort_values("Importance")

plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=imp_feats.sort_values('Importance'),x=imp_feats.sort_values('Importance').index,y='Importance')

plt.xticks(rotation=90);

sns.countplot(data=df,x='habitat',hue='class')
