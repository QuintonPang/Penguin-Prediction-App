#import modules
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle # for exporting the model

# obtain cleaned csv of penguins dataset
# must be raw content
penguins = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

# Ordinal feature encoding
df = penguins.copy()
# print(penguins.to_string())
# attribute to be predicted
target = 'species'
# attributes used to predict the target
encode=['sex','island']

for col in encode:
    # prefix means 
    dummy = pd.get_dummies(df[col],prefix=col)
    # combine two dataframes
    df = pd.concat([df,dummy],axis=1)
    del df[col]

#  map target to number
target_mapper = {'Adelie':0,'Chinstrap':1,'Gentoo':2}

# encode target according the the dictionary
def target_encode(val):
    return target_mapper[val]

# encode the species using the function
df['species'] = df['species'].apply(target_encode)

# separate x and y
# drop column species
X = df.drop('species',axis=1)
Y = df['species']

# Build Random Forest Classifier model
clf = RandomForestClassifier()
clf.fit(X,Y)

# export model
# w is write, b is binary
pickle.dump(clf,open('../penguins_clf.pkl','wb'))