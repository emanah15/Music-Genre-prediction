import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

df = pd.read_csv('music.csv')

X = df.drop(columns = 'genre')
y = df['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

 #build model using ML algorithm
 #decision tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train) #training

#tree.export_graphviz(model, out_file= 'music_recommender.dot', feature_names = ['age', 'gender'])

#predict
predictions = model.predict(X_test)
print(predictions)
score = accuracy_score(y_test, predictions)
print(score)

plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
