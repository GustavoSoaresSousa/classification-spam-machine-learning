from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle

with open('database_email.pkl', 'rb') as f:
  x_email_train, y_email_train, x_email_test,  y_email_test = pickle.load(f);



param_grid = {
  'hidden_layer_sizes': [(50,), (100,), (150,)],
  'activation': ['identity', 'logistic', 'tanh', 'relu'],
  'solver': ['adam', 'sgd'],
  'batch_size': [10, 56] # a atualização de pesos vai ser feita de 56 em 56 registros
}
scoring = 'accuracy'

neural_database = MLPClassifier()

# neural_database.fit(x_email_train, y_email_train)
# database_predict = neural_database.predict(x_email_test) # 97.32% all parameters defaults
#accuracy = accuracy_score(y_email_test, database_predict)
#print("Accuracy:", accuracy)
#print(classification_report(y_email_test, database_predict))


grid = GridSearchCV(neural_database, param_grid, scoring=scoring)
grid.fit(x_email_train, y_email_train)
best_parameters = grid.best_params_
best_results = grid.best_score_
print("Best parameters:", best_parameters)
print("Best Results:", best_results)











  