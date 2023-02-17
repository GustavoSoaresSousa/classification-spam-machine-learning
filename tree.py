import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

with open('database_email.pkl', 'rb') as f:
  x_email_train, y_email_train, x_email_test,  y_email_test = pickle.load(f);


trees_emails = DecisionTreeClassifier(criterion='entropy',random_state=0)
trees_emails.fit(x_email_train, y_email_train)

predicts_emails = trees_emails.predict(x_email_test)
accuracy_predicts_emails = accuracy_score(y_email_test, predicts_emails) # Gini - 92.28% /  entropy 93.22%

# print(accuracy_predicts_emails)
# print(classification_report(y_email_test, predicts_emails))










