import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

with open('database_email.pkl', 'rb') as f:
  x_email_train, y_email_train, x_email_test,  y_email_test = pickle.load(f);




svm_email = SVC(random_state=0, C=5.0, kernel='poly')
svm_email.fit(x_email_train, y_email_train)
predicts_emails = svm_email.predict(x_email_test)
accuracy_predict_emails = accuracy_score(y_email_test, predicts_emails) 
# 93.07% rbf C = 1 
# 95.43% rbf C = 5  recall e precision melhor
# 95.11% linear C = 5 recall e precison quase igual ao ultimo 
# 77.00% poly C = 10

print(accuracy_predict_emails)
print(classification_report(y_email_test, predicts_emails))






