import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

dataset = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Projetos-Pessoais/classification-spam/spam_ham_dataset.csv')
## print(dataset.isnull().sum())

dataset_copy = dataset.copy()
dataset_copy.drop(["Unnamed: 0", "label"], axis=1, inplace=True)
dataset_copy.drop_duplicates()
dataset_copy = dataset_copy.iloc[: -2000]

dataset_copy['text'] = dataset_copy['text'].str.replace(";", "")
dataset_copy['text'] = dataset_copy['text'].str.replace(":", "")
dataset_copy['text'] = dataset_copy['text'].str.replace("#", "")
dataset_copy['text'] = dataset_copy['text'].str.replace("(", "")
dataset_copy['text'] = dataset_copy['text'].str.replace(")", "")
dataset_copy['text'] = dataset_copy['text'].str.replace(".", "")
dataset_copy['text'] = dataset_copy['text'].str.replace(",", "")
dataset_copy['text'] = dataset_copy['text'].str.replace("  ", "")


stop_words = set(stopwords.words("english"))

filtered_texts = []

for text in dataset_copy["text"]:
  words = text.split()
  filtered_words = [word for word in words if word.lower() not in stop_words]
  filtered_texts.append(" ".join(filtered_words))


for i in range(len(filtered_texts)):
  dataset_copy['text'][i] = filtered_texts[i]


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

x_dataset = dataset_copy['text']
y_dataset = dataset_copy['label_num']

x_dataset= vectorizer.fit_transform(x_dataset).toarray()

from sklearn.model_selection import train_test_split

x_email_train, x_email_test, y_email_train, y_email_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)

import pickle

# with open('database_email.pkl', mode='wb') as f: ## rb para usar depois
#   pickle.dump([x_email_train, y_email_train, x_email_test,  y_email_test], f)


# print('Done')
















