import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from joblib import dump

df = pd.read_csv('training Data.csv')

st = SentenceTransformer("all-MiniLM-L6-v2")
x = st.encode(df['text'])

le = LabelEncoder()
y = le.fit_transform(df['category'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegressionCV(cv=5)
model.fit(x_train, y_train)

model_pred = model.predict(x_test)

acc = accuracy_score(model_pred, y_test)
print("Accuracy: {:.2f}%".format(acc * 100))

print(classification_report(y_test,model_pred))

dump(model, "logistic_model.joblib")
dump(st, "sentence_transformer.joblib")
dump(le, "label_encoder.joblib")
