from joblib import load, dump
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import spacy

text = load("transcription.joblib")
vector = load("sentence_transformer.joblib")
model = load("logistic_model.joblib")
le = load("label_encoder.joblib")

tokens = sent_tokenize(text)

dump(tokens, "sentences_list.joblib")

action_verbs = ["fix","update","write","design","create","implement",
                "develop","optimize","improve","resolve","repair"]

dump(action_verbs, "action_verbs_list.joblib")

final_sentence = []
for s in tokens:
  for i in action_verbs:
    if i in s:
      final_sentence.append(s)
print(final_sentence)

nlp = spacy.load("en_core_web_sm")
docs = [nlp(sentence) for sentence in final_sentence]

cleaned_docs = []

for doc in docs:
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    cleaned_docs.append(tokens)

dump(cleaned_docs, "cleaned_sentences.joblib")
print(cleaned_docs)

remove_prefix = cleaned_docs

for i in remove_prefix:
    for j in i:
        if j.lower() in action_verbs:
            break
        remove_prefix[remove_prefix.index(i)].remove(j)
        
print(remove_prefix)

deadlines = ["today", "tomorrow", "tonight","yesterday", "next", "this week", "end of week", "by", "before","after","monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

task_discription = []
for sentence in remove_prefix:
    temp = []
    for word in sentence:
        if word.lower() not in deadlines:
            temp.append(word)
    task_discription.append(" ".join(temp))
    
print(task_discription)

x_input = vector.encode(task_discription)
predictions = model.predict(x_input)
predictions = le.inverse_transform(predictions)
print(predictions)

task_summary = []

for i in range(len(task_discription)):
    task_summary.append((task_discription[i], predictions[i]))
    
print(task_summary)

dump(task_summary, "task_summary.joblib")
