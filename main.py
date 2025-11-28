from joblib import load
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.width', None)

task_summary = load("task_summary.joblib")
sentances = load("sentences_list.joblib")
action_verbs = load("action_verbs_list.joblib")
cleaned_docs = load("cleaned_sentences.joblib")
vector = load("sentence_transformer.joblib")

deadlines = [
    "tomorrow evening",
    "tomorrow",
    "end of this week",
    "friday's release",
    "friday",
    "next monday",
    "wednesday"
]

sent_tokens = [s.lower().strip() for s in sentances]
action_verbs = [v.lower() for v in action_verbs]
deadlines = [d.lower() for d in deadlines]
cleaned_docs = [" ".join(tokens) for tokens in cleaned_docs]

deadline_task = []

for idx, sentence in enumerate(sent_tokens):

    if any(verb in sentence for verb in action_verbs):

        same_deadline = None
        for d in deadlines:
            if d in sentence:
                same_deadline = d
                break

        if same_deadline:
            deadline_task.append(same_deadline)

        else:
            if idx + 1 < len(sent_tokens):
                next_sentence = sent_tokens[idx + 1]

                next_deadline = None
                for d in deadlines:
                    if d in next_sentence:
                        next_deadline = d
                        break

                if next_deadline:
                    deadline_task.append(next_deadline)
                else:
                    deadline_task.append("no specific deadline")
            else:
                deadline_task.append("no specific deadline")

x = len(deadlines)/2

priority = []

for i in range(len(deadlines)):
    if i < x:
        priority.append("High")
    else:
        priority.append("Medium")

# print(task_summary)

team_data = {
    'Name': ['Sakshi','Mohit','Arjun','Lata'],
    'Role': ['Frontend Developer','Backend Engineer','UI/UX Designer','QA Engineer'],
    'Skills': [['React', 'JavaScript', 'UI bugs'],['Database', 'APIs', 'Performance optimization'],
               ['Figma', 'User flows', 'Mobile design'],['Testing', 'Automation', 'Quality assurance']]
}

team_df = pd.DataFrame(team_data)
# print(team_df)

task_summary = [list(item) for item in task_summary]

querys = []

for i in task_summary:
    querys.append(' '.join(i))

#  print("Querys:", querys)

team_df['Meta'] = team_df['Name'] + ' ' + team_df['Role'] + ' ' + team_df['Skills'].apply(lambda x: ' '.join(x))

# print("Team Meta:", team_df)

output_df = pd.DataFrame(columns=['Task Summary', 'Assigned To', 'Deadline', 'Priority', 'Reason'])

output_df['Task Summary'] = [task_summary[i][0] for i in range(len(task_summary))]
output_df['Deadline'] = deadline_task
output_df['Reason'] = [task_summary[i][1] for i in range(len(task_summary))]

for idx, task in enumerate(cleaned_docs):
    task_vec = vector.encode([task])
    team_vecs = vector.encode(team_df['Meta'].to_list())

    similarities = cosine_similarity(task_vec, team_vecs).flatten()
    best_match_index = similarities.argmax()
    
    assigned_to = team_df.iloc[best_match_index]['Name']
    
    output_df.loc[idx,'Assigned To'] = assigned_to
    

for idx, d in enumerate(deadlines):
    if d in deadline_task:
        output_df.loc[output_df['Deadline'] == d, 'Priority'] = priority[idx]

output_df['Priority'].fillna('medium', inplace=True)

print(output_df)