import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

X_train = np.array(["heat map report is unavailable because data is empty!",
                    "Vivaldi profile got locked",
                    "batch details are blank when batch number was searched.",
                    "I don't see the commentary screen"
                    ])
y_train_text = [
                ['work notes for support team - refresh the solr collections'],
                ['Please raise a catalog request to reset the password'],
                ['Please provide the batch number'],
                ['Please raise an IAM request for Commenter role']
               ]

X_test = np.array(['Vivaldi screen displaying message that account got locked',
                   'heat map report is empty in core metrics dashboard',
                   'commentary module not visible or accessible',
                   'batch detail page returning blank results',
                   'heat map empty'
                   ])

#target_names = ['SCCM', 'Trace','Vivaldi']

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train_text)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

print("Chatbot is trained using following existing ticket descriptions:")
print("================================================================")
for item, labels in zip(X_train, y_train_text):
    print('{0} => {1}'.format(item, ', '.join(labels)))

print("\nTesting the chatbot with following modified descriptions:")
print("===========================================================")
for item, labels in zip(X_test, all_labels):
    print('{0} => {1}'.format(item, ', '.join(labels)))