# Random Forest for SMS Spam Classification
A simple implementation of a random forest using scikit-learn that determines if a text message is "ham" or "spam".

## Use
Create a virtualenv and install the needed dependencies for this project.
```
Random-Forest-Example $ python3 -m venv venv3
Random-Forest-Example $ source venv3/bin/activate
(venv3) Random-Forest-Example $ pip install -r requirements.txt 
```

Run the create_model.py file from the python directory to rebuild the model and pickle it to the data directory. The random forest included in this project by default uses 30 decision trees and took a few minutes to build. To change the number of decision trees when building a new model update the `n_estimators` argument when creating the random forest object.
```
(venv3) Random-Forest-Example/python $ python3 python/create_model.py 
```
Test against custom text messages using the python3 console and importing the pickled scikit-learn objects.
```python
>>> import pickle
>>> rf = pickle.load(open('../data/model.p', 'rb'))
>>> cv = pickle.load(open('../data/cv.p', 'rb'))
>>> 
>>> import numpy as np
>>> spam_text = np.array(["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"])
>>> spam_data = cv.transform(spam_text).toarray()
>>> print(f'Probability of spam: {rf.predict_proba(spam_data)[0][1]}')
>>> 0.96
```

## Explanation of Model
A random forest was used as the model for this application. It was trained on a data set of 5575 text messages each being labeled as either "ham" or "spam". The python package sklearn was used to create the random forest object and fit it with the training data.

To create the random forest we first needed to pull in our training data in a manner that could be used by machine learning algorithms. The data was initially in the format shown below, with one column containing either "ham" or "spam" and the other containing the contents of the SMS message.
```python
>>> data = pd.read_csv(_csv_file, encoding='ISO-8859-1')
>>> data.head(5)

     v1                                                 v2
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...
```
The v1 column that indicated "ham" or "spam" was easy to change, it was decided "ham" would be represented by 0 and "spam" by 1. The text column was a bit more tricky. To make the data processable by machine learning algorithms we needed to tokenize and normalize each text. This was accomplished by using the CountVectorizer object from scikit-learn. This object takes each text and turns each word, defined as a "feature", into a count of its occurrences. The CountVectorizer in this implementation only takes into account the most frequented 3000 words. A list of "stop words" was also applied so to the CountVectorizer object so common words that don't help classification (think articles, pronouns, and prepositions) are not taken into top 3000 frequented word. Therefore each set of features, or tokenized and normalized representations of each text, will be an array of size 3000 with each object in the array being a count of how many times that word occurs in the text. Below is the implementation.
```python
# replace "ham" and "spam" with 0 and 1
cleanup_data = {"v1": {"ham": 0, "spam": 1}}
data.replace(cleanup_data, inplace=True)
y = data.iloc[:, 0].values

# x values will be an array of the occurrences the words in each text tokenized by CountVectorizer
x = []
for i in range(0, len(data['v2'])):
    x.append(data['v2'][i])
cv = CountVectorizer(max_features=3000)
x = cv.fit_transform(x).toarray()
```
At this point the data is now in a format that can be used to create training and testing data for our model. scikit-learn makes it incredibly easy to generate this data and it can be done with a single line of code. In this implementation 75% of the data was used to train the model with the rest saved for testing. A random_state variable of 42 was used which will be used to seed the random number generator. This will ensure reproducible results for all subsequent function calls, and as to why 42 was used see [here](https://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker%27s_Guide_to_the_Galaxy#The_Answer_to_the_Ultimate_Question_of_Life,_the_Universe,_and_Everything_is_42).
```python
# split the train and test data using 75% of the data for training, 42 is used as the random state
train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.25,
random_state=42)
```
Now that the data has been split the random forest object can be instantiated. This implementation used a random forest with 100 decision trees but with more trial and error this number could be fine tuned. Once the object is created it can be fit with the training data which completes our model.
```python
# create a random forest with 30 decision trees
rf = RandomForestClassifier(n_estimators=100, random_state=42, criterion='entropy')
 
# train data
rf.fit(train_features, train_labels)
```

Once the forest has been fit with the training data we can test it out. Using the testing data that was created with scikit-learn we can gather predictions for each text and compare it against the actual classifications that we know each text to be. For this model we had an accuracy of roughly 98%.
```python
# test to see accuracy
predictions = rf.predict(test_features)
print(f'Accuracy of test: {accuracy_score(predictions, test_labels)}')
>>> Accuracy of test: 0.9770279971284996
```
The model could be fine tuned for greater accuracy but seemingly works well at a glance. More accurate or even a greater amount of data to train the model on could be beneficial in the future. Below is the results from predicting classifications for two texts, one obviously ham and the other more spam-like.
```python
# text that is spam like being predicted as 96% chance of being spam
spam_text = np.array(["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"])
spam_data = cv.transform(spam_text).toarray()
print(f'Probability of spam: {rf.predict_proba(spam_data)[0][1]}')
>>> 0.96

# text that is ham being predicted as 0% chance of being spam
ham_text = np.array(["Hello mate, when can I expect you today?"])
ham_data = cv.transform(ham_text).toarray()
print(f'Probability of ham: {rf.predict_proba(ham_data)[0][1]}')
>>> 0.0
```
