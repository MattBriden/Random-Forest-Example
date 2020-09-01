import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


stop_words = frozenset(["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","ll","t","re","d","ve","s",'aren', 'can', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'mustn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])


def create_model():
    csv_file = os.path.normpath(os.path.join('..', 'data', 'test-data.csv'))
    data = pd.read_csv(csv_file, encoding='cp437')

    # the y values will be the spam and ham values for each text represented by 1 and 0
    cleanup_data = {"v1": {"ham": 0, "spam": 1}}
    data.replace(cleanup_data, inplace=True)
    y = data.iloc[:, 0].values

    # x values will be an array of the words in each text tokenized by CountVectorizer
    x = []
    for i in range(0, len(data['v2'])):
        x.append(data['v2'][i])
    cv = CountVectorizer(max_features=3000, stop_words=stop_words)
    x = cv.fit_transform(x).toarray()

    # split the train and test data using 75% of the data for training, 42 is used as the random state
    train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.25, random_state=42)

    # create a random forest with 30 decision trees
    rf = RandomForestClassifier(n_estimators=100, random_state=42, criterion='entropy')

    # train data and pickle the random forest
    rf.fit(train_features, train_labels)

    # test to see accuracy
    predictions = rf.predict(test_features)
    print(f'Accuracy of test: {accuracy_score(predictions, test_labels)}')

    # pickle random forest and count vectorizer objects
    model_file = os.path.normpath(os.path.join('..', 'data', 'model.p'))
    cv_file = os.path.normpath(os.path.join('..', 'data', 'cv.p'))
    pickle.dump(rf, open(model_file, 'wb'))
    pickle.dump(cv, open(cv_file, 'wb'))


if __name__ == '__main__':
    create_model()
