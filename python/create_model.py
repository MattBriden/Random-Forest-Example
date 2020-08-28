import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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
    cv = CountVectorizer(max_features=3000)
    x = cv.fit_transform(x).toarray()

    # split the train and test data usinf 75% of the data for training, 42 is used as the random state
    train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.25, random_state=42)

    # create a random forest with 30 decision trees
    rf = RandomForestRegressor(n_estimators=30, random_state=42)

    # train data and pickle the random forest
    rf.fit(train_features, train_labels)

    # test to see error rate
    predictions = rf.predict(test_features)
    errors = abs(predictions - test_labels)
    print(f'Error rate for test data: {round(np.mean(errors), 2)}')

    # pickle random forest and count vectorizer objects
    model_file = os.path.normpath(os.path.join('..', 'data', 'model.p'))
    cv_file = os.path.normpath(os.path.join('..', 'data', 'cv.p'))
    pickle.dump(rf, open(model_file, 'wb'))
    pickle.dump(cv, open(cv_file, 'wb'))


if __name__ == '__main__':
    create_model()





