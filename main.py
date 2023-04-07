import pandas as pd
from dataclasses import dataclass
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


@dataclass
class MLDataSet:
    '''Class for representing dataset for ML.'''
    x_train: pd.DataFrame # feature dataframe for training.
    y_train: pd.DataFrame # label dataframe for training. Size is equal of x_train.
    x_test: pd.DataFrame  # feature dataframe for testing.
    y_test: pd.DataFrame  # label dataframe for testing. Size is equal to x_test.


def read_data() -> pd.DataFrame:
    '''Reads the intended data file and return it as Data Frame.'''
    df = pd.read_csv('mushrooms.csv')
    return shuffle(df)


def process_data_frame_for_sklearn(df: pd.DataFrame):
    ''' Processes data frame and prepare it for sklearn modules.
    Args:
        df: DataFrame containing the entire dataset.
    Returns:
        A MLDataSet instance.
    '''
    column_names = list(df.columns)

    for column in column_names:
        df[column] = df[column].astype('category').cat.codes
    
    class_column = column_names[0]
    feature_columns = column_names[1:]

    # Create data frame with features only.
    X = df[feature_columns]
    # Create data frame with labels only.
    Y = df[class_column]

    # Split input in 70/30 with 70% as train set, 30% as test set.
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    d = MLDataSet(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return d


def decision_tree_based_classification(d: MLDataSet):
    '''Classify the data with decision tree.'''
    clf = DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(d.x_train, d.y_train)
    y_pred = clf.predict(d.x_test)
    print('--------------')
    print('Decision tree performance:')
    print('Accuracy: ', metrics.accuracy_score(d.y_test, y_pred=y_pred))
    print('F1 score: ', metrics.f1_score(d.y_test, y_pred=y_pred, average='weighted'))


def naive_bayes_classification(d: MLDataSet):
    '''Naive Bayes based classfication'''
    clf = CategoricalNB()
    clf = clf.fit(d.x_train, d.y_train)
    y_pred = clf.predict(d.x_test)
    print('--------------')
    print('Naive Bayes performance:')
    print('Accuracy: ', metrics.accuracy_score(d.y_test, y_pred=y_pred))
    print('F1 score: ', metrics.f1_score(d.y_test, y_pred=y_pred, average='weighted'))


def main():
    df = read_data()
    ds = process_data_frame_for_sklearn(df)
    decision_tree_based_classification(ds)
    naive_bayes_classification(ds)


if __name__ == '__main__':
    main()