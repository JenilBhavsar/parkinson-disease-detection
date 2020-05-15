import pandas as pd
import numpy as np
import fnmatch
import os, gc
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import random
import scipy

def init_user_set():
    """
    Initializes a set of users that exist both in archived-data and archived-users
    :return: a python list of users common to both the folders
    """
    user_file_list = os.listdir('./input/Archived-users/Archived users/')
    user_set_v1 = set(map(lambda x: x[5: 15], user_file_list)) # [5: 15] to return just the user IDs

    tappy_file_list = os.listdir('./input/Archived-Data/Tappy Data/')
    user_set_v2 = set(map(lambda x: x[: 10], tappy_file_list)) # [: 10] to return just the user IDs

    user_set = user_set_v1.intersection(user_set_v2)
    user_set = sorted(user_set)
    print(len(user_set))
    print(user_set)
    return user_set


def read_user_file(file_name):
    """
    Reads file specified by file name
    :param file_name:
    :return: a Python list of each line
    """
    f = open('./input/Archived-users/Archived users/' + file_name)
    data = [line.split(': ')[1][: -1] for line in f.readlines()]
    f.close()

    return data


def generate_user_df(user_set):
    """
    Generates a dataframe for all the users in the user_set
    :param user_set: set of valid users
    :return: dataframe consisting of information of each valid user
    """
    files = os.listdir('./input/Archived-users/Archived users/')

    columns = [
        'BirthYear', 'Gender', 'Parkinsons', 'Tremors', 'DiagnosisYear',
        'Sided', 'UPDRS', 'Impact', 'Levadopa', 'DA', 'MAOB', 'Other'
    ]

    user_df = pd.DataFrame(columns=columns) # empty Data Frame for now

    for user_id in user_set:
        temp_file_name = 'User_' + user_id + '.txt' # tappy file names have the format of `User_[UserID].txt`
        if temp_file_name in files: # check to see if the user ID is in our valid user set
            temp_data = read_user_file(temp_file_name)
            user_df.loc[user_id] = temp_data # adding data to our DataFrame
    user_df.head()

    # force some columns to have numeric data type
    user_df['BirthYear'] = pd.to_numeric(user_df['BirthYear'], errors='coerce')
    user_df['DiagnosisYear'] = pd.to_numeric(user_df['DiagnosisYear'], errors='coerce')

    user_df = user_df.rename(index=str, columns={'Gender': 'Female'})  # renaming `Gender` to `Female`
    user_df['Female'] = user_df['Female'] == 'Female'  # change string data to boolean data
    user_df['Female'] = user_df['Female'].astype(int)  # change boolean data to binary data

    str_to_bin_columns = ['Parkinsons', 'Tremors', 'Levadopa', 'DA', 'MAOB',
                          'Other']  # columns to be converted to binary data

    for column in str_to_bin_columns:
        user_df[column] = user_df[column] == 'True'
        user_df[column] = user_df[column].astype(int)

    # prior processing for `Impact` column
    user_df.loc[
        (user_df['Impact'] != 'Medium') &
        (user_df['Impact'] != 'Mild') &
        (user_df['Impact'] != 'Severe'), 'Impact'] = 'None'

    to_dummy_column_indices = ['Sided', 'UPDRS', 'Impact']  # columns to be one-hot encoded
    for column in to_dummy_column_indices:
        user_df = pd.concat([
            user_df.iloc[:, : user_df.columns.get_loc(column)],
            pd.get_dummies(user_df[column], prefix=str(column)),
            user_df.iloc[:, user_df.columns.get_loc(column) + 1:]
        ], axis=1)

    user_df.head()

    return user_df

###################Typing Data####################

def read_tappy(file_name):
    df = pd.read_csv(
        file_name,
        delimiter=',',
        index_col=False,
        names=['Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time'],
        error_bad_lines=False
    )

    # df = df.drop('UserKey', axis=1)

    # df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%y%M%d').dt.date

    # converting time data to numeric
    #print(df[df['Hold time'] == '0105.0EA27ICBLF']) # for 0EA27ICBLF_1607.txt
    for column in ['Hold time', 'Latency time', 'Flight time']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(axis=0)

    # cleaning data in Hand
    df = df[
        (df['Hand'] == 'L') |
        (df['Hand'] == 'R') |
        (df['Hand'] == 'S')
    ]

    # cleaning data in Direction
    df = df[
        (df['Direction'] == 'LL') |
        (df['Direction'] == 'LR') |
        (df['Direction'] == 'LS') |
        (df['Direction'] == 'RL') |
        (df['Direction'] == 'RR') |
        (df['Direction'] == 'RS') |
        (df['Direction'] == 'SL') |
        (df['Direction'] == 'SR') |
        (df['Direction'] == 'SS')
    ]

    rows = ['LL', 'LR', 'LS', 'RL', 'RR', 'RS', 'SL', 'SR', 'SS']
    columns = ['Hold time', 'Latency time', 'Flight time']
    direction_group_df = df.groupby('Direction').mean()
    # print(direction_group_df)

    if len(direction_group_df.index) != len(rows):
        new_direction_group_df = pd.DataFrame(columns=columns)
        for index in rows:
            try:
                s = direction_group_df.loc[index]
                new_direction_group_df = new_direction_group_df.append(s)
            except KeyError:
                temp = pd.Series(data=[0, 0, 0], index=columns)
                temp.name = index
                new_direction_group_df = new_direction_group_df.append(temp)
        direction_group_df = new_direction_group_df
    del df
    gc.collect()

    return direction_group_df


def find_typing_file(user_set):
    """
    Return typing file name set
    :param user_set:
    :return:
    """
    typing_set = []

    for user in user_set:
        count = 0
        user_typing_set = []
        for file in os.listdir('./input/Archived-Data/Tappy Data/'):
            if fnmatch.fnmatch(file, user + "*"):
                user_typing_set.append(file)
                count = 1
        typing_set.append(user_typing_set)
    print(typing_set)
    print(len(typing_set))
    return typing_set


def generate_typing_df(typing_set):
    """
    Loops over all files to generate list of typing dataframe
    :param typing_set:
    :return:
    """
    np_array = np.array([[i for i in range(1, 28)]])


    temp_array = np.array([[i for i in range(1, 28)]])

    direction_group_df = read_tappy(typing_set)
    my_list = np.array([direction_group_df.values.flatten().tolist()])
    temp_array = np.append(temp_array, my_list, axis=0)

    # Delete first row and average out the values amongst rows
    temp_array = np.delete(temp_array, (0), axis=0)
    temp_array = np.mean(temp_array, axis=0)
    temp_array = temp_array.reshape((1, 27))

    # Append to row to main array
    np_array = np.append(np_array, temp_array, axis=0)

    print(np_array)
    print(np_array.shape)
    np_array = np.delete(np_array, (0), axis=0)

    return np_array


def main():
    random.seed(0)
    # Init user data
    user_set = init_user_set()
    print(user_set)

    user_df = generate_user_df(user_set=user_set)
    Y = user_df.loc[:]['Parkinsons']

    Y = Y.values
    Y = np.transpose(Y)
    print(Y.shape)
    print(user_df.head())

    # Init typing data

    typing_set = find_typing_file(user_set)
    if not os.path.isfile('X.csv'):
        X = generate_typing_df(typing_set)
        np.savetxt('X.csv', X, delimiter=',')
    else:
        X = np.genfromtxt('X.csv', delimiter=',')
    print(X.shape)
    print(Y.shape)

    # Normalize X
    X = normalize(X)
    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.tree import DecisionTreeClassifier

    clf1 = LogisticRegression(random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    clf3 = GaussianNB()
    clf4 = KNeighborsClassifier()
    clf5 = svm.SVC()
    clf6 = DecisionTreeClassifier()
    clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('knn', clf4),
                                       ('svm', clf5), ('dtc', clf6)], voting='hard')
    # clf = svm.SVC()
    clf.fit(X_train, y_train)

    print(clf.predict(X_test[0, :]))
    print(y_test[0])

    s = cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy')
    print(s.mean())



if __name__ == "__main__":
    # main()
    import csv

    with open("./user_csv/abc.csv", 'r') as f:
        reader = csv.reader(f)
        linenumber = 1
        try:
            for row in reader:
                linenumber += 1
        except Exception as e:
            print(("Error line %d: %s" % (linenumber, str(type(e)))))

    test_X = generate_typing_df('./user_csv/abc.csv')

    pass