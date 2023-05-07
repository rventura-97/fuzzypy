import pandas as pd

def load_occupancy():
    X_train = pd.read_csv('DATA/occupancy_data/datatraining.txt')
    X_test_1 = pd.read_csv('DATA/occupancy_data/datatest.txt')
    X_test_2 = pd.read_csv('DATA/occupancy_data/datatest2.txt')
    X_test = pd.concat((X_test_1,X_test_2))
    data = {'X_train':X_train.iloc[:,1:6].to_numpy(),\
            'y_train':X_train.iloc[:,-1].to_numpy(),\
            'X_test':X_test.iloc[:,1:6].to_numpy(),\
            'y_test':X_test.iloc[:,-1].to_numpy()}
    return data

def load_optical_digits():
    X_train = pd.read_csv('DATA/opt_digits_data/optdigits.tra',header=None)
    X_test = pd.read_csv('DATA/opt_digits_data/optdigits.tes',header=None)
    data = {'X_train':X_train.iloc[:,0:-1].to_numpy(),\
            'y_train':X_train.iloc[:,-1].to_numpy(),\
            'X_test':X_test.iloc[:,0:-1].to_numpy(),\
            'y_test':X_test.iloc[:,-1].to_numpy()}
    return data

def load_letter_recognition():
    X = pd.read_csv('DATA/letter-recognition.data',header=None)
    return 0