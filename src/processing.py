import glob
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocessing(raw_dir, data_dir, train_dir, test_dir):
    scaler = StandardScaler()
    x_train = np.load(os.path.join(raw_dir, 'x_train.npy'))
    scaler.fit(x_train)

    input_files = glob.glob('{}/raw/*.npy'.format(data_dir))
    print('\nINPUT FILE LIST: \n{}\n'.format(input_files))
    for file in input_files:
        raw = np.load(file)
        # only transform feature columns
        if 'y_' not in file:
            transformed = scaler.transform(raw)
        if 'train' in file:
            if 'y_' in file:
                output_path = os.path.join(train_dir, 'y_train.npy')
                np.save(output_path, raw)
                print('SAVED LABEL TRAINING DATA FILE\n')
            else:
                output_path = os.path.join(train_dir, 'x_train.npy')
                np.save(output_path, transformed)
                print('SAVED TRANSFORMED TRAINING DATA FILE\n')
        else:
            if 'y_' in file:
                output_path = os.path.join(test_dir, 'y_test.npy')
                np.save(output_path, raw)
                print('SAVED LABEL TEST DATA FILE\n')
            else:
                output_path = os.path.join(test_dir, 'x_test.npy')
                np.save(output_path, transformed)
                print('SAVED TRANSFORMED TEST DATA FILE\n')

