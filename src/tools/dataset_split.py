""" Split train-eval sets and print to a text files
    The randomness should be deterministic with a fixed seed
"""

import glob
import random
import os

SPLIT_RATIO = 0.8
SEED = 17


def split():
    ds_dir = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/Datasets/thermal_face/ds/'
    train_list_path = 'train.txt'
    eval_list_path = 'eval.txt'

    paths = glob.glob(ds_dir + '/*.hdf5')
    paths = [os.path.basename(p) for p in paths]

    # sort the path alphabetically
    paths.sort()

    # shuffle the paths
    random.Random(SEED).shuffle(paths)

    n_train_samples = int(SPLIT_RATIO * len(paths))

    train_paths = paths[:n_train_samples]
    eval_paths = paths[n_train_samples:]

    with open(train_list_path, 'w') as f:
        for p in train_paths:
            f.writelines('{}\n'.format(p))

    with open(eval_list_path, 'w') as f:
        for p in eval_paths:
            f.writelines('{}\n'.format(p))
            
    print('Finish splitting train-eval sets.\nTrain: {} | Eval: {}'.format(len(train_paths), len(eval_paths)))


if __name__ == '__main__':
    split()





