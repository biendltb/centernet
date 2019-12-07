import glob
import h5py
import os
import cv2
import numpy as np


def cvt():
    """ Steps:
    + Extract the base name of the image with the frame number
    + Access to the corresponding thermal video and get exact the frame + number
    + Save the raw data in HDF5 format

    """
    old_ds_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/Datasets/thermal_face/old_ds/'
    new_ds_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/Datasets/thermal_face/ds/'
    thermal_ds = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/HD/thermal/'

    label_paths = glob.glob(old_ds_path + '/*.txt')

    for l in label_paths:
        base_name = os.path.basename(l)[:-4]
        for i in range(len(base_name)):
            if base_name[i].isdigit():
                break
        vid_name = base_name[:i]
        frame_id = base_name[i:]

        # read the label
        with open(l) as f:
            lines = f.readlines()
        label_str = lines[0].strip()

        # create the path to YAML file
        yaml_path = '{}/{}/set1_0.yml'.format(thermal_ds, vid_name)

        if os.path.isfile(yaml_path):
            fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
            node_key = 'rawImage{}'.format(frame_id)
            mat = fs.getNode(node_key).mat()

            ds_key = 'image'.format(i)
            label_key = 'label'
            h5_path = '{}/{}_{}.hdf5'.format(new_ds_path, vid_name, frame_id)
            with h5py.File(h5_path, 'w') as f:
                ds = f.create_dataset(name=ds_key, data=mat, dtype=np.uint16)
                ds.attrs[label_key] = label_str
        else:
            print('ERROR: File {} not found.'.format(yaml_path))


if __name__ == '__main__':
    # cvt()
    pass



