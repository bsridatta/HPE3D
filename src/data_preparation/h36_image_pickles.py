import os
import numpy as np
import joblib
import glob
from tqdm import tqdm
from PIL import Image


if __name__ == "__main__":
    root = "/home/datta/lab/HPE_datasets"
    if not os.path.isdir(f'{root}/h36m_pickles'):
        os.mkdir(f'{root}/h36m_pickles')

    files = glob.glob(f'{root}/h36m/**/*')
    for file in tqdm(files):
        if file.split(".")[-1] != 'jpg':
            continue
        image_dir = file.split('/')[-2]
        image_id = file.split('/')[-1].split(".")[0]
        dir_ = f'{root}/h36m_pickles/{image_dir}'
        if not os.path.isdir(dir_):
            os.mkdir(dir_)

        image = Image.open(file)
        image = np.array(image)

        joblib.dump(image, f'{dir_}/{image_id}.pkl')

    print("DONE!")