import os
import numpy as numpy
import joblib
import glob
from tqdm import tqdm


if __name__ == "__main__":

    try:
        # High performance image loader from PyTorch
        import accimage as Image
    except ImportError as error:
        import PIL as Image

    # files = glob.glob(f'/home/datta/lab/HPE_datasets/h36m/')
    files = glob.glob(f'/home/datta/lab/HPE_datasets/h36m/s_05_act_06_subact_01_ca_01/*')
    for file in tqdm(files):
        image_tmp = Image.open(file)
        
        joblib.dump(image_array[idx, :], f'{os.path.dirname(os.getcwd())}/input/image_pickles/{img_id}.pkl')

        print(image_tmp.shape)


        exit()
        # image_array = df.values
        # for idx, img_id in tqdm(enumerate(image_ids)):
        #     joblib.dump(image_array[idx, :], f'{os.path.dirname(os.getcwd())}/input/image_pickles/{img_id}.pkl')
