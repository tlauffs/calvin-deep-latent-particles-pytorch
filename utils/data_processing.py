import os
import numpy as np
from tqdm import tqdm


root_src_folder = '/media/tim/E/datasets/task_D_D/'
root_parse_folder = '/media/tim/D/datasets_reduced/D_D_door/'

for i in ['training', 'validation']:
    src_folder = f'{root_src_folder}{i}'
    parse_folder = f'{root_parse_folder}{i}'
    caption_path = f'{src_folder}/lang_annotations/auto_lang_ann.npy'

    annotations = np.load(f"{caption_path}", allow_pickle=True).item()
    annotations = list(zip(annotations["info"]["indx"], annotations["language"]["task"]))
    tasks_to_extract = ['move_slider_left', 'move_slider_right']

    for idx, annotation in tqdm(enumerate(annotations)):
        task = annotation[1]
        if task in tasks_to_extract:
            index = annotation[0]
            for i in range(index[0], index[1]+1):
                data_episode_path = os.path.join(f'{src_folder}/episode_{i:07d}.npz')
                data = np.load(data_episode_path, allow_pickle=True)     
                img_static = data['rgb_static']
                processed_episode_path = data_episode_path = os.path.join(f'{parse_folder}/episode_{i:07d}.npz')
                np.savez(processed_episode_path,rgb_static=img_static)