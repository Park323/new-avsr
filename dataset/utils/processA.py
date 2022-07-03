from dataset.utils.generate_label import generate_script_from_path
from dataset.utils.match_split_files import match
from dataset.utils.sort_labels import sort_it
from dataset.utils.explore_dataset import *

def gogo(path, split):
    path = generate_script_from_path(path, unit='grapheme')
    match(path, split)
    sort_it(path.replace(".txt", f"_Train.txt"))
    get_file_size(path.replace(".txt", f"_Train.txt"),path.replace(".txt", f"_Train_max.txt"),limit=16)