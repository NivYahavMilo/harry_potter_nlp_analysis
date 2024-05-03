import os


def load_dataset(data_file: [os.PathLike, str]):
    root_dir = os.path.abspath(os.path.curdir)
    file_path = os.path.join(root_dir, "datasets", data_file)
    with open(file_path, 'r') as txt_file:
        data = txt_file.read()

    return data
