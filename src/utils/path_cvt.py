""" Path config"""


def get_path(line_id):
    with open('paths.config') as f:
        lines = f.readlines()
    return lines[line_id].strip()


def get_path_to_train_dataset():
    return get_path(line_id=0)


def get_path_to_eval_dataset():
    return get_path(line_id=1)


def get_path_to_ckpts():
    return get_path(line_id=2)


def get_path_to_vis_ims():
    return get_path(line_id=3)


def get_path_to_FDDB():
    return get_path(line_id=4)


def get_path_to_WIDERFace():
    return get_path(line_id=5)


if __name__ == '__main__':
    print(get_path_to_vis_ims())