import common

def return_something():
    root_data = common.STHV1_FRAMES
    filename_categories = 'data/somethingv1/classInd.txt'
    filename_imglist_train = "data/somethingv1/train_split.txt"
    filename_imglist_val = "data/somethingv1/validation_split.txt"
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data


def return_somethingv2():
    root_data = common.STHV2_FRAMES
    filename_categories ='data/somethingv2/classInd.txt'
    filename_imglist_train = "data/somethingv2/train_videofolder.txt"
    filename_imglist_val = "data/somethingv2/val_videofolder.txt"
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data

def return_minik():
    root_data = common.MINIK_FRAMES
    filename_categories = 'data/kinetics/minik_classInd.txt'
    filename_imglist_train = 'data/kinetics/mini_train_videofolder.txt'
    filename_imglist_val = 'data/kinetics/mini_val_videofolder.txt'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data


def return_jester():
    root_data = common.JESTER_FRAMES
    filename_categories = 'data/jester/classInd.txt'
    filename_imglist_train = 'data/jester/train_split.txt'
    filename_imglist_val = 'data/jester/validation_split.txt'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data

def return_dataset(dataset, data_path):
    dict_single = {'something': return_something,
                   'somethingv2': return_somethingv2,
                   'jester': return_jester,
                   'minik': return_minik}
    common.set_manual_data_path(data_path, None)
    file_categories, file_imglist_train, file_imglist_val, root_data = dict_single[dataset]()

    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    n_class = len(categories)
    prefix = '{:05d}.jpg'
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
