import argparse
import os
import sys
from random import randint
from itertools import islice


# import torch
def main(args):
    path_to_toyset = args.path_to_toyset #"./data/NYUv2-raw/train_test_subset"
    path_to_realset = args.path_to_realset#"./data/NYUv2-raw/train_test_inputs"

    scenes = args.scenes.replace(" ", "").split(',')

    lines_to_insert = []

    random_subset = args.random
    subset_percentage = args.subset_percentage #0.15

    if random_subset:
        line_ids = []
        count = 0
        for traintest_name in ["nyudepthv2_train_files_with_gt.txt", "nyudepthv2_test_files_with_gt.txt"]:
            with open(os.path.join(path_to_realset, traintest_name), 'r') as traintest:
                lines = traintest.readlines()
                max_length = int(subset_percentage * len(lines))
                print(max_length)
                while count < max_length:
                    id = randint(0, len(lines))
                    # print(id)
                    if id not in line_ids:
                        line_ids.append(id)
                        lines_to_insert.append(lines[id])
                        count += 1
            with open(os.path.join(path_to_toyset, traintest_name), 'w') as output:
                for line in lines_to_insert:
                    output.write(line)
            lines_to_insert = []
            line_ids = []
            count = 0


    else:
        for traintest_name in ["nyudepthv2_train_files_with_gt.txt", "nyudepthv2_test_files_with_gt.txt"]:
            with open(os.path.join(path_to_realset, traintest_name), 'r') as traintest:
                for line in traintest.readlines():
                    for scene in scenes:
                        if scene in line:
                            lines_to_insert.append(line)

            with open(os.path.join(path_to_toyset, traintest_name), 'w') as output:
                for line in lines_to_insert:
                    output.write(line)
            lines_to_insert = []

    """
    #Deprecated
        path_to_toyset = "./data/NYUv2-raw/train_test_subset"
        path_to_realset = "./data/NYUv2-raw/train_test_inputs"
        scenes = ['bathroom']
        subscenes = []
        lines_to_insert = []

        val_folder = "./data/NYUv2-raw/official_splits/train"

        #create train_val splits
        with open(os.path.join(path_to_realset, "nyudepthv2_train_files_with_gt.txt"), 'r') as train_whole:
            lines = train_whole.readlines()
            for line in lines:
                for scene in scenes:
                    if scene in line:
                        sub = line.split('/')[1]
                        if sub not in subscenes:
                            subscenes.append(sub)
            #print(subscenes)

            #take val subset of scenes

            val_percentage = 0.1
            val_size = int(len(subscenes) * val_percentage)

            val_scenes = []
            train_scenes = []
            val_counter = 0
            for scene in subscenes:
                if(val_counter <= val_size):
                    val_scenes.append(scene)
                    val_counter += 1
                else:
                    train_scenes.append(scene)

        with open(os.path.join(path_to_realset, "nyudepthv2_test_files_with_gt.txt"), 'r') as test:
            lines = test.readlines()
            for line in lines:
                for scene in scenes:
                    if scene in line:
                        lines_to_insert.append(line)
        with open(os.path.join(path_to_toyset, "nyudepthv2_test_files_with_gt.txt"), 'w') as output:
            for line in lines_to_insert:
                output.write(line)

        lines_to_insert = []

        for trainval_name in ["nyudepthv2_val_files_with_gt.txt", "nyudepthv2_train_files_with_gt.txt", "nyudepthv2_test_files_with_gt.txt"]:
            with open(os.path.join(path_to_realset, trainval_name), 'r') as trainvaltest:
                for line in trainvaltest.readlines():
                    scene_pool = train_scenes
                    if trainval_name == "nyudepthv2_test_files_with_gt.txt":
                        scene_pool = scenes
                    elif trainval_name == "nyudepthv2_val_files_with_gt.txt":
                        scene_pool = val_scenes
                    for scene in scene_pool:
                        if scene in line:
                            lines_to_insert.append(line)

            with open(os.path.join(path_to_toyset, trainval_name), 'w') as output:
                for line in lines_to_insert:
                    output.write(line)
            lines_to_insert = []
        """
    """
    for dir in os.listdir(val_folder):
        #print(dir)
        if dir in val_scenes:
            dir_path = os.path.join(val_folder, dir)
            rgbs, depths = split_list_in_half(os.listdir(dir_path))
            #print(rgbs)
            for i in range(int(len(os.listdir(dir_path)) // 2)):
                line = "/" + os.path.join(dir, rgbs[i]) + " " + os.path.join(dir, depths[i]) + " " + str(518.8579)+"\n"
                lines_to_insert.append(line)

    with open(os.path.join(path_to_toyset, "nyudepthv2_val_files_with_gt.txt"), 'w') as output:
        for line in lines_to_insert:
            output.write(line)
        #write val and train
    """
    # print(torch.cuda.get_device_name(1))


def split_list_in_half(original):
    half = len(original) // 2
    return original[:half], original[half:]


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates a subset of NYUv2. Default values result in subset only including Bathroom images',
        fromfile_prefix_chars='@',
        conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--path_to_toyset', default="./data/NYUv2-raw/train_test_subset", type=str,
                        help='folder to the text file of the toy NYUv2 dataset (train and test split) that should be written to')
    parser.add_argument('--path_to_realset', default="./data/NYUv2-raw/train_test_inputs", type=str,
                        help='folder to the text file of the complete NYUv2 dataset (train and test split) -> where to read from')
    parser.add_argument('--random', default=False, type=bool,
                        help='if set will make a random subset across all scenes according to subset_percentage')
    parser.add_argument('--subset_percentage', default=0.15, type=float,
                        help='percentage of the new dataset size relative to entire dataset')
    parser.add_argument('--scenes', default='bathroom', type=str,
                        help='the scenes to include in the toydataset, write scene names separated by comma (setting ignored when random is True)')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    main(args)
