import argarse, os, datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def process(example, idx, split, data_source, instruction):

    return 