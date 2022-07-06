import os
import shutil

opj = os.path.join


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def refresh_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)


def list_delta_dir(from_dir, wip_dir):
    final_lists = os.listdir(from_dir)
    current_lists = os.listdir(wip_dir)
    list_delta = len(final_lists) - (len(final_lists)-len(current_lists)) - 3

    return final_lists[list_delta:]
