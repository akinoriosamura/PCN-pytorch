
import os
import numpy.random as npr
import numpy as np

def assemble_data(output_file, anno_file_list=[]):

    #assemble the pos, neg, part annotations to one file
    if len(anno_file_list)==0:
        return 0

    if os.path.exists(output_file):
        os.remove(output_file)

    for anno_file in anno_file_list:
        with open(anno_file, 'r') as f:
            print(anno_file)
            anno_lines = f.readlines()

        chose_count = 0
        with open(output_file, 'a+') as f:
            for anno_line in anno_lines:
                # write lables of pos, neg, part images
                f.write(anno_line)
                chose_count+=1

    return chose_count
