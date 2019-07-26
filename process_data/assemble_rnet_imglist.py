import os
import sys
sys.path.append(os.getcwd())
import mtcnn.data_preprocess.assemble as assemble

rnet_postive_file = './dataset/anno_store/pos_24.txt'
rnet_part_file = './dataset/anno_store/part_24.txt'
rnet_neg_file = './dataset/anno_store/neg_24.txt'
imglist_filename = './dataset/anno_store/imglist_anno_24.txt'

if __name__ == '__main__':

    anno_list = []

    anno_list.append(rnet_postive_file)
    anno_list.append(rnet_part_file)
    anno_list.append(rnet_neg_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
    print("data num: ", chose_count)
