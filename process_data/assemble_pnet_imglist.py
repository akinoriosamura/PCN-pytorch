import os
import sys
sys.path.append(os.getcwd())
import process_data.assemble as assemble

pnet_postive_file = './dataset/anno_store/pos_12.txt'
pnet_part_file = './dataset/anno_store/part_12.txt'
pnet_neg_file = './dataset/anno_store/neg_12.txt'
imglist_filename = './dataset/anno_store/imglist_anno_12.txt'

if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
    print("data num: ", chose_count)
