import sys
import os
import shutil
import re
import glob


TAG_STR = 'InitialTransformParametersFileName'
NEW_TAG_STR = '(InitialTransformParametersFileName "{}/TransformParameters.{}.txt")'
NO_TFORM_STR = 'NoInitialTransform'
TFORM_FILE_STR = 'TransformParameters.?.txt'

#(InitialTransformParametersFileName "/home/ssatrawada/Desktop/AV2AT100Registrations/276/1stStep:Automatic/TransformParameters.0.txt")

def edit_txt(file_name,new_path):
    bkp_file_name = file_name+'.bkp'
    tmp_file_name = file_name+'.tmp'

    #get file tranfs number
    basename = os.path.basename(file_name)
    idx = []
    for r in re.finditer('\.',basename):
        idx.append(r.start())
    file_num = int(basename[idx[0]+1:idx[1]])

    shutil.copyfile(file_name,bkp_file_name)
    with open(file_name,'r') as txt:
        with open(tmp_file_name,'w+') as tmp:
            for line in txt:
                if line.find(TAG_STR) > -1:
                    if line.find(NO_TFORM_STR) <= -1:
                        line = NEW_TAG_STR.format(new_path,file_num-1)+'\n'
                tmp.write(line)
                tmp.flush()

    shutil.move(tmp_file_name,file_name)

def get_tform_list(tform_dir):
    tform_list = glob.glob(os.path.join(tform_dir,TFORM_FILE_STR))
    return tform_list

def change_tform_files(tform_dir,new_path):
    file_list = get_tform_list(tform_dir)
    for file_name in file_list:
        print('Editing file {}'.format(file_name))
        edit_txt(file_name,new_path)


def main():
    if len(sys.argv) != 3:
        print('Usage: elastix_edit_transf_path.py <transf_files_dir> <new_path>')
        exit()

    files_dir = sys.argv[1]
    new_path = sys.argv[2] #histology or heatmap
    change_tform_files(files_dir,new_path)


if __name__ == '__main__':
    main()