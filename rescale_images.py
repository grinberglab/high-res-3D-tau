import os
import subprocess
import sys 
import fnmatch
import tifffile

def get_img_info(root_dir):
    file_list = {}

    for root, dir, files in os.walk(root_dir):
        if fnmatch.fnmatch(root,'*/RES(*'): #verifies that it's inside /RES*
                for fn in fnmatch.filter(files,'*_*_*.tif'): #makes sure to fetch only full resolution images
                    if fn.find('res10') == 0: #skip res10 images
                            continue
                    file_name = os.path.join(root,fn)
                    file_list[fn] = root
                    print(fn)

    return file_list

def convert(root_dir):
    #create Image Magick tmp directory
    home_dir = os.getcwd()

    TMP_DIR = os.path.join(root_dir, "magick_tmp")
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR, 0777)

    #export Image Magick env variables
    os.environ['MAGICK_TMPDIR'] = TMP_DIR
    os.environ['MAGICK_TMPDIR'] = '24Gb'

    #get file info for all unprocessed tif files
    image_files = get_img_info(root_dir)
    
    #iterate over files in dict
    for input_fn in image_files.keys():
        output_fn = 'res10_' + input_fn
        #enter image directory
        os.chdir(image_files[input_fn])
        if os.path.isfile(output_fn):
            print('This image has already been resized.')
            os.chdir(home_dir)
            continue
        
        #create file to be written to
        temp = open(output_fn, 'w')
        
        #do rescaling
        subprocess.call(['convert', input_fn, '-resize', '10%', output_fn], env=dict(os.environ))
        os.chdir(home_dir)

def main():
    #check for user input
    if (len(sys.argv) == 2):
        root_dir = sys.argv[1]
        convert(root_dir)
    else: 
        print("Use: enter one directory")

if __name__ == "__main__":
    main()
