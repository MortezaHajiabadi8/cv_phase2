import os

os.chdir('./dataset')
root_path = os.getcwd()

digit_folders = [str(i*100) for i in range(10)]
letter_folders = [str(i+1) for i in range(32)]
folders = digit_folders + letter_folders

for folder in folders:
    os.mkdir(os.path.join(root_path,folder))
    
