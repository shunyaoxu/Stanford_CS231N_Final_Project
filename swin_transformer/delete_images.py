import os

files = os.listdir('/home/ubuntu/dataset/0')
for file in files[6000:]:
    os.remove(file)