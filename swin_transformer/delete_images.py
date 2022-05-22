import os

files = os.listdir('/home/ubuntu/dataset/0')
for file in files[2000:]:
    os.remove('/home/ubuntu/dataset/0/' + file)
    
files = os.listdir('/home/ubuntu/dataset/1')
for file in files[2000:]:
    os.remove('/home/ubuntu/dataset/1/' + file)
    
files = os.listdir('/home/ubuntu/dataset/2')
for file in files[2000:]:
    os.remove('/home/ubuntu/dataset/2/' + file)