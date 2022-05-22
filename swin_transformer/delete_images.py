import os

files = os.listdir('/home/users/shunyaox/dataset/data/0')
for file in files[1000:]:
    os.remove('/home/users/shunyaox/dataset/data/0/' + file)
    
files = os.listdir('/home/users/shunyaox/dataset/data/1')
for file in files[1000:]:
    os.remove('/home/users/shunyaox/dataset/data/1/' + file)
    
files = os.listdir('/home/users/shunyaox/dataset/data/2')
for file in files[1000:]:
    os.remove('/home/users/shunyaox/dataset/data/2/' + file)