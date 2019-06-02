import os
from os.path import isfile

def get_partition_and_labels():
    list_of_dirs=os.listdir('./cifar/train')
    classes = [f for f in list_of_dirs if not isfile(f)]
    classes = sorted(classes)
    partition = { 'train' : [] , 'validation' : []}
    labels = {}
    for part in ['train','validation'] :
        for index_of_class,single_class in enumerate(classes):
            list_of_samples = os.listdir('./cifar/'+str(part)+'/'+str(single_class))
            partition[part].extend(list_of_samples)
            for i in list_of_samples:
                labels[i]=index_of_class
    return partition , labels



'''
print(len(partition['train']))
print(len(partition['validation']))
print(len(labels))
'''
#print(labels)
'''
batch_size=32
batch_by_class = math.floor(batch_size/10)
batch_by_class_rem = batch_size%10

for i in range(0,32):

#print(len(classwise_samples[0]))
'''