import os
from os.path import isfile

def get_partition_and_labels():
    list_of_dirs=os.listdir('./cifar/train')
    classes = [f for f in list_of_dirs if not isfile(f)]
    classes = sorted(classes)
    #print(classes)
    partition = { 'train' : [] , 'validation' : []}
    labels = {}
    for part in ['train','validation'] :
        for index_of_class,single_class in enumerate(classes):
            list_of_samples = os.listdir('./cifar/'+str(part)+'/'+str(single_class))
            filtered_list = []
            for i in list_of_samples:
                if i not in labels:
                  labels[i]=index_of_class
                  filtered_list.append(i)
            partition[part].extend(filtered_list)
            #print(part,single_class,len(list_of_samples),len(filtered_list),len(labels))
    return partition , labels


