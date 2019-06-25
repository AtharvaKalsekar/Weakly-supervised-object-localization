#!/bin/bash

# replace "_dog" with respective class name
i=0
for fi in *_dog.png; do
    mv "$fi" $i.png
    i=$((i+1))
done