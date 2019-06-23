#!/bin/bash

i=0
for fi in *_dog.png; do
    mv "$fi" $i.png
    i=$((i+1))
done