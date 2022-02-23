#!/bin/sh
k=1200
while [ "$k" -le 1400 ]; do
minSize=1
    while [ "$minSize" -le 100 ]; do 
        ./build/default/default/patch_based_resizing --SegmentK "$k" --MinSize "$minSize"
        minSize=$(( minSize + 3 ))
    done 
k=$(( k + 5 ))
done 