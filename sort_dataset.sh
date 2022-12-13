#!/bin/bash

TRAIN_DIR="$(pwd)/../data/train"

cd ../data/hybrid-renderer/

i=0
j=0
k=0
# in the hybrid-renderer
for d in */ ; do
    cd $d
    # in each ai_0XX_XXX
    for sub_d in */ ; do
        cd $sub_d
        for f in *.color.jpg ; do
            cp $f "$TRAIN_DIR/gt_$i.jpg"
            ((i++))

            if ((i % 100 == 0 )) 
            then 
                echo "Done copying gt $i"
            fi
        done 

        for f in *.diffuse_reflectance.jpg ; do
            cp $f "$TRAIN_DIR/albedo_$j.jpg"
            ((j++))
            if ((j % 100 == 0 )) 
            then 
                echo "Done copying gt $j"
            fi
        done 

        for f in *.low_spp_rt.jpg ; do
            cp $f "$TRAIN_DIR/low_spp_rt_$k.jpg"
            ((k++))

            if ((k % 100 == 0 )) 
            then 
                echo "Done copying gt $k"
            fi
        done 

        cd ../ 
    done 
    cd ../
done

echo $i
