#!/bin/bash

while getopts ":d:s:" Option
do
    case $Option in
        d ) dataset_dir=$OPTARG;;
        s ) sequencing_dir=$OPTARG;;
        * ) echo "invalid option specified"; exit 1;;
    esac
done

error=0
mounts=""
if [ -d "$dataset_dir" ]; then
    mounts="$mounts -v $(realpath $dataset_dir):/tf/open_images"
elif [ -n "$dataset_dir" ]; then
    echo "dataset directory is invalid!"
    error=1
fi

if [ -d "$sequencing_dir" ]; then
    mounts="$mounts -v $(realpath $sequencing_dir):/tf/sequencing"
elif [ -n "$sequencing_dir" ]; then
    echo "sequencing directory is invalid!"
    error=1
fi

uid=$(id -u $SUDO_USER)

docker run \
    --rm -it -u $uid:$uid -p 8888:8888 --gpus all \
    -v $(pwd):/tf/primo \
    $mounts \
    -e PYTHONPATH=/tf/primo \
    --name primo \
    primo
