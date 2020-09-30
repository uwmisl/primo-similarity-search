#!/bin/bash

dataset_dir=$1
if [ ! -d "$dataset_dir" ]; then
    echo "must provide a valid dataset directory"
    exit 1
fi

uid=$(id -u $SUDO_USER)

docker run \
    --rm -it -u $uid:$uid -p 8888:8888 --gpus all \
    -v $(pwd):/tf/primo \
    -v $dataset_dir:/tf/open_images \
    -e PYTHONPATH=/tf/primo \
    --name primo \
    primo
