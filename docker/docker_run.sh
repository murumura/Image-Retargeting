#!/bin/bash
container_id=$(docker ps -aq --filter name=resizing_container) 
if [ ! -z "$container_id" ]
then
    docker rm ${container_id}
fi
docker run  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it \
--privileged -v /dev/bus/usb:/dev/bus/usb \
-v $(pwd):/home/ImageResize \
-p 1235:8888 \
 --name resizing_container resizing-dev

