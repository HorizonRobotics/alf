#!/bin/bash

while getopts f:t: flag
do
    case "${flag}" in
        f) Dockerfile=${OPTARG};;
        t) tag=${OPTARG};;
    esac
done
echo "Building docker image: $tag from Dockerfile: $Dockerfile";

docker build -f $Dockerfile --tag $tag .
