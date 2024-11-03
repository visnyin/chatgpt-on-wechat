#!/bin/bash

unset KUBECONFIG

cd .. && docker build -f docker/Dockerfile.latest \
             -t visnyin/chatgpt-on-wechat .

docker tag visnyin/chatgpt-on-wechat visnyin/chatgpt-on-wechat:$(date +%y%m%d)