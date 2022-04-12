#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/11/12 13:08
# @Author      :weiz
# @ProjectName :vibe
# @File        :video.py
# @Description :
# Copyright (C) 2021-2025 Jiangxi Institute Of Intelligent Industry Technology Innovation
import cv2

cap = cv2.VideoCapture("rtsp://192.168.10.163:554/user=admin&password=&channel=1&stream=0.sdp?")
cv2.namedWindow("ret", 0)

while True:
    ret, frame = cap.read()
    cv2.imshow("ret", frame)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break