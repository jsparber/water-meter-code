#!/bin/bash
#Take a picture, then push it to a remote webserver

# Enable flash
echo "18" > /sys/class/gpio/export
echo "out" > /sys/class/gpio/gpio18/direction

# Turn on the led connected to the pin 18 of the RPI
echo "1" > /sys/class/gpio/gpio18/value
sleep 2
#Take a photo
fswebcam -d v4l2:/dev/video0 -r 640x480 -D 10 -F 100 -S 50 --no-banner /home/alarm/img/$(date +\%Y\%m\%d\%H\%M).jpeg &> /home/alarm/log

sleep 2

# Turn of the led after the picture
echo "0" > /sys/class/gpio/gpio18/value

# Workaround: The webcam stopes working after some time, if that happes we reboot the RPI
string=`cat /home/alarm/log | grep "VIDIOC_STREAMON: Broken pipe"`

if [ ! -z "$string" ]; then
	reboot
else
	NEW_JPEG=$(ls /home/alarm/img -t | grep '\>.jpeg' | head -1)
	su alarm -c "/home/alarm/code/measure.sh $NEW_JPEG"
fi
