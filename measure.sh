#!/bin/bash

new_value=$1
cd /home/alarm/water-meter-data/
git pull origin master

mv data.js data.tmp
echo "var dataSeries = [" > data.js
cat data.tmp | grep "date" >> data.js
echo $new_value >> data.js
echo "]" >> data.js
rm data.tmp



git commit -am "update data"
git push origin master
