#!/bin/bash
for num in $(seq 1 287)
do
  echo $num
  python main_select.py --clf $1 --select $2 --num $num --nor mean
done
