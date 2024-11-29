#!/bin/sh


for y in {2000..2007}
do

    y2=`expr $y + 1`
    echo ${y}060100 ${y2}063023
    do.nodes.sh $y o9
    
done


for y in {2008..2015}
do

    y2=`expr $y + 1`
    echo ${y}060100 ${y2}063023
    do.nodes.sh $y o10
    
done


for y in {2016..2023}
do

    y2=`expr $y + 1`
    echo ${y}060100 ${y2}063023
    do.nodes.sh $y o11
    
done


for y in {1980..1987}
do

    y2=`expr $y + 1`
    echo ${y}060100 ${y2}063023
    do.nodes.sh $y o12
    
done


for y in {1991..1993}
do

    y2=`expr $y + 1`
    echo ${y}060100 ${y2}063023
    do.nodes.sh $y o1
    
done


for y in {1988..1990}
do

    y2=`expr $y + 1`
    echo ${y}060100 ${y2}063023
    do.nodes.sh $y o2
    
done

for y in {1997..1999}
do

    y2=`expr $y + 1`
    echo ${y}060100 ${y2}063023
    do.nodes.sh $y o3
    
done

for y in {1994..1996}
do

    y2=`expr $y + 1`
    echo ${y}060100 ${y2}063023
    do.nodes.sh $y o4
    
done


exit 0
