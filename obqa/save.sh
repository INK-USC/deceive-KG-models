#!/bin/sh

if [ ! -d $1 ]; then
  mkdir $1
fi

if ls *.py>/dev/null 2>&1; then
  echo 'py exist'
  cp *.py $1
fi
if ls *.csv>/dev/null 2>&1; then
  cp *.csv $1
fi
if ls *.txt>/dev/null 2>&1; then
  cp *.txt $1
fi
