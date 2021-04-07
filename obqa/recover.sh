#!/bin/sh


if ls $1/*.py >/dev/null 2>&1; then
  cp $1/*.py ./
fi
if ls $1/*.csv >/dev/null 2>&1; then
  cp $1/*.csv ./
fi
if ls $1/*.txt >/dev/null 2>&1; then
  cp $1/*.txt ./
fi