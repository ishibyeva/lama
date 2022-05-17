#!/bin/bash

git clone https://github.com/ishibyeva/lama.git;
cd lama;
export TORCH_HOME=$(pwd) && export PYTHONPATH=.;
pip3 install wldhx.yadisk-direct;
curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip;
unzip big-lama.zip;
pytest bin/small_test.py $(pwd)/big-lama $(pwd)/Small_Test;