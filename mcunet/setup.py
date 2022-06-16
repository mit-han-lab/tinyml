#!/usr/bin/env python
import os, sys
import shutil
import datetime

from setuptools import setup, find_packages
from setuptools.command.install import install

readme = open('README.md').read()
readme = '''
# MCUNet: Tiny Deep Learning on IoT Devices 

###  [website](http://mcunet.mit.edu/) | [paper](https://arxiv.org/abs/2007.10319) | [demo](https://www.youtube.com/watch?v=YvioBgtec4U&feature=emb_logo)


## News

- **(2022/06)** We refactor the MCUNet repo as a standalone repo (previous repo: https://github.com/mit-han-lab/tinyml)
- **(2021/10)** Checkout our new paper **MCUNetV2**: https://arxiv.org/abs/2110.15352 !
- Our projects are covered by: [MIT News](https://news.mit.edu/2020/iot-deep-learning-1113), [WIRED](https://www.wired.com/story/ai-algorithms-slimming-fit-fridge/), [Morning Brew](https://www.morningbrew.com/emerging-tech/stories/2020/12/07/researchers-figured-fit-ai-ever-onto-internet-things-microchips), [Stacey on IoT](https://staceyoniot.com/researchers-take-a-3-pronged-approach-to-edge-ai/), [Analytics Insight](https://www.analyticsinsight.net/amalgamating-ml-and-iot-in-smart-home-devices/), [Techable](https://techable.jp/archives/142462), etc.

'''
VERSION = "0.1.1"

requirements = [
    "torch",
    "torchvision"
]

# import subprocess
# commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True).decode('UTF-8').rstrip()
# VERSION += "_" + str(int(commit_hash, 16))[:8]
VERSION += "_" + datetime.datetime.now().strftime("%Y%m%d%H%M")

setup(
    # Metadata
    name="mcunet",
    version=VERSION,
    author="MTI HAN LAB ",
    author_email="hanlab.eecs+github@gmail.com",
    url="https://github.com/mit-han-lab/mcunet",
    description="MCUNet: Tiny Deep Learning on IoT Devices",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    # Package info
    packages=find_packages(exclude=("*test*",)),
    #
    zip_safe=True,
    install_requires=requirements,
    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
