#!/usr/local/bin/python

## 
## Created by: Grant McGovern 
## Date: 30 March 20145
## Purpose: Setup environment for Lab 3.
## 

from setuptools import setup, find_packages

print "Setting up environment...\n"

## Get our requirements from our .txt file
with open('requirements.txt') as requirements:
	modules = [line.strip('\n') for line in requirements]

print "Install pip modules... (This might take awhile)\n"

setup(name = 'Lab3',
	description = 'Exploring the Fast Fourier Transformation.',
	author = 'Grant McGovern',
	author_email = 'mcgoga12@wfu.edu',
	install_requires = modules
)
