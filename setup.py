#  Copyright (C) 2013 Matt Hagy <hagy@gatech.edu>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

name = 'sputil'
version = '0.0.1'

import sys
import os

from distutils.core import setup, Extension
from distutils.sysconfig import customize_compiler
from distutils.ccompiler import new_compiler

os.chdir(os.path.dirname(__file__) or os.getcwd())

setup(
    name=name,
    version=version,
    url='https://github.com/matthagy/sputil',
    author='Matt Hagy',
    author_email='hagy@gatech,.edu',
    description='scipy sparse matrices utilities',
    classifiers = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    'Topic :: Education',
    'Topic :: Utilities'
    ],
    packages = ['sputil'],
    )
