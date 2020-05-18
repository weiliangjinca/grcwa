#!/usr/bin/env python

"""The setup script."""
import os,re,codecs
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

here = os.path.abspath(os.path.dirname(__file__))
def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()
    
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')

setup(
    author="Weiliang Jin",
    author_email='jwlaaa@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Rigorous coupled wave analysis supporting automatic differentation with autograd",
    install_requires=['autograd','numpy'],
    license="GPL license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='grcwa',
    name='grcwa',
    packages=find_packages(),
    url='https://github.com/weiliangjinca/grcwa',
    version=find_version('grcwa', '__init__.py'),
)
