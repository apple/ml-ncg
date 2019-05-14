from setuptools import setup

NAME = 'optimization'
DESCRIPTION = ' NCG optimization library'
URL = 'https://arxiv.org/abs/1812.02886'
EMAIL = 'sadya@apple.com','vinaypalakkode@apple.com'
AUTHOR = 'Saurabh Adya', 'Vinay Palakkode'
VERSION = 0.1
REQUIRED=['tensorflow-gpu>=1.4, <=1.8',\
          'horovod==0.13.3']

def find_packages():
    return [
        'optimization',
        ]

setup(name='ncg',
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      license='Apple Confidential', # to be consulted with Apple Lawyers
      packages=find_packages(),
      install_requires=REQUIRED
)

