from setuptools import find_packages, setup

setup(
   name='inpars',
   version='0.1.0',
   author='Hugo Abonizio',
   author_email='hugo.abonizio@gmail.com',
   packages=find_packages("src"),
   url='https://github.com/hugoabonizio/inpars',
   license='LICENSE.txt',
   description='InPars',
   long_description=open('README.md').read(),
   install_requires=[
      'torch',
      'pyserini>=0.19',
      'transformers>=4.0',
      'pyyaml>=6.0',
      'ir_datasets>=0.5',
      'ftfy>=6.0',
      'accelerate>=0.15',
   ],
)
