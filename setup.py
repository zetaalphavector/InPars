from setuptools import find_packages, setup

setup(
   name='inpars',
   version='0.2.1',
   author='Hugo Abonizio',
   author_email='hugo.abonizio@gmail.com',
   packages=find_packages(),
   url='https://github.com/zetaalphavector/InPars',
   license='LICENSE.txt',
   description='InPars',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
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
