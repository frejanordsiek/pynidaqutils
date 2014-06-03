import ez_setup
ez_setup.use_setuptools()

from setuptools import setup

with open('README.rst') as file:
    long_description = file.read()

setup(name='pynidaqutils',
      version='0.1.1',
      description='Utilities to work with NI DAQs through PyDAQmx.',
      long_description=long_description,
      author='Freja Nordsiek',
      author_email='fnordsie at gmail dt com',
      url='https://github.com/frejanordsiek/pynidaqutils',
      packages=['pynidaqutils'],
      requires=['numpy'],
      license='BSD',
      keywords='DAQ nidaq daqmx daqmxbase pydaqmx',
      classifiers=[
          "Programming Language :: Python :: 3",
          "Development Status :: 3 - Alpha",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering",
          "Topic :: Software Development :: Libraries :: Python Modules"
          ],
      )
