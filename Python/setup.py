import os
import sys
import inspect
import setuptools
from setuptools import setup


package_dir = '.' # the directory that would get added to the path, expressed relative to the location of this setup.py file



try: __file__
except:
	try: frame = inspect.currentframe(); __file__ = inspect.getfile( frame )
	finally: del frame  # https://docs.python.org/3/library/inspect.html#the-interpreter-stack
HERE = os.path.realpath( os.path.dirname( __file__ ) )


setup_args = dict(name='ARMBR',
package_dir={ '' : package_dir },
      version='1.0.0', # @VERSION_INFO@
      description='Pyhton implementation of the ARMBR blink removal method',
      url='https://github.com/S-Shah-Lab/NatLang_Paradigm.git',
      author='Ludvik Alkhoury',
      author_email='Ludvik.Alkhoury@gmail.com',
      packages=['ARMBR'],
      install_requires=['scipy', 'numpy', 'mne', 'BCI2kReader', 'matplotlib', 'tqdm'])
      
      
if __name__ == '__main__' and getattr( sys, 'argv', [] )[ 1: ]:
	setuptools.setup( **setup_args )
else:
	sys.stderr.write( """
The ARMBR setup.py file should not be run or imported directly.
Instead, it is used as follows::

    python -m pip install -e  "%s"

""" % HERE )

