import os
import sys
import ast
import inspect

import setuptools
from setuptools import setup


package_dir = 'Python' # the directory that would get added to the path, expressed relative to the location of this setup.py file



try: __file__
except:
	try: frame = inspect.currentframe(); __file__ = inspect.getfile( frame )
	finally: del frame  # https://docs.python.org/3/library/inspect.html#the-interpreter-stack
HERE = os.path.realpath( os.path.dirname( __file__ ) )


def get_version( *pargs ):
	version_file = os.path.join(HERE, *pargs)
	with open(version_file, 'r') as f:
		for line in f:
			if line.strip().startswith('__version__'):
				# Extract the version from the line, e.g., 'versions = "0.0.11"'
				version = ast.literal_eval( line.split('=')[-1].strip() )
				print('Version ' + version)
				return version
				
	raise ValueError("Version not found in " + version_file)


setup_args = dict(name='ARMBR',
package_dir={ '' : package_dir },
      version=get_version(package_dir, 'ARMBR', 'armbr.py'), # @VERSION_INFO@
      description='Python implementation of the ARMBR blink removal method',
      url='https://github.com/S-Shah-Lab/ARMBR.git',
      author='Ludvik Alkhoury',
      author_email='Ludvik.Alkhoury@gmail.com',
      packages=['ARMBR'],
      install_requires=['scipy', 'numpy'])
      
      
if __name__ == '__main__' and getattr( sys, 'argv', [] )[ 1: ]:
	setuptools.setup( **setup_args )
else:
	sys.stderr.write( """
The ARMBR setup.py file should not be run or imported directly.
Instead, it is used as follows::

    python -m pip install -e  "%s"

""" % HERE )

