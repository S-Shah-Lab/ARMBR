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


long_description = """
This is the Python implementation of Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements (for full algorithm, see citation below). 

ARMBR is a lightweight and easy-to-use method for blink artifact removal from EEG signals using multivariate backward regression. The algorithm detects the times at which eye blinks occur and then estimates their linear scalp projection by regressing a simplified, time-locked reference signal against the multichannel EEG. This projection is used to suppress blink-related components while preserving underlying brain signals. ARMBR requires minimal training data, does not depend on dedicated EOG channels, and operates robustly in both offline and real-time (online) settings, including BCI applications.

This release contains some improvements to the BCI2000 GUI.

The code is maintained at:	
https://github.com/S-Shah-Lab/ARMBR


The semi-synthetic data used in the ARMBR paper, along with example codes that allow testing ARMBR of the semi-synthetic data used in the paper, are available on OSF at: https://osf.io/th2g6/


If you use ARMBR in your work, please cite:

**Citation**:

Alkhoury L, Scanavini G, Louviot S, Radanovic A, Shah SA & Hill NJ (2025). *Artifact-Reference Multivariate Backward Regression (ARMBR): A Novel Method for EEG Blink Artifact Removal with Minimal Data Requirements.* *Journal of Neural Engineering*, 22(3). [DOI: 10.1088/1741-2552/ade566](https://doi.org/10.1088/1741-2552/ade566) [PubMed: 40527334](https://www.ncbi.nlm.nih.gov/pubmed/40527334)

**BibTeX**:

```bibtex
@article{alkhoury2025armbr,
    author  = {Alkhoury, Ludvik and Scanavini, Giacomo and Louviot, Samuel and Radanovic, Ana and Shah, Sudhin A and Hill, N Jeremy},
    title   = {Artifact-Reference Multivariate Backward Regression ({ARMBR}): A Novel Method for {EEG} Blink Artifact Removal with Minimal Data Requirements},
    journal = {Journal of Neural Engineering},
    volume  = {22},
    number  = {3},
    pages   = {036048},
    year    = {2025},
    date    = {2025-06-25},
    doi     = {10.1088/1741-2552/ade566},
    url     = {https://doi.org/10.1088/1741-2552/ade566},
}			

"""

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


def update_readme_version(readme_path, version):
	import re
	with open(readme_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	
	version_line_pattern = re.compile(r"^Version\s+\d+\.\d+\.\d+")
	updated_lines = []
	replaced = False
	for line in lines:
		if version_line_pattern.match(line) and not replaced:
			updated_lines.append(f"Version {version}\n")
			replaced = True
		else:
			updated_lines.append(line)

	if replaced:
		with open(readme_path, 'w', encoding='utf-8') as f:
			f.writelines(updated_lines)
		print(f"✅ README.md updated to Version {version}")
	else:
		print("⚠️ No Version line found in README.md")


current_version = get_version(package_dir, 'ARMBR', 'armbr.py')
update_readme_version("README.md", current_version)

setup_args = dict(name='ARMBR',
package_dir={ '' : package_dir },
      version=current_version, # @VERSION_INFO@
      description='Python implementation of the ARMBR blink removal method',
	  long_description=long_description,
	  long_description_content_type="text/markdown",
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

