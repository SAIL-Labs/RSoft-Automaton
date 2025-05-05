# RSoft-Automaton
Optimisation code that uses RSoft's Python API along with scikit-optimisation. It can take numerous fibre cores and allocates them within a hexagonal, Circular, Pentagonal or Square grid, while also running many instances of beamprop through the terminal via multiprocessing and determining the best fitting parameter for a given prior, which alleviates the need to run RSoft through the GUI. 

Currently works for SMF28 and am now working towards initial testing for PL. In its current state it can be used but expect some bugs!

## Setup Instructions:

### 1) Install miniconda
- Go to the Miniconda website (https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation) and complete the 1st and 3rd steps (the 2nd just checks the integrity of the Miniconda installer).

- Run the installer, be sure to have  the "Install for: Just Me" option enabled.
- Ensure that Miniconda installs under your own user directory.
- The next step is a matter of preference. I've just selected the recommended options:
  - Create shortcuts
  - Register Miniconda3 as my default Python 3.12.....
  - Clear the package cache upon completion
- After installing you should be able to search for "Anaconda Prompt" in the search bar. This application will load in your base Anaconda environment.

### 2) Clone the repository
- Create a folder on the desktop. You can name it whatever you like.
- In Anaconda Prompt, ```cd``` into that folder and then clone this repository with:
```Anaconda Prompt
git clone https://github.com/SAIL-Labs/RSoft-Automaton.git
cd Rsoft-Automaton
```
### 3) Create/activate the environment using Conda
- You need to clone the environment I have been using to interface with RSoft. While in the same directory, do this with:
```Anaconda Prompt
conda env create -f myrsoft_env.yml
conda activate myrsoft
```
- Add this line while the environment is active to interface with the RSoft interpreter
```Anaconda Prompt
python -c "import site; open(site.getsitepackages()[0] + r'\rsoft_api.pth', 'w').write(r'C:\Synopsys\PhotonicSolutions\2024.09-SP1\RSoft\products\most\python')"
```
- Open the repository folder with VSCode using ```code .```
- Test if the environment works correctly by opening Data_manip.ipynb and running a cell with the following:
```
import numpy as np
import subprocess, json

from Circuit_Properties import *
from Functions import *
from HexProperties import *
from rstools import RSoftUserFunction, RSoftCircuit
```
- If not errors present themselves, good. If not, let me know. It'll either be a package version issue or a step was missed (or poorly explained) above.

### 4) API Shenanigans
- You _really_ only need to have the following to run Rsoft. **NOTE:** I have not yet tested if ```custom_priors``` needs to be filled or not, so just use the following definition as a placeholder for now **and ensure RunRsoft(custom_priors, False)** (else you will get an error, I think).
```
name="MCF_Test"
sim = RSoftSim(name)

# setup custom priors
custom_priors = {
            # "Corediam": (0.0001, 20),
            "Length": (10.0, 5000.0),
            "acore_taper_ratio": (1.0, 30)
            }
sim.RunRSoft(custom_priors, simulate = False)
```
- If set ``` simulate = True ``` beamprop will run.
- By default the properties for SMF28 populate the standard symbols table. To change any property you need to type the following _before calling_ ```sim.RunRSoft()```:
```
sim.core_num = 7
sim.acore_taper_ratio = 10
sim.launch_type = LaunchType.SM
sim.monitor_type = Monitor_Prop.TOTAL_POWER
sim.comp = Monitor_comp.BOTH
sim.Core_delta = 0.0034
sim.length = 5000
sim.grid_type = "Hex"
sim.Dx = 1
sim.Dz = 1
```
- These are a selection fo what you can change. To see everything else, consult ```RSoftSimultation.py``` (for fibre/simulation properties), ```Circuit_Properties.py``` (for monitor and circuit things) and importantly ```HexProperties.py``` (for grid types).
- After running ```RunRSoft()``` a .ind file will generate with you desired properties. It is important you inspect this to ensure what you desire is in there and that there are no discrepancies present. 
