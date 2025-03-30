"""
.. include:: ../README.md

# Examples

## 📦 Simple cubic growth

Below is an example of the growth of a small simple cubic lattice, printing a LAMMPS-style dump string

```py
.. include:: ../examples/simple_cubic.py
```

"""

__version__ = "0.0.3"
__authors__ = ["Jacob Jeffries"]
__author_emails__ = ["jwjeffr@clemson.edu"]
__url__ = "https://github.com/jwjeffr/cgkmc"

from . import containers as containers
from . import simulations as simulations
from . import utils as utils
