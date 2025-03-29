from enum import Enum, auto
from hashlib import sha1

import numpy as np


class Units(Enum):
    """
    Available units. See https://docs.lammps.org/units.html for details
    """

    real = auto()
    metal = auto()
    si = auto()
    cgs = auto()
    electron = auto()
    micro = auto()
    nano = auto()

    def boltzmann_constant(self):

        if self == Units.real:
            return 1.987e-3

        if self == Units.metal:
            return 8.617e-5

        if self == Units.si:
            return 1.381e-23

        if self == Units.cgs:
            return 1.381e-16

        if self == Units.electron:
            return 3.167e-6

        if self == Units.micro:
            return 1.381e-6

        if self == Units.nano:
            return 1.381e-2

        raise ValueError("invalid unit system")


def temp_to_beta(temperature: float, units: Units):
    r"""
    helper class to convert temperature to thermodynamic $\beta$ for a given choice of units
    """

    return 1.0 / (units.boltzmann_constant() * temperature)


def array_to_hex(x: np.typing.NDArray[np.floating]) -> str:

    return sha1(x.tobytes()).hexdigest()
