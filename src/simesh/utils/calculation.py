import numpy as np
from typing import Iterable

# table to be implemented
morton_table_32 = {}

def morton_number(ign):
    return spread_32bits(ign)

def spread_32bits(ign):
    x, y, z = ign
    def extract_10bits(x):
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x <<  8)) & 0x0300F00F
        x = (x | (x <<  4)) & 0x030C30C3
        x = (x | (x <<  2)) & 0x09249249
        return x
    x, y, z= extract_10bits(x), extract_10bits(y), extract_10bits(z)
    return x | (y << 1) | (z << 2)