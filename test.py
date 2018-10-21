# -*- coding: utf-8 -*-
from re import split
import matplotlib.pyplot as plt
import numpy as np
import math

a = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a = np.delete(a, 2, axis=0)
print(a)
