import addons as ad
"""
import struct
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from itertools import product
try:
    from .cwrapped import tessellate
    c_lib = True
except ImportError:
    c_lib = False
"""

if __name__ == "__main__":

    desired_coords = '18676.05018252586,6845773.541229122,117163.78576648101,6915575.52858474'
    
    # Instantiate STL class
    print3d = ad.CreateSTL()

    # Get test image from geonorge WMS
    image_wms = print3d.img_from_wms(desired_coords)

    # Convert from img -> np -> stl
    print3d.numpy2stl(image_wms,'testfil.stl',solid=True)

    