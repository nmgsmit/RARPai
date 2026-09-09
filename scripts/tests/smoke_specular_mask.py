"""specular_mask must remove highlights and KEEP instruments.

The first version masked anything bright+desaturated, which is also exactly what a white
da Vinci instrument shaft looks like -- it deleted the nearest objects in frame. This pins
the behaviour so that cannot come back silently.

    python scripts/tests/smoke_specular_mask.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from make_stereo_proxy_gt import specular_mask

img = np.full((400, 400, 3), 60, np.uint8)            # dark tissue-ish background
cv2.circle(img, (60, 60), 4, (255, 255, 255), -1)     # highlight speckle, ~50 px
cv2.circle(img, (60, 200), 9, (255, 255, 255), -1)    # bigger highlight, ~250 px
cv2.rectangle(img, (200, 100), (340, 300), (250, 250, 250), -1)   # instrument, 28000 px

m = specular_mask(img)
assert m[60, 60], "small highlight must be masked"
assert m[200, 60], "mid highlight must be masked"
assert not m[200, 270], "instrument body must NOT be masked"
assert not m.all() and m.any(), "mask should be partial"

# A saturated red highlight is tissue colour, not a specular reflection.
red = np.full((100, 100, 3), 60, np.uint8)
cv2.circle(red, (50, 50), 5, (0, 0, 255), -1)
assert not specular_mask(red).any(), "saturated colour is not a specular highlight"

# Threshold is a knob: with max_blob huge, the instrument gets masked again (old behaviour).
assert specular_mask(img, max_blob=10 ** 9)[200, 270], "max_blob should control the cut"

print("OK  highlights masked, instrument kept")
