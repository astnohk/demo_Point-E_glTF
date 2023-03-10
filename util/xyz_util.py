from typing import TextIO
import numpy as np
from point_e.util.mesh import TriMesh

def write_xyz(
    f: TextIO,
    pc: TriMesh,
):
    #np.stack([mesh.channels[x] for x in "RGB"], axis=1)
    length = pc.coords.shape[0]
    # X Y Z R G B
    for l in range(length):
        f.write('{x} {y} {z} {r} {g} {b}\n'.format(
            x=pc.coords[l, 0],
            y=pc.coords[l, 1],
            z=pc.coords[l, 2],
            r=pc.channels['R'][l],
            g=pc.channels['G'][l],
            b=pc.channels['B'][l]))

