import numpy as np
import pygltflib
from point_e.util.mesh import TriMesh

def write_gltf(
    path: str,
    mesh: TriMesh,
):
    triangles = np.asarray(mesh.faces, dtype=np.uint16)
    points = np.asarray(mesh.verts, dtype=np.float32)
    triangles_binary_blob = triangles.flatten().tobytes() # Flatten triangle index [N x 3]
    points_binary_blob = points.flatten().tobytes()
    color = None
    if mesh.has_vertex_colors():
        colors = np.stack([mesh.vertex_channels[x] for x in "RGB"], axis=1)
        colors = np.asarray(colors, dtype=np.float32)
    colors_binary_blob = colors.flatten().tobytes()

    gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[
                pygltflib.Mesh(
                    primitives=[
                        pygltflib.Primitive(
                            attributes=pygltflib.Attributes(POSITION=1, COLOR_0=2), indices=0
                        ),
                    ]
                )
            ],
            accessors=[
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_SHORT,
                    count=triangles.size,
                    type=pygltflib.SCALAR,
                    max=[int(triangles.max())],
                    min=[int(triangles.min())],
                ),
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,
                    count=len(points),
                    type=pygltflib.VEC3,
                    max=points.max(axis=0).tolist(),
                    min=points.min(axis=0).tolist(),
                ),
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.FLOAT,
                    count=len(colors),
                    type=pygltflib.VEC3,
                    max=[1.0, 1.0, 1.0],
                    min=[0.0, 0.0, 0.0],
                ),
            ],
            bufferViews=[
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(triangles_binary_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(triangles_binary_blob),
                    byteLength=len(points_binary_blob),
                    target=pygltflib.ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(points_binary_blob) + len(triangles_binary_blob),
                    byteLength=len(colors_binary_blob),
                    target=pygltflib.ARRAY_BUFFER,
                ),
            ],
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(triangles_binary_blob) + len(points_binary_blob) + len(colors_binary_blob)
                )
            ],
        )
    gltf.set_binary_blob(triangles_binary_blob + points_binary_blob + colors_binary_blob)
    gltf.save(path)

