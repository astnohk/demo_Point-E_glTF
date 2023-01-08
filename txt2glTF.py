from util import txt2mesh
from util import gltf_util



if __name__ == '__main__':
    point_e = txt2mesh.point_e_wrapper(guidance_scale=3.0)
    # Set a prompt to condition on.
    prompt = 'a red motorcycle'

    pc = point_e.sample(prompt)
    mesh = point_e.get_mesh(pc)

    # Write the mesh to a PLY file to import into some other program.
    gltf_util.write_gltf('mesh.glb', mesh)

