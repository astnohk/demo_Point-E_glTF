from util import txt2mesh
from util import gltf_util



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', help='Prompt', type=str, default='a red motorcycle')
    parser.add_argument('--output', help='Output glTF filepath (*.glb or *.gltf)', type=str, default='mesh.glb')
    args = parser.parse_args()

    point_e = txt2mesh.point_e_wrapper(guidance_scale=3.0)
    # Set a prompt to condition on.

    pc = point_e.sample(args.prompt)
    mesh = point_e.get_mesh(pc)

    # Write the mesh to a PLY file to import into some other program.
    gltf_util.write_gltf(args.output, mesh)

