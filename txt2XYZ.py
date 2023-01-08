from util import txt2mesh
from util import xyz_util


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', help='Prompt', type=str, default='a red motorcycle')
    parser.add_argument('--output', help='Output XYZ filepath (*.xyz)', type=str, default='pc.xyz')
    args = parser.parse_args()

    point_e = txt2mesh.point_e_wrapper(guidance_scale=3.0)
    # Set a prompt to condition on.

    pc = point_e.sample(args.prompt)

    # Write the mesh to a PLY file to import into some other program.
    with open(args.output, 'w') as f:
        xyz_util.write_xyz(f, pc)

