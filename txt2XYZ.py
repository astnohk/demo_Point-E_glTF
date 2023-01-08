from util import txt2mesh
from util import xyz_util


if __name__ == '__main__':
    point_e = txt2mesh.point_e_wrapper(guidance_scale=3.0)
    # Set a prompt to condition on.
    prompt = 'a red motorcycle'

    pc = point_e.sample(prompt)

    # Write the mesh to a PLY file to import into some other program.
    with open('pc.xyz', 'w') as f:
        xyz_util.write_xyz(f, pc)

