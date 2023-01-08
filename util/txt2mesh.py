import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.point_cloud import PointCloud
from point_e.util.mesh import TriMesh


class point_e_wrapper:

    def __init__(
        self,
        guidance_scale: float = 3.0,
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('creating base model...')
        base_name = 'base40M-textvec'
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        print('creating upsample model...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

        print('downloading base checkpoint...')
        base_model.load_state_dict(load_checkpoint(base_name, device))

        print('downloading upsampler checkpoint...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))

        self.sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[guidance_scale, 0.0],
            model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )

        print('creating SDF model...')
        name = 'sdf'
        self.sdf_model = model_from_config(MODEL_CONFIGS[name], device)
        self.sdf_model.eval()

        print('loading SDF model...')
        self.sdf_model.load_state_dict(load_checkpoint(name, device))


    def sample(
        self,
        prompt: str,
    ):
        # Produce a sample from the model.
        samples = None
        for x in tqdm(self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x

        return self.sampler.output_to_point_clouds(samples)[0]


    def get_mesh(
        self,
        pc: PointCloud,
    ):
        # Produce a mesh (with vertex colors)
        mesh = marching_cubes_mesh(
            pc=pc,
            model=self.sdf_model,
            batch_size=4096,
            grid_size=128, # increase to 128 for resolution used in evals
            progress=True,
        )

        return mesh



if __name__ == '__main__':
    point_e = point_e_wrapper(guidance_scale=3.0)
    # Set a prompt to condition on.
    prompt = 'a red motorcycle'

    pc = point_e.sample(prompt)
    mesh = point_e.get_mesh(pc)

