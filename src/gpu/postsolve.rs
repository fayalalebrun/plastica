use std::mem;

use wgpu::{BindGroup, Buffer, ComputePass, ComputePipeline, Device};

use crate::Particle;

use super::shaders::BufferDesc;

pub struct Postsolve {
    pipeline: ComputePipeline,
    bind_group: Option<BindGroup>,
}

impl Postsolve {
    pub fn new(device: &Device) -> Self {
        let pipeline = super::shaders::create_pipeline(
            device,
            "postsolve",
            std::iter::once(BufferDesc { read_only: false }),
            super::shaders::COMMON_SRC.to_string() + super::shaders::POSTSOLVE_SRC,
        );

        Self {
            pipeline,
            bind_group: None,
        }
    }

    fn create_bind_group(
        &self,
        device: &Device,
        sim_params: &Buffer,
        particles: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
            ],
        })
    }

    pub fn update_bind_group(&mut self, device: &Device, sim_params: &Buffer, particles: &Buffer) {
        self.bind_group = Some(self.create_bind_group(device, sim_params, particles));
    }

    pub fn run<'a: 'b, 'b>(&'a self, compute_pass: &'b mut ComputePass<'a>, particles: &Buffer) {
        const WORKGROUP_SIZE: u64 = 64;
        let particles_n = particles.size() / mem::size_of::<Particle>() as u64;
        let particle_work_groups = ((particles_n / WORKGROUP_SIZE) + 1) as u32;

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(particle_work_groups, 1, 1);
    }
}
