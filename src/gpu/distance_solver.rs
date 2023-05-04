use std::mem;

use encase::CalculateSizeFor;
use wgpu::{
    BindGroup, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePass,
    ComputePipeline, Device,
};

use crate::DistanceC;

use super::shaders::BufferDesc;

pub struct DistanceSolver {
    pipeline: ComputePipeline,
    bind_group: Option<BindGroup>,
    distance_constraints_res: Buffer,
}

impl DistanceSolver {
    pub fn new(device: &Device, particles_n: u64) -> Self {
        let pipeline = super::shaders::create_pipeline(
            device,
            "distance_solver",
            [
                BufferDesc { read_only: true },
                BufferDesc { read_only: true },
                BufferDesc { read_only: false },
            ]
            .into_iter(),
            super::shaders::COMMON_SRC.to_string() + super::shaders::SOLVE_DIST_SRC,
        );

        let distance_constraints_res = device.create_buffer(&BufferDescriptor {
            label: Some("Distance constraints results"),
            size: Vec::<crate::ParticleConstraintDeltas>::calculate_size_for(particles_n).into(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group: None,
            distance_constraints_res,
        }
    }

    pub fn update_bind_group(
        &mut self,
        device: &Device,
        sim_params: &Buffer,
        particles: &Buffer,
        distance_constraints: &Buffer,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: distance_constraints.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.distance_constraints_res.as_entire_binding(),
                },
            ],
        }))
    }
    pub fn prerun(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.distance_constraints_res, 0, None);
    }

    pub fn run<'a: 'b, 'b>(
        &'a self,
        compute_pass: &'b mut ComputePass<'a>,
        distance_constraints: &Buffer,
    ) {
        const WORKGROUP_SIZE: u64 = 64;
        let constraints_n = distance_constraints.size() / mem::size_of::<DistanceC>() as u64;
        let work_groups = ((constraints_n / WORKGROUP_SIZE) + 1) as u32;

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(work_groups, 1, 1);
    }

    pub fn results(&self) -> &Buffer {
        &self.distance_constraints_res
    }
}
