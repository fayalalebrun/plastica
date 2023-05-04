use std::mem;

use encase::CalculateSizeFor;
use wgpu::{
    BindGroup, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePass,
    ComputePipeline, Device,
};

use crate::TetrahedralVolumeC;

use super::shaders::BufferDesc;

pub struct TetSolver {
    pipeline: ComputePipeline,
    bind_group: Option<BindGroup>,
    results: Buffer,
}

impl TetSolver {
    pub fn new(device: &Device, particles_n: u64) -> Self {
        let pipeline = super::shaders::create_pipeline(
            device,
            "tet_solver",
            [
                BufferDesc { read_only: true },
                BufferDesc { read_only: true },
                BufferDesc { read_only: false },
            ]
            .into_iter(),
            super::shaders::COMMON_SRC.to_string() + super::shaders::SOLVE_TET_SRC,
        );

        let results = device.create_buffer(&BufferDescriptor {
            label: Some("Tet constraints results"),
            size: Vec::<crate::ParticleConstraintDeltas>::calculate_size_for(particles_n).into(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group: None,
            results,
        }
    }

    pub fn update_bind_group(
        &mut self,
        device: &Device,
        sim_params: &Buffer,
        particles: &Buffer,
        tet_constraints: &Buffer,
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
                    resource: tet_constraints.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.results.as_entire_binding(),
                },
            ],
        }))
    }
    pub fn prerun(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.results, 0, None);
    }

    pub fn run<'a: 'b, 'b>(
        &'a self,
        compute_pass: &'b mut ComputePass<'a>,
        tet_constraints: &Buffer,
    ) {
        const WORKGROUP_SIZE: u64 = 64;
        let constraints_n = tet_constraints.size() / mem::size_of::<TetrahedralVolumeC>() as u64;
        let work_groups = ((constraints_n / WORKGROUP_SIZE) + 1) as u32;

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(work_groups, 1, 1);
    }

    pub fn results(&self) -> &Buffer {
        &self.results
    }
}
