use std::{
    mem,
    sync::{Arc, Mutex},
};

use bytemuck::{Pod, Zeroable};
use encase::{private::WriteInto, ShaderSize, ShaderType, StorageBuffer};
use wgpu::{
    util::{DeviceExt, DownloadBuffer},
    Buffer, BufferUsages, CommandEncoder, Device, Queue,
};

use crate::{
    gpu::{distance_solver::DistanceSolver, tet_solver::TetSolver},
    DistanceC, Particle, TetrahedralVolumeC,
};

use self::{add_deltas::AddDeltas, postsolve::Postsolve, presolve::Presolve};

mod add_deltas;
mod distance_solver;
mod postsolve;
mod presolve;
mod shaders;
mod tet_solver;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimParams {
    delta: f32,
    jacobi_w: f32,
}

pub struct GpuSimulation {
    presolve: Presolve,
    distance_solver: DistanceSolver,
    tet_solver: TetSolver,
    add_deltas_dist: AddDeltas,
    add_deltas_tet: AddDeltas,
    postsolve: Postsolve,
    particles: Buffer,
    distance_constraints: Buffer,
    tet_constraints: Buffer,
    sim_params: Buffer,
    downloaded_particles: Arc<Mutex<Vec<Particle>>>,
}

impl GpuSimulation {
    pub fn new(
        device: &Device,
        particles: &[Particle],
        distance_constraints: &[DistanceC],
        tet_constraints: &[TetrahedralVolumeC],
    ) -> Self {
        pub fn create_buffer<T: ShaderType + WriteInto + ShaderSize>(
            device: &Device,
            els: &[T],
            usage: BufferUsages,
            label: &str,
        ) -> Buffer {
            let mut buffer = StorageBuffer::new(Vec::new());
            buffer.write(&els).unwrap();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &buffer.into_inner(),
                usage,
            })
        }

        let distance_solver = DistanceSolver::new(device, particles.len() as u64);

        let tet_solver = TetSolver::new(device, particles.len() as u64);

        let particles = create_buffer(
            device,
            particles,
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
            "Particles",
        );

        let distance_constraints = create_buffer(
            device,
            distance_constraints,
            BufferUsages::COPY_DST | BufferUsages::STORAGE,
            "Distance constraints",
        );

        let tet_constraints = create_buffer(
            device,
            tet_constraints,
            BufferUsages::COPY_DST | BufferUsages::STORAGE,
            "Tetrahedral volume constraints",
        );

        let sim_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim params"),
            contents: &[0u8; mem::size_of::<SimParams>()],
            usage: BufferUsages::UNIFORM,
        });

        let presolve = Presolve::new(device);

        let add_deltas_dist = AddDeltas::new(device);
        let add_deltas_tet = AddDeltas::new(device);

        let postsolve = Postsolve::new(device);

        Self {
            presolve,
            distance_solver,
            tet_solver,
            add_deltas_dist,
            add_deltas_tet,
            postsolve,
            particles,
            distance_constraints,
            tet_constraints,
            sim_params,
            downloaded_particles: Default::default(),
        }
    }

    pub fn simulate(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        substeps: u32,
        delta: f32,
    ) {
        let sub_delta = delta / substeps as f32;

        let params = SimParams {
            delta: sub_delta,
            jacobi_w: 1.5,
        };

        self.sim_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM,
        });

        self.presolve
            .update_bind_group(device, &self.sim_params, &self.particles);
        self.postsolve
            .update_bind_group(device, &self.sim_params, &self.particles);
        self.distance_solver.update_bind_group(
            device,
            &self.sim_params,
            &self.particles,
            &self.distance_constraints,
        );
        self.tet_solver.update_bind_group(
            device,
            &self.sim_params,
            &self.particles,
            &self.tet_constraints,
        );
        self.add_deltas_dist.update_bind_group(
            device,
            &self.sim_params,
            &self.particles,
            self.distance_solver.results(),
        );
        self.add_deltas_tet.update_bind_group(
            device,
            &self.sim_params,
            &self.particles,
            self.tet_solver.results(),
        );

        for i in 0..substeps {
            self.distance_solver.prerun(encoder);
            self.tet_solver.prerun(encoder);
            let cpass_name = format!("substep {i}");
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&cpass_name),
            });
            self.presolve.run(&mut cpass, &self.particles);
            self.distance_solver
                .run(&mut cpass, &self.distance_constraints);
            self.tet_solver.run(&mut cpass, &self.tet_constraints);
            self.add_deltas_dist.run(&mut cpass, &self.particles);
            self.add_deltas_tet.run(&mut cpass, &self.particles);
            self.postsolve.run(&mut cpass, &self.particles);
        }
    }

    pub fn download_particles(&self, device: &Device, queue: &Queue) -> Vec<Particle> {
        let downloaded_particles = self.downloaded_particles.clone();
        DownloadBuffer::read_buffer(device, queue, &self.particles.slice(..), move |buff| {
            let buff = buff.unwrap();
            let buffer = StorageBuffer::new(&buff[..]);
            let mut particles = downloaded_particles.lock().unwrap();
            buffer.read(&mut *particles).unwrap();
        });

        self.downloaded_particles.lock().unwrap().clone()
    }
}
