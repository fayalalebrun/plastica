use encase::ShaderType;
use glam::Vec3;

pub mod cpu;
pub mod gpu;

#[repr(C)]
#[derive(Clone, Copy, ShaderType)]
pub struct Particle {
    prev_position: Vec3,
    pub position: Vec3,
    velocity: Vec3,
    pub ext_acc: Vec3,
    pub inv_mass: f32,
}

impl Particle {
    pub fn new(position: Vec3, inv_mass: f32) -> Self {
        Self {
            prev_position: position,
            position,
            velocity: Vec3::new(0., 0., 0.),
            inv_mass,
            ext_acc: Vec3::new(0., 0., 0.),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct DistanceC {
    particles_idx: [u32; 2],
    rest_distance: f32,
    compliance: f32,
}
impl DistanceC {
    pub fn new(particles_idx: [u32; 2], rest_distance: f32, compliance: f32) -> Self {
        Self {
            particles_idx,
            rest_distance,
            compliance,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct TetrahedralVolumeC {
    particles_idx: [u32; 4],
    rest_volume: f32,
    compliance: f32,
}

impl TetrahedralVolumeC {
    pub fn new(particles_idx: [u32; 4], rest_volume: f32, compliance: f32) -> Self {
        assert!(rest_volume >= 0.);
        Self {
            particles_idx,
            rest_volume,
            compliance,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, ShaderType)]
struct ParticleConstraintDeltas<const N: usize = 64> {
    pub n: u32,
    pub deltas: [Vec3; N],
}

#[repr(C)]
#[derive(Clone, Copy, ShaderType)]
pub struct ConstraintDelta {
    pub delta: Vec3,
    pub particle_idx: u32,
}
