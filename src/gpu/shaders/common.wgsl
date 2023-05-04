struct Particle {
 prev_position: vec3f,
 position: vec3f,
 velocity: vec3f,
 ext_acc: vec3f,
 inv_mass: f32,
};

struct DistanceC {
 particles_idx: array<u32, 2>,
 rest_distance: f32,
 compliance: f32,
};

struct TetrahedralVolumeC {
 particles_idx: array<u32, 4>,
 rest_volume: f32,
 compliance: f32,
};

struct SimParams {
 delta: f32,
 jacobi_w: f32
};

const DELTAS_SIZE = 64u;

fn length2(x: vec3<f32>) -> f32 {
  let x2 = pow(x, vec3(2.0));
  return x2.x + x2.y + x2.z;
}
