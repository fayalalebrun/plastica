struct ParticleConstraintDeltas {
 n: atomic<u32>,
 deltas: array<vec3f, DELTAS_SIZE>,
};

@binding(0) @group(0) var<uniform> params: SimParams;
@binding(1) @group(0) var<storage, read> particles: array<Particle>;
@binding(2) @group(0) var<storage, read> distance_constraints: array<DistanceC>;
@binding(3) @group(0) var<storage, read_write> results: array<ParticleConstraintDeltas>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let index = GlobalInvocationID.x;

  if index >= arrayLength(&distance_constraints) {
      return;
  }

  let c = distance_constraints[index];
  let ps_idx = array(c.particles_idx[0], c.particles_idx[1]);
  let ps = array(particles[ps_idx[0]], particles[ps_idx[1]]);

  let value = distance(ps[0].position, ps[1].position) - c.rest_distance;

  let dir = normalize(ps[0].position - ps[1].position);

  let grad_1 = dir;
  let grad_2 = -dir;

  let xpbd_stiff = c.compliance / params.delta / params.delta;

  let lambda = -value/((ps[0].inv_mass * length2(grad_1) + ps[1].inv_mass * length2(grad_2)) + xpbd_stiff);

  let x1_delta = lambda * ps[0].inv_mass * grad_1;
  let x2_delta = lambda * ps[1].inv_mass * grad_2;

  add_delta_to_list(x1_delta, ps_idx[0]);
  add_delta_to_list(x2_delta, ps_idx[1]);
}

fn add_delta_to_list(delta: vec3<f32>, idx: u32) {
  
  let n = &results[idx].n;
  let index = atomicAdd(n, 1u);
  
  if index >= DELTAS_SIZE {
      return;
    }
  results[idx].deltas[index] = delta;
}
