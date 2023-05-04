struct ParticleConstraintDeltas {
 n: u32,
 deltas: array<vec3f, DELTAS_SIZE>,
};

@binding(0) @group(0) var<uniform> params: SimParams;
@binding(1) @group(0) var<storage, read_write> particles: array<Particle>;
@binding(2) @group(0) var<storage, read> results: array<ParticleConstraintDeltas>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let index = GlobalInvocationID.x;

  if index >= arrayLength(&particles) {
      return;
  }

  let n = results[index].n;

  var total = vec3(0.0);
  for (var i = 0u; i < n && i < DELTAS_SIZE; i++) {
    total = total + results[index].deltas[i];
  }
  total = params.jacobi_w * total / f32(n);
  particles[index].position += total;
}
