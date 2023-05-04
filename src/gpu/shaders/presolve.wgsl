@binding(0) @group(0) var<uniform> params: SimParams;
@binding(1) @group(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let index = GlobalInvocationID.x;

  if index >= arrayLength(&particles) {
      return;
  }

  var p = particles[index];

  p.velocity += p.ext_acc * params.delta;
  p.prev_position = p.position;
  p.position += p.velocity * params.delta;

  // Ground plane clipping
  if p.position.z < 0.0 {
      p.position = p.prev_position;
      p.position.z = 0.;
  }

  particles[index] = p;
}
