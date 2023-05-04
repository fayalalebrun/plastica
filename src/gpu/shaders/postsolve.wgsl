@binding(0) @group(0) var<uniform> params: SimParams;
@binding(1) @group(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let index = GlobalInvocationID.x;

  if index >= arrayLength(&particles) {
      return;
  }

  particles[index].velocity = (particles[index].position - particles[index].prev_position) / params.delta;
}
