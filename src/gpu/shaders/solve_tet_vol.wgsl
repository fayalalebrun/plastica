struct ParticleConstraintDeltas {
 n: atomic<u32>,
 deltas: array<vec3f, DELTAS_SIZE>,
};

@binding(0) @group(0) var<uniform> params: SimParams;
@binding(1) @group(0) var<storage, read> particles: array<Particle>;
@binding(2) @group(0) var<storage, read> constraints: array<TetrahedralVolumeC>;
@binding(3) @group(0) var<storage, read_write> results: array<ParticleConstraintDeltas>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let c_idx = GlobalInvocationID.x;

  if c_idx >= arrayLength(&constraints) {
      return;
  }

  let vol = dot(cross(pos(c_idx, 1u) - pos(c_idx, 0u), pos(c_idx, 2u) - pos(c_idx, 0u)), pos(c_idx, 3u) - pos(c_idx, 0u)) / 6.0;

  let value = 6.0 * (vol - constraints[c_idx].rest_volume);

  var grad: array<vec3<f32>, 4>;

  var grad_sum = 0.0;

  var orderN: array<vec4u, 4> = array<vec4u, 4>(vec4u(3u, 1u, 2u, 1u),
			   vec4u(2u, 0u, 3u, 0u),
			   vec4u(3u, 0u, 1u, 0u),
			   vec4u(1u, 0u, 2u, 0u));
  
  for (var i = 0u; i < 4u; i++) {
    // var order = vec4u(0u);
    // switch i {
    // 	case 0u {
    // 	  order = vec4u(3u, 1u, 2u, 1u);
    // 	}
    // 	case 1u {
    // 	  order = vec4u(2u, 0u, 3u, 0u);
    // 	}
    // 	case 2u {
    // 	  order = vec4u(3u, 0u, 1u, 0u);
    // 	}
    // 	case 3u, default {
    // 	  order = vec4u(1u, 0u, 2u, 0u);
    // 	}
    //   }
    let order = orderN[i];
    
    grad[i] = cross(pos(c_idx, order.x) - pos(c_idx, order.y), pos(c_idx, order.z) - pos(c_idx, order.w));
    grad_sum += length2(grad[i])*inv_mass(c_idx, i);
  }

  let xpbd_stiff = constraints[c_idx].compliance / params.delta / params.delta;
  
  let lambda = -value / (grad_sum + xpbd_stiff);

  for (var i = 0u; i < 4u; i++) {
    let delta = lambda * inv_mass(c_idx, i) * grad[i];
    add_delta_to_list(delta, constraints[c_idx].particles_idx[i]);
  }
}

fn inv_mass(c_idx: u32, num: u32) -> f32 {
  return particles[constraints[c_idx].particles_idx[num]].inv_mass;
}

fn pos(c_idx: u32, num: u32) -> vec3<f32> {
  return particles[constraints[c_idx].particles_idx[num]].position;
}

fn add_delta_to_list(delta: vec3<f32>, idx: u32) {  
  let n = &results[idx].n;
  let index = atomicAdd(n, 1u);
  
  if index >= DELTAS_SIZE {
      return;
    }
  results[idx].deltas[index] = delta;
}

