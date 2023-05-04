use glam::Vec3;
use rayon::prelude::*;

use crate::{ConstraintDelta, DistanceC, Particle, TetrahedralVolumeC};

pub trait Constraint {
    /// Returns delta_x for each particle associated to this constraint
    fn solve(&self, particles: &[Particle], delta: f32) -> Vec<ConstraintDelta> {
        let gradients = self.gradients(particles);
        let particles_idx = self.particles_idx();

        let inv_masses: Vec<_> = particles_idx
            .iter()
            .map(|i| particles[*i as usize].inv_mass)
            .collect();

        let xpbd_stiff = self.compliance() / delta / delta;

        let lambda = -self.value(particles)
            / (gradients
                .iter()
                .zip(inv_masses.iter())
                .map(|(g, w)| w * g.length_squared())
                .sum::<f32>()
                + xpbd_stiff);
        let deltas = gradients.iter().zip(inv_masses).map(|(g, im)| {
            debug_assert!(lambda.is_finite());
            debug_assert!(im.is_finite());
            debug_assert!(g.is_finite());
            debug_assert!(
                (lambda * im * (*g)).is_finite(),
                "l: {} im: {} g: {}",
                lambda,
                im,
                g
            );
            lambda * im * (*g)
        });

        deltas
            .zip(particles_idx.iter())
            .map(|(delta, particle_idx)| {
                debug_assert!(delta.is_finite());
                ConstraintDelta {
                    particle_idx: *particle_idx,
                    delta,
                }
            })
            .collect()
    }

    fn compliance(&self) -> f32;

    fn particles_idx(&self) -> Vec<u32>;

    fn value(&self, particles: &[Particle]) -> f32;

    fn gradients(&self, particles: &[Particle]) -> Vec<Vec3>;
}

impl Constraint for TetrahedralVolumeC {
    #[inline]
    fn compliance(&self) -> f32 {
        self.compliance
    }

    #[inline]
    fn particles_idx(&self) -> Vec<u32> {
        self.particles_idx.to_vec()
    }

    #[inline]
    fn value(&self, particles: &[Particle]) -> f32 {
        let (i1, i2, i3, i4) = (
            self.particles_idx[0],
            self.particles_idx[1],
            self.particles_idx[2],
            self.particles_idx[3],
        );

        let (p1, p2, p3, p4) = (
            particles[i1 as usize].position,
            particles[i2 as usize].position,
            particles[i3 as usize].position,
            particles[i4 as usize].position,
        );

        let v = (p2 - p1).cross(p3 - p1).dot(p4 - p1) / 6.;

        6. * (v - self.rest_volume)
    }

    #[inline]
    fn gradients(&self, particles: &[Particle]) -> Vec<Vec3> {
        let (i1, i2, i3, i4) = (
            self.particles_idx[0],
            self.particles_idx[1],
            self.particles_idx[2],
            self.particles_idx[3],
        );

        let (p1, p2, p3, p4) = (
            particles[i1 as usize].position,
            particles[i2 as usize].position,
            particles[i3 as usize].position,
            particles[i4 as usize].position,
        );

        vec![
            (p4 - p2).cross(p3 - p2),
            (p3 - p1).cross(p4 - p1),
            (p4 - p1).cross(p2 - p1),
            (p2 - p1).cross(p3 - p1),
        ]
    }
}

impl Constraint for DistanceC {
    #[inline]
    fn compliance(&self) -> f32 {
        self.compliance
    }

    #[inline]
    fn particles_idx(&self) -> Vec<u32> {
        vec![self.particles_idx[0], self.particles_idx[1]]
    }

    #[inline]
    fn value(&self, particles: &[Particle]) -> f32 {
        let (i1, i2) = (self.particles_idx[0], self.particles_idx[1]);
        let dist = (particles[i2 as usize].position - particles[i1 as usize].position).length();
        debug_assert!(dist.is_finite());
        debug_assert!(dist >= 0.);
        dist - self.rest_distance
    }

    #[inline]
    fn gradients(&self, particles: &[Particle]) -> Vec<Vec3> {
        let (i1, i2) = (self.particles_idx[0], self.particles_idx[1]);
        let (x1, x2) = (
            particles[i1 as usize].position,
            particles[i2 as usize].position,
        );
        let dir = (x1 - x2).normalize_or_zero();
        vec![dir, -dir]
    }
}

#[derive(Default)]
pub struct CpuSimulation {
    particles: Vec<Particle>,
    distance_constraints: Vec<DistanceC>,
    volume_constraints: Vec<TetrahedralVolumeC>,
    solver: SolverType,
}

#[derive(Clone, Copy)]
pub enum SolverType {
    GaussSeidel,
    Jacobi,
}

impl Default for SolverType {
    fn default() -> Self {
        Self::GaussSeidel
    }
}

impl CpuSimulation {
    pub fn new(solver: SolverType) -> Self {
        Self {
            solver,
            ..Default::default()
        }
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn add_particles(&mut self, particles: Vec<Particle>) {
        self.particles.extend(particles)
    }

    pub fn add_distance_constraints(&mut self, constraints: Vec<DistanceC>) {
        self.distance_constraints.extend(constraints)
    }

    pub fn add_volume_constraints(&mut self, constraints: Vec<TetrahedralVolumeC>) {
        self.volume_constraints.extend(constraints)
    }

    pub fn simulate(&mut self, substeps: u32, delta: f32, print_error: bool) {
        fn add_constraints_jacobi<T: Constraint + Sync>(
            particles: &mut [Particle],
            constraints: &[T],
            delta: f32,
        ) {
            let x_deltas: Vec<_> = constraints
                .par_iter()
                .flat_map(|c| c.solve(particles, delta))
                .collect();

            particles.par_iter_mut().enumerate().for_each(|(idx, p)| {
                let mut total_delta = Vec3::ZERO;
                let mut num_constraints = 0;

                x_deltas.iter().for_each(|s| {
                    if s.particle_idx == idx as u32 {
                        total_delta += s.delta;
                        num_constraints += 1;
                    }
                });

                const W: f32 = 1.5;
                if num_constraints > 0 {
                    p.position += W * total_delta / num_constraints as f32;
                }
            })
        }

        fn add_constraints_gauss_seidel<T: Constraint + Sync>(
            particles: &mut [Particle],
            constraints: &[T],
            delta: f32,
        ) {
            for c in constraints {
                for p_delta in c.solve(particles, delta) {
                    particles[p_delta.particle_idx as usize].position += p_delta.delta;
                }
            }
        }

        fn error<T: Constraint + Sync>(particles: &[Particle], constraints: &[T]) -> f32 {
            constraints.iter().map(|c| c.value(particles).abs()).sum()
        }

        let Self {
            particles,
            distance_constraints,
            volume_constraints,
            solver,
        } = self;

        let sub_delta = delta / substeps as f32;

        for _ in 0..substeps {
            particles.iter_mut().for_each(|p| {
                p.velocity += p.ext_acc * sub_delta;
                p.prev_position = p.position;
                p.position += p.velocity * sub_delta;

                // TODO: Remove
                if p.position.z < 0.0 {
                    p.position = p.prev_position;
                    p.position.z = 0.;
                }
                debug_assert!(
                    p.position.is_finite(),
                    "p: {}, v: {}",
                    p.position,
                    p.velocity
                )
            });

            if print_error {
                println!("Distance error: {}", error(particles, distance_constraints));
                println!("Tet error: {}", error(particles, volume_constraints));
            }

            match solver {
                SolverType::GaussSeidel => {
                    add_constraints_gauss_seidel(particles, distance_constraints, sub_delta);
                    add_constraints_gauss_seidel(particles, volume_constraints, sub_delta);
                }
                SolverType::Jacobi => {
                    add_constraints_jacobi(particles, distance_constraints, sub_delta);
                    add_constraints_jacobi(particles, volume_constraints, sub_delta);
                }
            }

            particles.iter_mut().for_each(|p| {
                debug_assert!(p.position.is_finite());
                p.velocity = (p.position - p.prev_position) / sub_delta;
                debug_assert!(p.velocity.is_finite());
            })
        }
    }
}
