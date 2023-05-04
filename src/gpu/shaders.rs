use wgpu::{ComputePipeline, Device};

pub const COMMON_SRC: &str = include_str!("shaders/common.wgsl");
pub const PRESOLVE_SRC: &str = include_str!("shaders/presolve.wgsl");
pub const SOLVE_DIST_SRC: &str = include_str!("shaders/solve_dist.wgsl");
pub const SOLVE_TET_SRC: &str = include_str!("shaders/solve_tet_vol.wgsl");
pub const ADD_DELTAS_SRC: &str = include_str!("shaders/add_deltas.wgsl");
pub const POSTSOLVE_SRC: &str = include_str!("shaders/postsolve.wgsl");

pub struct BufferDesc {
    pub read_only: bool,
}

pub fn create_pipeline(
    device: &Device,
    label: &str,
    storage_buffers_read_only: impl Iterator<Item = BufferDesc>,
    shader_src: String,
) -> ComputePipeline {
    let entries: Vec<_> =
        std::iter::once(wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .chain(storage_buffers_read_only.enumerate().map(|(idx, desc)| {
            wgpu::BindGroupLayoutEntry {
                binding: (idx + 1) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: desc.read_only,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        }))
        .collect();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &entries,
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&(label.to_owned() + " layout")),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_src)),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&(label.to_owned() + " pipeline")),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    })
}
