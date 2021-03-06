#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "THC/THC.h"
#include "THC/THCDeviceTensor.cuh"
#include "THC/THCAtomics.cuh"
#include "THC/THCDeviceUtils.cuh"
#include "device_tensor.h"


///////////////////////////////////////////////////////////////////////////////
//                         Project forward                                   //
///////////////////////////////////////////////////////////////////////////////


__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 operator+(const int &a, const float3 &b) {
    return make_float3(a+b.x, a+b.y, a+b.z);
}

__device__ float3 operator+(const float &a, const float3 &b) {
    return make_float3(a+b.x, a+b.y, a+b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ float3 operator-(const int &a, const float3 &b) {
    return make_float3(a-b.x, a-b.y, a-b.z);
}

__device__ float3 operator-(const float &a, const float3 &b) {
    return make_float3(a-b.x, a-b.y, a-b.z);
}

__device__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device__ float3 operator*(const int3 &a, const float3 &b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device__ float3 operator*(const int &a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float3 operator*(const float &a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float3 operator/(const float3 &a, const float &b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}


template <typename SuperTensor>
__inline__ __device__
float3 load_vec(detail::THCDeviceSubTensor<SuperTensor, 1, RestrictPtrTraits> vec) {
    float3 res = make_float3((float) vec[0],
			     (float) vec[1],
			     (float) vec[2]);
    return res;
}

__device__ __inline__ float sum3(const float3 &a) {
    return a.x + a.y + a.z;
}

__inline__ __device__
float intersect_ball(float3 ray_origin,
		      float3 ray_dir,
		      float3 ball_origin,
		      float ball_radius)
{

	auto t = sum3(ray_dir * (ball_origin - ray_origin)) / sum3(ray_dir * ray_dir);
	if (isfinite(t)) {
	    auto min_p = t * ray_dir + ray_origin;
	    auto ortho_projection = min_p - ball_origin;
	    auto d = sum3(ortho_projection * ortho_projection);

	    // Use pythagoras (d is already squared) to find the length of
	    // the intersection of the ray with the ball.
	    auto int_square = ball_radius * ball_radius - d;
	    if (int_square > 0) {
		auto intersection_len = 2.0f * sqrt(int_square);
		return intersection_len;
	    } else {
		return 0;
	    }
	} else {
	    return 0;
	}
}

__inline__ __device__
float intersect_ball_point(float3 point,
			   float3 ball_origin,
			   float ball_radius)
{

    auto d = (point - ball_origin);
    auto distance = sum3(d * d);
    if (distance < ball_radius * ball_radius) {
	return 1.0;
    } else{
	return 0.0;
    }
}


template <typename scalar_t>
__global__ void
_cuda_ball_volume(dTensor1R lower_left_voxel_center,
		  dTensor1R voxel_size,
		  dTensor2R ball_origin,
		  dTensor1R ball_radius,
		  dTensor3R out_volume,
		  int super_sampling)
{
    int num_balls = ball_origin.getSize(0);

    int X_len = out_volume.getSize(2);
    int Y_len = out_volume.getSize(1);
    int Z_len = out_volume.getSize(0);

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (X_len <= x || Y_len <= y) {
	return;
    }

    float3 llc_o = make_float3((float) lower_left_voxel_center[0], (float) lower_left_voxel_center[1], (float) lower_left_voxel_center[2]);
    float3 voxel_sz = make_float3((float) voxel_size[0], (float) voxel_size[1], (float) voxel_size[2]);

    for (int z = 0; z < Z_len; z++) {
	float3 voxel_o = llc_o + make_int3(z, y, x) * voxel_sz;

	float out = 0.0;
	for (int ball=0; ball < num_balls; ball++) {
	    float3 ball_o = load_vec(ball_origin[ball]);
	    float ball_r = ball_radius[ball];

	    // out += intersect_ball_point(voxel_o, ball_o, ball_r);

	    auto step = voxel_sz / (float) super_sampling;

	    for (int i = 0; i < super_sampling; i += 1) {
	    	for (int j = 0; j < super_sampling; j += 1) {
		    for (int k = 0; k < super_sampling; k += 1) {
			auto sample_point = voxel_o
			    - 0.5f * (super_sampling - 1) * step
			    + make_float3(i, j, k) * step;
			out += intersect_ball_point(sample_point, ball_o, ball_r);
		    }
		}
	    }
	}
	float super_sampling_divisor = super_sampling * super_sampling * super_sampling;
	out_volume[z][y][x] = out / super_sampling_divisor;
    }
}

template <typename scalar_t>
__global__ void
_cuda_project_balls_cone(dTensor2R ray_origin,
			 dTensor2R detector_center,
			 dTensor2R detector_u,
			 dTensor2R detector_v,
			 dTensor2R ball_origin,
			 dTensor1R ball_radius,
			 dTensor3R out_projections)
{
    int num_angles = ray_origin.getSize(0);
    int num_balls = ball_origin.getSize(0);

    int H = out_projections.getSize(1);
    int W = out_projections.getSize(2);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;

    if (W <= w || H <= h) {
	return;
    }

    for (int angle=0; angle < num_angles; angle++) {
	float3 ray_o = load_vec(ray_origin[angle]);
	float3 det_o = load_vec(detector_center[angle]);
	const float3 det_u = load_vec(detector_u[angle]);
	const float3 det_v = load_vec(detector_v[angle]);

	const float3 det_u_half = 0.5f * det_u;
	const float3 det_v_half = 0.5f * det_v;

	// The detector origin is at the center of the detector. Move
	// detector origin to the center of the left-bottom corner
	// pixel of detector.
	det_o = det_o - ((H - 1) * det_v_half + (W - 1) * det_u_half);

	const float3 det_u_quart = 0.5f * det_u_half;
	const float3 det_v_quart = 0.5f * det_v_half;
	float y = 0;

	for (int i = -1; i < 2; i+=2) {
	    for (int j = -1; j < 2; j+=2) {
		auto pixel_o = det_o + (4 * h + i) * det_v_quart + (4 * w + j) * det_u_quart;
		// Calculate ray direction
		auto ray_dir = pixel_o - ray_o;

		for (int ball=0; ball < num_balls; ball++) {
		    float3 ball_o = load_vec(ball_origin[ball]);
		    float ball_r = ball_radius[ball];

		    y += intersect_ball(ray_o, ray_dir, ball_o, ball_r) / 4.0f;
		}
	    }
	}
	out_projections[angle][h][w] = y;
    }
}


template <typename scalar_t>
__global__ void
_cuda_project_balls_parallel(dTensor2R ray_dir,
			     dTensor2R detector_center,
			     dTensor2R detector_u,
			     dTensor2R detector_v,
			     dTensor2R ball_origin,
			     dTensor1R ball_radius,
			     dTensor3R out_projections)
{
    int num_angles = ray_dir.getSize(0);
    int num_balls = ball_origin.getSize(0);

    int H = out_projections.getSize(1);
    int W = out_projections.getSize(2);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;

    if (W <= w || H <= h) {
	return;
    }

    for (int angle=0; angle < num_angles; angle++) {
	float3 ray_dir_angle = load_vec(ray_dir[angle]);
	float3 det_o = load_vec(detector_center[angle]);
	const float3 det_u = load_vec(detector_u[angle]);
	const float3 det_v = load_vec(detector_v[angle]);

	const float3 det_u_half = 0.5f * det_u;
	const float3 det_v_half = 0.5f * det_v;

	// The detector origin is at the center of the detector. Move
	// detector origin to the center of the left-bottom corner
	// pixel of detector.
	det_o = det_o - ((H - 1) * det_v_half + (W - 1) * det_u_half);

	const float3 det_u_quart = 0.5f * det_u_half;
	const float3 det_v_quart = 0.5f * det_v_half;
	float y = 0;

	for (int i = -1; i < 2; i+=2) {
	    for (int j = -1; j < 2; j+=2) {
		auto pixel_o = det_o + (4 * h + i) * det_v_quart + (4 * w + j) * det_u_quart;

		for (int ball=0; ball < num_balls; ball++) {
		    float3 ball_o = load_vec(ball_origin[ball]);
		    float ball_r = ball_radius[ball];

		    y += intersect_ball(pixel_o, ray_dir_angle, ball_o, ball_r) / 4.0f;
		}
	    }
	}
	out_projections[angle][h][w] = y;
    }
}

///////////////////////////////////////////////////////////////////////////////
//                        Kernel preparation functions                       //
///////////////////////////////////////////////////////////////////////////////

at::Tensor cuda_project_balls(at::Tensor ray_,            // dim: num_angles * 3
			      at::Tensor detector_center, // dim: num_angles * 3
			      at::Tensor detector_u,      // dim: num_angles * 3
			      at::Tensor detector_v,      // dim: num_angles * 3
			      at::Tensor ball_origin,     // dim: num_balls  * 3
			      at::Tensor ball_radius,     // dim: num_balls
			      at::Tensor out_projections, // dim: num_angles * num_v_pixels * num_u_pixels
			      bool cone)
{

    int block_size = 16;

    AT_DISPATCH_FLOATING_TYPES(ray_.scalar_type(), "cuda_project_balls", ([&] {
        // Create device tensors:
        dTensor2R ray_d = toDeviceTensorR<scalar_t, 2>(ray_);
        dTensor2R detector_center_d = toDeviceTensorR<scalar_t, 2>(detector_center);
        dTensor2R detector_u_d = toDeviceTensorR<scalar_t, 2>(detector_u);
        dTensor2R detector_v_d = toDeviceTensorR<scalar_t, 2>(detector_v);
        dTensor2R ball_origin_d = toDeviceTensorR<scalar_t, 2>(ball_origin);
        dTensor1R ball_radius_d = toDeviceTensorR<scalar_t, 1>(ball_radius);
        dTensor3R out_projections_d = toDeviceTensorR<scalar_t, 3>(out_projections);

        dim3 gridSize(THCCeilDiv((int) out_projections_d.getSize(2), block_size),
                      THCCeilDiv((int) out_projections_d.getSize(1), block_size));
        dim3 blockSize(block_size, block_size);
	if (cone) {
	    _cuda_project_balls_cone<scalar_t><<<gridSize, blockSize>>>
		(ray_d,
		 detector_center_d,
		 detector_u_d,
		 detector_v_d,
		 ball_origin_d,
		 ball_radius_d,
		 out_projections_d);
	} else {
	    _cuda_project_balls_parallel<scalar_t><<<gridSize, blockSize>>>
		(ray_d,
		 detector_center_d,
		 detector_u_d,
		 detector_v_d,
		 ball_origin_d,
		 ball_radius_d,
		 out_projections_d);
	}

        THCudaCheck(cudaGetLastError());
    }));
  return out_projections;
}

at::Tensor cuda_volume(at::Tensor lower_left_voxel_center,      // dim: 3
                       at::Tensor voxel_size,                   // dim:  3
                       at::Tensor ball_origin,                  // dim: num_balls  * 3
                       at::Tensor ball_radius,                  // dim: num_balls
                       at::Tensor out_volume,                   // dim: shape_z * shape_y * shape_x
		       int super_sampling)
{

    int block_size = 16;

    AT_DISPATCH_FLOATING_TYPES(out_volume.scalar_type(), "cuda_volume", ([&] {
        // Create device tensors:
        dTensor1R lower_left_voxel_center_d = toDeviceTensorR<scalar_t, 1>(lower_left_voxel_center);
	dTensor1R voxel_size_d = toDeviceTensorR<scalar_t, 1>(voxel_size);
        dTensor2R ball_origin_d = toDeviceTensorR<scalar_t, 2>(ball_origin);
        dTensor1R ball_radius_d = toDeviceTensorR<scalar_t, 1>(ball_radius);
        dTensor3R out_volume_d = toDeviceTensorR<scalar_t, 3>(out_volume);

        dim3 gridSize(THCCeilDiv((int) out_volume_d.getSize(2), block_size),
                      THCCeilDiv((int) out_volume_d.getSize(1), block_size));
        dim3 blockSize(block_size, block_size);
        _cuda_ball_volume<scalar_t><<<gridSize, blockSize>>>
	    (lower_left_voxel_center_d,
	     voxel_size_d,
	     ball_origin_d,
	     ball_radius_d,
	     out_volume_d,
	     super_sampling);


        THCudaCheck(cudaGetLastError());
    }));
  return out_volume;
}
