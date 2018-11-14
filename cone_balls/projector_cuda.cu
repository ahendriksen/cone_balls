#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "THC/THC.h"
#include "THC/THCDeviceTensor.cuh"
#include "THC/THCAtomics.cuh"
#include "THC/THCDeviceUtils.cuh"
#include "THC/THCReduceApplyUtils.cuh"
#include <THC/THCApply.cuh>
#include "device_tensor.h"


__device__ __forceinline__ int
reflect(int i, int dimi) {
    if (i < 0) {
	i = -i - 1;
    } else if (i >= dimi) {
	i = 2 * dimi - i - 1;
    }
    return i;
}

template <typename scalar_t>
__inline__ __device__
scalar_t warpReduceSum(int mask, scalar_t val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
      val += __shfl_down_sync(mask, val, offset);
  return val;
}

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

__device__ float3 operator+(const double &a, const float3 &b) {
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

__device__ float3 operator-(const double &a, const float3 &b) {
    return make_float3(a-b.x, a-b.y, a-b.z);
}

__device__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device__ float3 operator*(const int &a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float3 operator*(const float &a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float3 operator*(const double &a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
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
		auto intersection_len = 2 * sqrt(int_square);
		return intersection_len;
	    } else {
		return 0;
	    }
	} else {
	    return 0;
	}
}

template <typename scalar_t>
__global__ void
cuda_project_balls0(dTensor2R ray_origin,
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
	float3 det_u = load_vec(detector_u[angle]);
	float3 det_v = load_vec(detector_v[angle]);

	// Move detector origin to lowerleft corner from center
	det_o = det_o - 0.5 * (H * det_v + W * det_u);
	// Calculate pixel position
	auto pixel_o = det_o + h * det_v + w * det_u;
	// Calculate ray direction
	auto ray_dir = pixel_o - ray_o;

	float y = 0;
	for (int ball=0; ball < num_balls; ball++) {
	    float3 ball_o = load_vec(ball_origin[ball]);
	    float ball_r = ball_radius[ball];

	    y += intersect_ball(ray_o, ray_dir, ball_o, ball_r);
	}
	out_projections[angle][h][w] = y;
    }
}
///////////////////////////////////////////////////////////////////////////////
//                            Convolution:Forward                            //
///////////////////////////////////////////////////////////////////////////////


template <typename scalar_t>
__global__ void
conv2(dTensor4R input,
      dTensor4R kernel,
      dTensor1R bias,
      dTensor4R output,
      int dilation)
{
    // This is an unoptimized reference implementation. It could serve
    // as a starting point for further optimization.

    int B = output.getSize(0);
    int C_OUT = output.getSize(1);
    int C_IN = input.getSize(1);
    int H = input.getSize(2);
    int W = input.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;

    if (W <= w || H <= h) {
	return;
    }

    for (int b=0; b < B; b++) {
	for (int c_out=0; c_out < C_OUT; c_out++) {
	    scalar_t o = bias[c_out];
	    for (int c_in=0; c_in < C_IN; c_in++) {
		for (int p=-1; p <= 1; p++) {
		    for (int q=-1; q <= 1; q++) {
			int hp = reflect(h + dilation * p, (int) H);
			int wq = reflect(w + dilation * q, (int) W);
			o += kernel[c_out][c_in][p + 1][q + 1] // p and q can be negative
			    * input[b][c_in][hp][wq];
		    }
		}
	    }
	    output[b][c_out][h][w] = o;
	}
    }
}

template <typename scalar_t>
__global__ void
conv3(dTensor4R input,
      dTensor4R kernel,
      dTensor1R bias,
      dTensor4R output,
      int dilation)
{
    // This implementation caches the kernel weights.

    // LIMITS:
    //    49152 bytes of shared memory per block
    //    12288 floats of shared memory per block
    // +-  1300 kernels can be stored in shared mem
    // So we must have:
    //     C_IN * C_OUT < 1300
    extern __shared__ int shared_memory[];

    int B = output.getSize(0);
    int C_OUT = output.getSize(1);
    int C_IN = input.getSize(1);
    int H = input.getSize(2);
    int W = input.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    for (int i=pId; i < kernel.numElements(); i+=num_threads) {
	kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 4d tensor.
    dTensor4R kernel_buffer = THCDeviceTensor<scalar_t, 4, THC_INDEX, RestrictPtrTraits>
	(kernel_buf, kernel.sizes(), kernel.strides());

    __syncthreads();

    if (W <= w || H <= h) {
	return;
    }

    for (int b=0; b < B; b++) {
	for (int c_out=0; c_out < C_OUT; c_out++) {
	    scalar_t o = bias[c_out];
	    for (int c_in=0; c_in < C_IN; c_in++) {
		for (int p=-1; p <= 1; p++) {
		    for (int q=-1; q <= 1; q++) {
			int hp = reflect(h + dilation * p, (int) H);
			int wq = reflect(w + dilation * q, (int) W);
			o += kernel_buffer[c_out][c_in][p + 1][q + 1]
			    * input[b][c_in][hp][wq];
		    }
		}
	    }
	    output[b][c_out][h][w] = o;
	}
    }
}

template <typename scalar_t>
__global__ void
conv4(dTensor4R input,
      dTensor4R kernel,
      dTensor1R bias,
      dTensor4R output,
      int dilation)
{
    // Performance improvements:
    // 1) This implementation caches the kernel weights.
    // 2) This implementation precomputes data and kernel offsets.

    // LIMITS:
    //    49152 bytes of shared memory per block
    //    12288 floats of shared memory per block
    // +-  1300 kernels can be stored in shared mem
    // So we must have:
    //     C_IN * C_OUT < 1300
    extern __shared__ int shared_memory[];

    int B = output.getSize(0);
    int C_OUT = output.getSize(1);
    int C_IN = input.getSize(1);
    int H = input.getSize(2);
    int W = input.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    for (int i=pId; i < kernel.numElements(); i+=num_threads) {
	kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 4d tensor.
    dTensor4R kernel_buffer = THCDeviceTensor<scalar_t, 4, THC_INDEX, RestrictPtrTraits>
	(kernel_buf, kernel.sizes(), kernel.strides());

    __syncthreads();

    if (W <= w || H <= h) {
	return;
    }

    // Precompute data offsets:
    THC_INDEX data_offsets[9];
    scalar_t *data0 = &input[0][0][0][0];
    int i = 0;
    for (int p=-1; p <= 1; p++) {
	for (int q=-1; q <= 1; q++) {
	    int hp = reflect(h + dilation * p, (int) H);
	    int wq = reflect(w + dilation * q, (int) W);
	    data_offsets[i] = &input[0][0][hp][wq] - data0;
	    i++;
	}
    }
    // Actually compute the convolution
    for (int b=0; b < B; b++) {
	for (int c_out=0; c_out < C_OUT; c_out++) {
	    scalar_t o = bias[c_out];
	    for (int c_in=0; c_in < C_IN; c_in++) {
		data0 = &input[b][c_in][0][0];
		scalar_t *kernel0 = &kernel_buffer[c_out][c_in][0][0];
		for (int i= 0; i < 9; i++) {
		    o += *(data0 + data_offsets[i]) * (*kernel0);
		    // Incrementing the kernel pointer works because
		    // the kernel weights are contiguous and the
		    // data_offsets are prepared to be in the same
		    // order as the kernel weights.
		    kernel0++;
		}
	    }
	    output[b][c_out][h][w] = o;
	}
    }
}

template <typename scalar_t>
__global__ void
conv5(dTensor4R input,
      dTensor4R kernel,
      dTensor1R bias,
      dTensor4R output,
      int dilation)
{
    // Performance improvements:
    // 1) This implementation caches the kernel weights.
    // 2) This implementation precomputes data offsets in x and y
    //    direction instead of pointers.

    // LIMITS:
    //    49152 bytes of shared memory per block
    //    12288 floats of shared memory per block
    // +-  1300 kernels can be stored in shared mem
    // So we must have:
    //     C_IN * C_OUT < 1300
    extern __shared__ int shared_memory[];

    int B = output.getSize(0);
    int C_OUT = output.getSize(1);
    int C_IN = input.getSize(1);
    int H = input.getSize(2);
    int W = input.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    for (int i=pId; i < kernel.numElements(); i+=num_threads) {
	kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 4d tensor.
    dTensor4R kernel_buffer = THCDeviceTensor<scalar_t, 4, THC_INDEX, RestrictPtrTraits>
	(kernel_buf, kernel.sizes(), kernel.strides());

    __syncthreads();

    if (W <= w || H <= h) {
	return;
    }

    // Precompute data offsets:
    int hs[3] = {0};
    int ws[3] = {0};

    for (int i=-1; i <= 1; i++) {
    	hs[i + 1] = reflect(h + dilation * i, (int) H);
    	ws[i + 1] = reflect(w + dilation * i, (int) W);
    }

    // Actually compute the convolution
    for (int b=0; b < B; b++) {
	for (int c_out=0; c_out < C_OUT; c_out++) {
	    scalar_t o = bias[c_out];
	    for (int c_in=0; c_in < C_IN; c_in++) {
		scalar_t *kernel0 = &kernel_buffer[c_out][c_in][0][0];
		#pragma unroll
		for (int p=-1; p <= 1; p++) {
		    #pragma unroll
		    for (int q=-1; q <= 1; q++) {
			o += input[b][c_in][hs[p + 1]][ws[q + 1]] * (*kernel0);
			// Incrementing the kernel pointer works because
			// the kernel weights are contiguous and the
			// data_offsets are prepared to be in the same
			// order as the kernel weights.
			kernel0++;
		    }
		}
	    }
	    output[b][c_out][h][w] = o;
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
//                        Kernel preparation functions                       //
///////////////////////////////////////////////////////////////////////////////

at::Tensor cuda_project_balls(at::Tensor ray_origin,      // dim: num_angles * 3
			      at::Tensor detector_center, // dim: num_angles * 3
			      at::Tensor detector_u,	  // dim: num_angles * 3
			      at::Tensor detector_v,	  // dim: num_angles * 3
			      at::Tensor ball_origin,     // dim: num_balls  * 3
			      at::Tensor ball_radius,     // dim: num_balls
			      at::Tensor out_projections) // dim: num_angles * num_v_pixels * num_u_pixels
{

    int block_size = 16;
    int implementation = 0;

    AT_DISPATCH_FLOATING_TYPES(ray_origin.type(), "cuda_project_balls", ([&] {
        // Create device tensors:
        dTensor2R ray_origin_d = toDeviceTensorR<scalar_t, 2>(ray_origin);
        dTensor2R detector_center_d = toDeviceTensorR<scalar_t, 2>(detector_center);
        dTensor2R detector_u_d = toDeviceTensorR<scalar_t, 2>(detector_u);
        dTensor2R detector_v_d = toDeviceTensorR<scalar_t, 2>(detector_v);
        dTensor2R ball_origin_d = toDeviceTensorR<scalar_t, 2>(ball_origin);
        dTensor1R ball_radius_d = toDeviceTensorR<scalar_t, 1>(ball_radius);
        dTensor3R out_projections_d = toDeviceTensorR<scalar_t, 3>(out_projections);

        dim3 gridSize(THCCeilDiv((int) out_projections_d.getSize(2), block_size),
    		      THCCeilDiv((int) out_projections_d.getSize(1), block_size));
        dim3 blockSize(block_size, block_size);
	if (implementation == 0) {
	    cuda_project_balls0<scalar_t><<<gridSize, blockSize>>>
		(ray_origin_d,
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
