#include <torch/torch.h>
#include <vector>
#include "ATen/TensorUtils.h"
#include "ATen/ScalarType.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

///////////////////////////////////////////////////////////////////////////////
//                   Forward declaration of CUDA functions                   //
///////////////////////////////////////////////////////////////////////////////
at::Tensor cuda_project_balls(at::Tensor ray_origin,      // dim: num_angles * 3
			      at::Tensor detector_center, // dim: num_angles * 3
			      at::Tensor detector_u,	     // dim: num_angles * 3
			      at::Tensor detector_v,	     // dim: num_angles * 3
			      at::Tensor ball_origin,     // dim: num_balls * 3
			      at::Tensor ball_radius,     // dim: num_balls
			      at::Tensor out_projections); // dim: num_angles * num_v_pixels * num_u_pixels

///////////////////////////////////////////////////////////////////////////////
//                                  Macros                                   //
///////////////////////////////////////////////////////////////////////////////
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)


///////////////////////////////////////////////////////////////////////////////
//                                 Functions                                 //
///////////////////////////////////////////////////////////////////////////////

at::Tensor project_balls(at::Tensor ray_origin,      // dim: num_angles * 3
			 at::Tensor detector_center, // dim: num_angles * 3
			 at::Tensor detector_u,	     // dim: num_angles * 3
			 at::Tensor detector_v,	     // dim: num_angles * 3
			 at::Tensor ball_origin,     // dim: num_balls * 3
			 at::Tensor ball_radius,     // dim: num_balls
			 at::Tensor out_projections) // dim: num_angles * num_v_pixels * num_u_pixels
{
    CHECK_CUDA(ray_origin);
    CHECK_CUDA(detector_center);
    CHECK_CUDA(detector_u);
    CHECK_CUDA(detector_v);
    CHECK_CUDA(ball_origin);
    CHECK_CUDA(ball_radius);
    CHECK_CUDA(out_projections);

    // Check dimensions
    AT_ASSERTM(ray_origin.dim() == 2, "ray_origin must be two-dimensional");
    AT_ASSERTM(detector_center.dim() == 2, "detector_center must be two-dimensional");
    AT_ASSERTM(detector_u.dim() == 2, "detector_u must be two-dimensional");
    AT_ASSERTM(detector_v.dim() == 2, "detector_v must be two-dimensional");
    AT_ASSERTM(ball_origin.dim() == 2, "ball_origin must be two-dimensional");
    AT_ASSERTM(ball_radius.dim() == 1, "ball_radius must be one-dimensional");
    AT_ASSERTM(out_projections.dim() == 3, "out_projections must be three-dimensional");

    // Check sizes match
    AT_ASSERTM(ray_origin.size(0) == detector_center.size(0), "");
    AT_ASSERTM(ray_origin.size(0) == detector_u.size(0), "");
    AT_ASSERTM(ray_origin.size(0) == detector_v.size(0), "");
    AT_ASSERTM(ball_origin.size(0) == ball_radius.size(0), "");
    AT_ASSERTM(ray_origin.size(0) == out_projections.size(0), "");

    AT_ASSERTM(ray_origin.size(1) == 3, "");
    AT_ASSERTM(detector_center.size(1) == 3, "");
    AT_ASSERTM(detector_u.size(1) == 3, "");
    AT_ASSERTM(detector_v.size(1) == 3, "");
    AT_ASSERTM(ball_origin.size(1) == 3, "");

    return cuda_project_balls(ray_origin, detector_center, detector_u, detector_v, ball_origin, ball_radius, out_projections);
}

///////////////////////////////////////////////////////////////////////////////
//                             Module declaration                            //
///////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_balls", &project_balls, "Project balls forward",
	  "ray_origin"_a, "detector_center"_a, "detector_u"_a, "detector_v"_a,
	  "ball_origin"_a, "ball_radius"_a, "out_projections"_a);
}
