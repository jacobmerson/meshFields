#include "MeshField_Bell.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

// Tests for the Bell element.
//
// 1. reference duality: the 21 functionals of the extended Bell reference
//    element (vertex values, gradients, and Hessians, and the edge moments of
//    the normal derivative against the degree 4 Legendre polynomial) applied
//    to the 21 reference basis functions give the identity.  Hessians are
//    computed by Richardson-extrapolated central differences of the
//    gradients, which is exact for these polynomials up to rounding.

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::Real;
using MeshField::Vector2;

// 5 point Gauss-Legendre rule on [-1, 1]; exact to degree 9
constexpr int numGaussPts = 5;
constexpr Real gaussPts[numGaussPts] = {-0.9061798459386640,
                                        -0.5384693101056831, 0.0,
                                        0.5384693101056831, 0.9061798459386640};
constexpr Real gaussWts[numGaussPts] = {0.2369268850561891, 0.4786286704993665,
                                        0.5688888888888889, 0.4786286704993665,
                                        0.2369268850561891};

KOKKOS_INLINE_FUNCTION Real legendre4(Real u) {
  return (35 * u * u * u * u - 30 * u * u + 3) / 8;
}

// reference triangle vertices and edges (a, b), matching BellReferenceShape
constexpr Real refVerts[3][2] = {{0, 0}, {1, 0}, {0, 1}};
constexpr int refEdges[3][2] = {{1, 2}, {0, 2}, {0, 1}};

// the functionals applied to basis function k: 18 vertex functionals, then 3
// edge moments
KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 21> applyFunctionals(int k) {
  const MeshField::BellReferenceShape shape;
  const Real h = 1e-3; // difference step for the Hessian
  Kokkos::Array<Real, 21> f;
  for (int v = 0; v < 3; ++v) {
    const Vector2 x{refVerts[v][0], refVerts[v][1]};
    const auto grad = shape.getLocalGradients(x);
    f[v * 6] = shape.getValues(x)[k];
    f[v * 6 + 1] = grad[k * 2];
    f[v * 6 + 2] = grad[k * 2 + 1];
    // central differences of the gradient with steps h and 2h, combined to
    // cancel the h^2 error term; exact for the quartic gradients here
    const auto d = [&](int dir, int comp, Real step) {
      Vector2 p = x, m = x;
      p[dir] += step;
      m[dir] -= step;
      return (shape.getLocalGradients(p)[k * 2 + comp] -
              shape.getLocalGradients(m)[k * 2 + comp]) /
             (2 * step);
    };
    const auto hess = [&](int dir, int comp) {
      return (4 * d(dir, comp, h) - d(dir, comp, 2 * h)) / 3;
    };
    f[v * 6 + 3] = hess(0, 0);
    f[v * 6 + 4] = hess(1, 0);
    f[v * 6 + 5] = hess(1, 1);
  }
  for (int e = 0; e < 3; ++e) {
    const auto *va = refVerts[refEdges[e][0]];
    const auto *vb = refVerts[refEdges[e][1]];
    const Real t[2] = {vb[0] - va[0], vb[1] - va[1]};
    const Real len = Kokkos::sqrt(t[0] * t[0] + t[1] * t[1]);
    const Real n[2] = {t[1] / len, -t[0] / len};
    Real moment = 0;
    for (int q = 0; q < numGaussPts; ++q) {
      const Real u = gaussPts[q];
      const Real s = (u + 1) / 2;
      const auto grad =
          shape.getLocalGradients(Vector2{va[0] + s * t[0], va[1] + s * t[1]});
      const Real dn = n[0] * grad[k * 2] + n[1] * grad[k * 2 + 1];
      moment += gaussWts[q] / 2 * legendre4(u) * dn;
    }
    f[18 + e] = moment;
  }
  return f;
}

// 1. the reference functionals applied to the reference basis are the identity
bool testReferenceDuality() {
  int errors = 0;
  Kokkos::parallel_reduce(
      "checkReferenceDuality", Kokkos::RangePolicy<ExecutionSpace>(0, 21),
      KOKKOS_LAMBDA(const int k, int &lerr) {
        const auto f = applyFunctionals(k);
        for (int i = 0; i < 21; ++i) {
          const bool hessian = i < 18 && i % 6 >= 3;
          const Real tol = hessian ? 1e-8 : MeshField::Epsilon;
          const Real expected = i == k ? 1.0 : 0.0;
          if (!(Kokkos::fabs(f[i] - expected) <= tol)) {
            Kokkos::printf("functional %d of basis function %d: %.3e\n", i, k,
                           f[i]);
            lerr++;
          }
        }
      },
      errors);
  std::cout << "testReferenceDuality: " << (errors ? "FAIL" : "pass") << "\n";
  return errors == 0;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  failed += !testReferenceDuality();
  Kokkos::finalize();
  if (failed) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
