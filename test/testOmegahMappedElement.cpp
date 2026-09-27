#include "MeshField.hpp"
#include "MeshField_Integrate.hpp"
#include "MeshField_MappedElement.hpp"
#include "Omega_h_build.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <type_traits>

// MappedElement tests on triangles and tetrahedra.  The meshes are sheared so
// the Jacobians are full matrices.
//
// 1. P1 and P2 fields on P1 geometry: the map is affine, so a field that
//    interpolates a linear (P1) or quadratic (P2) f reproduces f and grad f
//    exactly.  Component c of the field is (c+1)*f, which checks the
//    component ordering.
// 2. finite differences on curved P2 geometry: for P1 and P2 fields the
//    physical gradient g must satisfy dx/dxi_k . g = dv/dxi_k, with central
//    differences on the right.
// 3. measure through the Integrator: bulging boundary edge nodes by s adds
//    (2/3)*chord*s of area per edge (2d); shearing keeps the volume (3d).
// 4. a DofTransformation whose reference basis differs from the stored dofs:
//    P1 dofs interpolated onto the P2 nodes, plus a bubble b = xi0*xi1 that
//    only the P2 basis represents, must give the P1 values plus b and the P1
//    gradients plus the mapped gradient of b.

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
template <size_t dim> using Vec = Kokkos::Array<MeshField::Real, dim>;
template <size_t dim>
using OmegahMeshField = MeshField::OmegahMeshField<ExecutionSpace, dim>;

template <size_t dim> struct LinearFunction {
  KOKKOS_INLINE_FUNCTION MeshField::Real operator()(Vec<dim> x) const {
    MeshField::Real f = 0;
    for (size_t d = 0; d < dim; ++d)
      f += (d + 2.0) * x[d];
    return f;
  }
  KOKKOS_INLINE_FUNCTION Vec<dim> grad(Vec<dim>) const {
    Vec<dim> g;
    for (size_t d = 0; d < dim; ++d)
      g[d] = d + 2.0;
    return g;
  }
};

template <size_t dim> struct QuadraticFunction {
  KOKKOS_INLINE_FUNCTION MeshField::Real operator()(Vec<dim> x) const {
    if constexpr (dim == 2)
      return x[0] * x[0] + 2.0 * x[1];
    else
      return x[0] * x[0] + 2.0 * x[1] * x[2] + x[2];
  }
  KOKKOS_INLINE_FUNCTION Vec<dim> grad(Vec<dim> x) const {
    if constexpr (dim == 2)
      return {2.0 * x[0], 2.0};
    else
      return {2.0 * x[0], 2.0 * x[2], 2.0 * x[1] + 1.0};
  }
};

// interior parametric coordinates; the first dim entries are used
constexpr int numPts = 3;
constexpr MeshField::Real pts[numPts][3] = {
    {0.25, 0.25, 0.25}, {0.1, 0.5, 0.15}, {0.4, 0.1, 0.3}};

template <size_t dim> KOKKOS_INLINE_FUNCTION Vec<dim> point(int pt) {
  Vec<dim> xi;
  for (size_t d = 0; d < dim; ++d)
    xi[d] = pts[pt][d];
  return xi;
}

template <size_t dim, typename CoordField>
KOKKOS_INLINE_FUNCTION Vec<dim> nodePosition(CoordField const &cf, int ent,
                                             MeshField::Mesh_Topology topo) {
  Vec<dim> x;
  for (size_t d = 0; d < dim; ++d)
    x[d] = cf(ent, 0, d, topo);
  return x;
}

// call f(ent, topo) for every node of a Lagrange field
template <typename Field, typename Func>
void forEachNode(Omega_h::Mesh &mesh, Func f) {
  Kokkos::parallel_for(
      "vtxNodes", mesh.nverts(),
      KOKKOS_LAMBDA(const int i) { f(i, MeshField::Vertex); });
  if constexpr (Field::Order == 2) {
    Kokkos::parallel_for(
        "edgeNodes", mesh.nedges(),
        KOKKOS_LAMBDA(const int i) { f(i, MeshField::Edge); });
  }
}

// x += 0.3y + 0.1z, y += 0.2z; affine, so edge midpoints stay midpoints
template <size_t dim, typename CoordField>
void shear(Omega_h::Mesh &mesh, CoordField cf) {
  forEachNode<CoordField>(
      mesh, KOKKOS_LAMBDA(const int ent, const MeshField::Mesh_Topology topo) {
        const auto x = nodePosition<dim>(cf, ent, topo);
        cf(ent, 0, 0, topo) = x[0] + 0.3 * x[1];
        if constexpr (dim == 3) {
          cf(ent, 0, 0, topo) += 0.1 * x[2];
          cf(ent, 0, 1, topo) = x[1] + 0.2 * x[2];
        }
      });
}

// move every edge node by a smooth offset, curving the elements
template <size_t dim, typename CoordField>
void curveEdgeNodes(Omega_h::Mesh &mesh, CoordField cf, MeshField::Real s) {
  Kokkos::parallel_for(
      "curveEdges", mesh.nedges(), KOKKOS_LAMBDA(const int i) {
        const auto x = nodePosition<dim>(cf, i, MeshField::Edge);
        for (size_t d = 0; d < dim; ++d)
          cf(i, 0, d, MeshField::Edge) =
              x[d] + s * Kokkos::sin(3.0 * x[(d + 1) % dim] + d + 1);
      });
}

// set component c of a Lagrange field to (c+1)*func at the node positions of
// a coordinate field of the same order
template <size_t dim, typename Func, typename Field, typename CoordField>
void setField(Omega_h::Mesh &mesh, Func func, Field field, CoordField cf) {
  static_assert(Field::Order == CoordField::Order);
  forEachNode<Field>(
      mesh, KOKKOS_LAMBDA(const int ent, const MeshField::Mesh_Topology topo) {
        const auto f = func(nodePosition<dim>(cf, ent, topo));
        for (size_t c = 0; c < Field::numComp; ++c)
          field(ent, 0, c, topo) = (c + 1) * f;
      });
}

// 1. a field on P1 geometry reproduces a polynomial of its order exactly
template <size_t dim, size_t FieldOrder>
bool testFieldOnP1Geometry(Omega_h::Mesh &mesh, OmegahMeshField<dim> &omf) {
  std::conditional_t<FieldOrder == 1, LinearFunction<dim>,
                     QuadraticFunction<dim>>
      func;
  auto cf1Wc = omf.template CreateLagrangeCoordinateField<1>();
  auto nodeCoordsWc = omf.template CreateLagrangeCoordinateField<FieldOrder>();
  shear<dim>(mesh, cf1Wc.field);
  shear<dim>(mesh, nodeCoordsWc.field);
  constexpr size_t numComp = 2;
  auto fieldWc =
      omf.template CreateLagrangeField<MeshField::Real, FieldOrder, numComp>();
  setField<dim>(mesh, func, fieldWc.field, nodeCoordsWc.field);

  auto geomElm = omf.makeElement(cf1Wc.field);
  auto elm = omf.makeElement(fieldWc.field, geomElm);
  auto coordElm = omf.makeElement(cf1Wc.field, geomElm);

  MeshField::LO errors = 0;
  Kokkos::parallel_reduce(
      "checkFieldOnP1Geometry", mesh.nelems(),
      KOKKOS_LAMBDA(const int ent, MeshField::LO &lerr) {
        const auto c = elm.coefficients(ent);
        const auto geo = elm.geometry(ent);
        const auto cx = coordElm.coefficients(ent);
        for (int pt = 0; pt < numPts; ++pt) {
          const auto xi = point<dim>(pt);
          const auto pos = coordElm.valueFromCoefficients(cx, xi);
          Vec<dim> x;
          for (size_t d = 0; d < dim; ++d)
            x[d] = pos[d];
          const auto v = elm.valueFromCoefficients(c, xi);
          const auto g = elm.gradientFromCoefficients(c, geo, xi);
          const auto gf = func.grad(x);
          for (size_t ci = 0; ci < numComp; ++ci) {
            if (!(Kokkos::fabs(v[ci] - (ci + 1) * func(x)) <=
                  MeshField::Epsilon))
              lerr++;
            for (size_t d = 0; d < dim; ++d)
              if (!(Kokkos::fabs(g[ci * dim + d] - (ci + 1) * gf[d]) <=
                    MeshField::Epsilon))
                lerr++;
          }
        }
      },
      errors);
  std::cout << "testFieldOnP1Geometry(" << dim << "d, P" << FieldOrder
            << " field): " << (errors ? "FAIL" : "pass") << "\n";
  return errors == 0;
}

// 2. physical gradients on curved geometry agree with central differences
template <size_t dim, size_t FieldOrder>
bool testCurvedGradientFiniteDifference(Omega_h::Mesh &mesh,
                                        OmegahMeshField<dim> &omf) {
  QuadraticFunction<dim> func;
  auto cfWc = omf.template CreateLagrangeCoordinateField<2>();
  shear<dim>(mesh, cfWc.field);
  curveEdgeNodes<dim>(mesh, cfWc.field, 0.03);
  auto fieldWc =
      omf.template CreateLagrangeField<MeshField::Real, FieldOrder, 1>();
  auto nodeCoordsWc = omf.template CreateLagrangeCoordinateField<FieldOrder>();
  shear<dim>(mesh, nodeCoordsWc.field);
  setField<dim>(mesh, func, fieldWc.field, nodeCoordsWc.field);

  auto geomElm = omf.makeElement(cfWc.field);
  auto elm = omf.makeElement(fieldWc.field, geomElm);
  auto coordElm = omf.makeElement(cfWc.field, geomElm);

  const MeshField::Real h = 1e-5;
  MeshField::LO errors = 0;
  Kokkos::parallel_reduce(
      "checkFiniteDifference", mesh.nelems(),
      KOKKOS_LAMBDA(const int ent, MeshField::LO &lerr) {
        const auto c = elm.coefficients(ent);
        const auto geo = elm.geometry(ent);
        const auto cx = coordElm.coefficients(ent);
        for (int pt = 0; pt < numPts; ++pt) {
          const auto xi = point<dim>(pt);
          const auto g = elm.gradientFromCoefficients(c, geo, xi);
          for (size_t k = 0; k < dim; ++k) {
            auto xp = xi;
            auto xm = xi;
            xp[k] += h;
            xm[k] -= h;
            const auto dv = (elm.valueFromCoefficients(c, xp)[0] -
                             elm.valueFromCoefficients(c, xm)[0]) /
                            (2 * h);
            const auto posP = coordElm.valueFromCoefficients(cx, xp);
            const auto posM = coordElm.valueFromCoefficients(cx, xm);
            MeshField::Real dvFromGrad = 0;
            for (size_t j = 0; j < dim; ++j)
              dvFromGrad += (posP[j] - posM[j]) / (2 * h) * g[j];
            if (!(Kokkos::fabs(dvFromGrad - dv) <= 1e-7))
              lerr++;
          }
        }
      },
      errors);
  std::cout << "testCurvedGradientFiniteDifference(" << dim << "d, P"
            << FieldOrder << " field): " << (errors ? "FAIL" : "pass") << "\n";
  return errors == 0;
}

// compute the measure (3D volume, 2D area, 1D length)
class MeasureIntegrator : public MeshField::Integrator {
public:
  MeasureIntegrator() : MeshField::Integrator(2) {}
  void atPoints(Kokkos::View<MeshField::Real **>,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) override {
    MeshField::Real sum = 0;
    Kokkos::parallel_reduce(
        "integrateMeasure", w.extent(0),
        KOKKOS_LAMBDA(const int i, MeshField::Real &lsum) {
          lsum += w(i) * dV(i);
        },
        sum);
    measure += sum;
  }
  MeshField::Real measure = 0;
};

// 3. measure of P2 geometry through an Integrator over a MappedElement
template <size_t dim>
bool testMeasure(Omega_h::Mesh &mesh, OmegahMeshField<dim> &omf) {
  auto cfWc = omf.template CreateLagrangeCoordinateField<2>();
  auto cf = cfWc.field;
  MeshField::Real expected = 0;
  if constexpr (dim == 2) {
    // bulge the midpoints of the edges on y=1 (unit square, 3x3 divisions)
    const MeshField::Real bulge = 0.1;
    const auto edgeVerts = mesh.ask_verts_of(Omega_h::EDGE);
    const auto coords = mesh.coords();
    MeshField::LO numCurvedEdges = 0;
    Kokkos::parallel_reduce(
        "bulgeTopEdges", mesh.nedges(),
        KOKKOS_LAMBDA(const int i, MeshField::LO &lcount) {
          const auto v0 = edgeVerts[i * 2];
          const auto v1 = edgeVerts[i * 2 + 1];
          if (Kokkos::fabs(coords[v0 * 2 + 1] - 1.0) < MeshField::Epsilon &&
              Kokkos::fabs(coords[v1 * 2 + 1] - 1.0) < MeshField::Epsilon) {
            cf(i, 0, 1, MeshField::Edge) += bulge;
            lcount++;
          }
        },
        numCurvedEdges);
    if (numCurvedEdges != 3) {
      std::cout << "testMeasure(2d): expected 3 top edges, found "
                << numCurvedEdges << " FAIL\n";
      return false;
    }
    const MeshField::Real chord = 1.0 / 3.0;
    expected = 1.0 + numCurvedEdges * (2.0 / 3.0) * chord * bulge;
  } else {
    shear<dim>(mesh, cf); // 1x2x3 box
    expected = 6.0;
  }
  auto fieldWc = omf.template CreateLagrangeField<MeshField::Real, 1, 1>();
  auto elm = omf.makeElement(fieldWc.field, omf.makeElement(cf));
  MeasureIntegrator integrator;
  integrator.process(elm);
  const bool pass =
      std::abs(integrator.measure - expected) <= MeshField::Epsilon;
  std::cout << "testMeasure(" << dim << "d): " << integrator.measure
            << " expected " << expected << (pass ? " pass" : " FAIL") << "\n";
  return pass;
}

// parametric bubble xi0*xi1: zero at the vertices, in P2
template <size_t dim>
KOKKOS_INLINE_FUNCTION MeshField::Real bubble(Vec<dim> xi) {
  return xi[0] * xi[1];
}

// interpolates the stored dofs plus (c+1)*bubble onto the nodes of a Lagrange
// RefShape, so the coefficients are for a different basis than the stored
// dofs and use all of its nodes
template <typename RefShape> struct InterpolateToShape {
  template <typename FieldElem> using ReferenceShape = RefShape;

  template <typename FieldElem, typename GeomElem>
  KOKKOS_INLINE_FUNCTION auto coefficients(FieldElem const &field,
                                           GeomElem const &, int ent) const {
    constexpr size_t dim = FieldElem::MeshEntDim;
    constexpr size_t numComp = FieldElem::NumComponents;
    const auto d = field.coefficients(ent);
    const auto nodes = RefShape{}.getNodeParametricCoords();
    Kokkos::Array<MeshField::Real, RefShape::numNodes * numComp> c;
    for (size_t n = 0; n < RefShape::numNodes; ++n) {
      Vec<dim> xi;
      for (size_t k = 0; k < dim; ++k)
        xi[k] = nodes[n * dim + k];
      const auto v = field.valueFromCoefficients(d, xi);
      for (size_t ci = 0; ci < numComp; ++ci)
        c[n * numComp + ci] = v[ci] + (ci + 1) * bubble<dim>(xi);
    }
    return c;
  }
};

// 4. P1 dofs plus a bubble on a P2 reference basis
template <size_t dim>
bool testReferenceShape(Omega_h::Mesh &mesh, OmegahMeshField<dim> &omf) {
  using P2Shape =
      std::conditional_t<dim == 2, MeshField::QuadraticTriangleShape,
                         MeshField::QuadraticTetrahedronShape>;
  auto cfWc = omf.template CreateLagrangeCoordinateField<1>();
  shear<dim>(mesh, cfWc.field);
  constexpr size_t numComp = 2;
  auto fieldWc =
      omf.template CreateLagrangeField<MeshField::Real, 1, numComp>();
  setField<dim>(mesh, LinearFunction<dim>(), fieldWc.field, cfWc.field);

  auto geomElm = omf.makeElement(cfWc.field);
  auto p1 = omf.makeElement(fieldWc.field, geomElm);
  auto onP2 = MeshField::MappedElement(omf.makeElement(fieldWc.field), geomElm,
                                       InterpolateToShape<P2Shape>{});
  static_assert(sizeof(decltype(onP2.coefficients(0))) ==
                P2Shape::numNodes * numComp * sizeof(MeshField::Real));

  MeshField::LO errors = 0;
  Kokkos::parallel_reduce(
      "checkReferenceShape", mesh.nelems(),
      KOKKOS_LAMBDA(const int ent, MeshField::LO &lerr) {
        const auto c1 = p1.coefficients(ent);
        const auto c2 = onP2.coefficients(ent);
        const auto geo = p1.geometry(ent);
        for (int pt = 0; pt < numPts; ++pt) {
          const auto xi = point<dim>(pt);
          const auto v1 = p1.valueFromCoefficients(c1, xi);
          const auto v2 = onP2.valueFromCoefficients(c2, xi);
          const auto g1 = p1.gradientFromCoefficients(c1, geo, xi);
          const auto g2 = onP2.gradientFromCoefficients(c2, geo, xi);
          // physical gradient of the bubble, for each component
          Kokkos::Array<MeshField::Real, numComp * dim> localBubbleGrad;
          for (size_t ci = 0; ci < numComp; ++ci)
            for (size_t d = 0; d < dim; ++d)
              localBubbleGrad[ci * dim + d] =
                  (ci + 1) * (d == 0 ? xi[1] : (d == 1 ? xi[0] : 0.0));
          const auto bubbleGrad =
              MeshField::IdentityMap{}.gradient(localBubbleGrad, geo);
          for (size_t ci = 0; ci < numComp; ++ci) {
            const auto expected = v1[ci] + (ci + 1) * bubble<dim>(xi);
            if (!(Kokkos::fabs(v2[ci] - expected) <= MeshField::Epsilon))
              lerr++;
            for (size_t d = 0; d < dim; ++d)
              if (!(Kokkos::fabs(
                        g2[ci * dim + d] -
                        (g1[ci * dim + d] + bubbleGrad[ci * dim + d])) <=
                    MeshField::Epsilon))
                lerr++;
          }
        }
      },
      errors);
  std::cout << "testReferenceShape(" << dim
            << "d): " << (errors ? "FAIL" : "pass") << "\n";
  return errors == 0;
}

template <size_t dim> int runTests(Omega_h::Mesh &mesh) {
  OmegahMeshField<dim> omf(mesh);
  int failed = 0;
  failed += !testFieldOnP1Geometry<dim, 1>(mesh, omf);
  failed += !testFieldOnP1Geometry<dim, 2>(mesh, omf);
  failed += !testCurvedGradientFiniteDifference<dim, 1>(mesh, omf);
  failed += !testCurvedGradientFiniteDifference<dim, 2>(mesh, omf);
  failed += !testMeasure<dim>(mesh, omf);
  failed += !testReferenceShape<dim>(mesh, omf);
  return failed;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  {
    auto lib = Omega_h::Library(&argc, &argv);
    MeshField::Debug = true;
    auto tris = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0,
                                   3, 3, 0);
    auto tets = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 2.0, 3.0,
                                   2, 2, 2);
    failed += runTests<2>(tris);
    failed += runTests<3>(tets);
  }
  Kokkos::finalize();
  if (failed) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
