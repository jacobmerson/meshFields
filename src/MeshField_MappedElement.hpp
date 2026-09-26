#ifndef MESHFIELD_MAPPEDELEMENT_HPP
#define MESHFIELD_MAPPEDELEMENT_HPP

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Element.hpp>
#include <MeshField_Fail.hpp>
#include <utility> // std::declval

/**
 * @file MeshField_MappedElement.hpp
 * @brief Elements whose geometry, dofs, or values need mapping between the
 * reference element and the physical element.
 *
 * A MappedElement composes three things:
 *
 *  - **a geometry element**: a FieldElement over the coordinate field that
 *    supplies the map from the reference element to the physical element,
 *    and hence the Jacobian.  Using a different shape for the geometry than
 *    for the field is what makes an element non-isoparametric.
 *  - **a DofTransformation**: converts the dofs stored on mesh entities to
 *    coefficients on the reference basis, once per element.  This is the
 *    matrix \f$M\f$ of R.C. Kirby, "A general approach to transforming finite
 *    elements", SMAI J. Comput. Math. 4 (2018) 197-224.  It is the identity
 *    for Lagrange elements; for elements such as Bell or Argyris it is a
 *    general matrix built from the element geometry.  Note this is more than
 *    the edge-orientation permutations that "DOF transformation" often
 *    refers to (e.g., in Basix).
 *  - **a MapType**: converts values and derivatives on the reference element
 *    to the physical element, at each point.  The names follow DefElement
 *    (https://defelement.org): identity for H1 elements, covariant Piola for
 *    H(curl), contravariant Piola for H(div), and so on.
 */

namespace MeshField {

namespace detail {
/**
 * @brief invert a 1x1, 2x2, or 3x3 matrix stored row-major
 */
template <size_t Dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, Dim * Dim>
invert(Kokkos::Array<Real, Dim * Dim> const &A) {
  static_assert(Dim >= 1 && Dim <= 3, "only 1x1, 2x2, and 3x3 are supported");
  Kokkos::Array<Real, Dim * Dim> inv;
  if constexpr (Dim == 1) {
    assert(A[0] != 0);
    inv[0] = 1 / A[0];
  } else if constexpr (Dim == 2) {
    const auto det = A[0] * A[3] - A[1] * A[2];
    assert(det != 0);
    inv[0] = A[3] / det;
    inv[1] = -A[1] / det;
    inv[2] = -A[2] / det;
    inv[3] = A[0] / det;
  } else {
    // adjugate divided by the determinant
    inv[0] = A[4] * A[8] - A[5] * A[7];
    inv[1] = A[2] * A[7] - A[1] * A[8];
    inv[2] = A[1] * A[5] - A[2] * A[4];
    inv[3] = A[5] * A[6] - A[3] * A[8];
    inv[4] = A[0] * A[8] - A[2] * A[6];
    inv[5] = A[2] * A[3] - A[0] * A[5];
    inv[6] = A[3] * A[7] - A[4] * A[6];
    inv[7] = A[1] * A[6] - A[0] * A[7];
    inv[8] = A[0] * A[4] - A[1] * A[3];
    const auto det = A[0] * inv[0] + A[1] * inv[3] + A[2] * inv[6];
    assert(det != 0);
    for (size_t i = 0; i < 9; ++i)
      inv[i] /= det;
  }
  return inv;
}
} // namespace detail

/**
 * @brief The DofTransformation for elements whose stored dofs already are the
 * coefficients on the reference basis.
 *
 * @details
 * Correct for every Lagrange element.  A DofTransformation must provide
 * @code
 * template <typename FieldElem, typename GeomElem>
 * KOKKOS_INLINE_FUNCTION auto
 * coefficients(FieldElem const &field, GeomElem const &geom, int ent) const;
 * @endcode
 * returning coefficients indexed `[node * NumComponents + component]`.  It
 * takes no parametric coordinate: the transformation depends only on the
 * element, so it is computed once per element and reused at every point.
 */
struct IdentityDofTransformation {
  template <typename FieldElem, typename GeomElem>
  KOKKOS_INLINE_FUNCTION auto coefficients(FieldElem const &field,
                                           GeomElem const &, int ent) const {
    return field.coefficients(ent);
  }
};

/**
 * @brief The MapType for H1 elements: values are unchanged and gradients map
 * with the inverse Jacobian.
 *
 * @details
 * A MapType must provide
 *  - `value(ref)`: the physical value from the value on the reference
 *    element
 *  - `gradient(refGrad, J)`: physical gradients, indexed
 *    `[component * Dim + direction]`, from gradients with respect to the
 *    parametric coordinates and the Jacobian at the same point
 *
 * The Piola maps also need the Jacobian for values; `value` will gain a
 * Jacobian argument when the first of them is added.
 */
struct IdentityMap {
  template <typename ValArray>
  KOKKOS_INLINE_FUNCTION ValArray value(ValArray const &ref) const {
    return ref;
  }

  /**
   * @details
   * With J[i][j] = dx_j/dxi_i the chain rule gives grad_xi = J grad_x, so
   * grad_x = J^{-1} grad_xi for each component.
   */
  template <typename GradArray, typename JacobianArray>
  KOKKOS_INLINE_FUNCTION GradArray gradient(GradArray const &refGrad,
                                            JacobianArray const &J) const {
    constexpr size_t Dim = JacobianArray::size() == 1   ? 1
                           : JacobianArray::size() == 4 ? 2
                                                        : 3;
    static_assert(Dim * Dim == JacobianArray::size());
    constexpr auto numComp = GradArray::size() / Dim;
    const auto Jinv = detail::invert<Dim>(J);
    GradArray g;
    for (size_t ci = 0; ci < numComp; ++ci) {
      for (size_t i = 0; i < Dim; ++i) {
        Real sum = 0;
        for (size_t j = 0; j < Dim; ++j)
          sum += Jinv[i * Dim + j] * refGrad[ci * Dim + j];
        g[ci * Dim + i] = sum;
      }
    }
    return g;
  }
};

/**
 * @brief An element composed of a field element, a geometry element, a
 * DofTransformation, and a MapType.
 *
 * @details
 * The field element supplies the basis and the stored dofs.  The geometry
 * element supplies the map from the reference element to the physical
 * element.  It carries its own field accessor and dof map, so coordinates
 * are always read from the coordinate field, never from the field being
 * evaluated.  An isoparametric Lagrange element is the case where both use
 * the same shape and both policies are the identity.
 *
 * Evaluation is two-phase, like FieldElement:
 * @code
 * const auto c = elm.coefficients(ent);              // once per element
 * for (pt : points) elm.valueFromCoefficients(c, xi); // once per point
 * @endcode
 * The coefficients include the geometry's node coordinates, so they are also
 * gathered once per element.
 *
 * @tparam FieldElem a FieldElement over the field being evaluated
 * @tparam GeomElem a FieldElement over the coordinate field
 * @tparam DofTransformation see IdentityDofTransformation
 * @tparam MapType see IdentityMap
 */
template <typename FieldElem, typename GeomElem,
          typename DofTransformation = IdentityDofTransformation,
          typename MapType = IdentityMap>
struct MappedElement {
  static_assert(FieldElem::MeshEntDim == GeomElem::MeshEntDim,
                "the field and geometry elements must have the same "
                "parametric dimension");

  FieldElem field;
  GeomElem geom;
  DofTransformation dofTransformation;
  MapType mapType;
  const size_t numMeshEnts;

  static constexpr size_t MeshEntDim = FieldElem::MeshEntDim;
  static constexpr size_t NumComponents = FieldElem::NumComponents;
  using ValArray = typename FieldElem::ValArray;
  using Shape = typename FieldElem::Shape;
  using DofMap = typename FieldElem::DofMap;
  using GeometryShape = typename GeomElem::Shape;
  /** @brief physical gradients at one point, indexed
   *  [component * MeshEntDim + direction] */
  using GradArray = typename FieldElem::LocalGradArray;
  /** @brief the Jacobian at one point, row-major with J[i][j] =
   *  dx_j/dxi_i */
  using JacobianArray = typename GeomElem::JacobianArray;

  /** @brief everything about one element that does not depend on the
   *  evaluation point */
  struct Coefficients {
    /** coefficients on the reference basis, from the DofTransformation */
    decltype(std::declval<DofTransformation>().coefficients(
        std::declval<FieldElem>(), std::declval<GeomElem>(), 0)) field;
    /** the geometry element's node coordinates */
    typename GeomElem::NodeArray geometry;
  };

  MappedElement(
      FieldElem const &fieldIn, GeomElem const &geomIn,
      DofTransformation const &dofTransformationIn = DofTransformation(),
      MapType const &mapTypeIn = MapType())
      : field(fieldIn), geom(geomIn), dofTransformation(dofTransformationIn),
        mapType(mapTypeIn), numMeshEnts(fieldIn.numMeshEnts) {
    if (geomIn.numMeshEnts != fieldIn.numMeshEnts) {
      fail("the field and geometry elements must cover the same number of "
           "mesh entities (%zu != %zu)\n",
           fieldIn.numMeshEnts, geomIn.numMeshEnts);
    }
  }

  /**
   * @brief the coefficients and geometry of one element
   *
   * Depends only on the element; compute it once per element and pass it to
   * the *FromCoefficients functions for each point.
   */
  KOKKOS_INLINE_FUNCTION Coefficients coefficients(int ent) const {
    return {dofTransformation.coefficients(field, geom, ent),
            geom.getNodeValues(ent)};
  }

  /** @brief the physical value of the field at a parametric coordinate */
  KOKKOS_INLINE_FUNCTION ValArray valueFromCoefficients(
      Coefficients const &c, Kokkos::Array<Real, MeshEntDim> localCoord) const {
    return mapType.value(field.valueFromCoefficients(c.field, localCoord));
  }

  /** @brief the Jacobian of the geometry at a parametric coordinate */
  KOKKOS_INLINE_FUNCTION JacobianArray jacobianFromCoefficients(
      Coefficients const &c, Kokkos::Array<Real, MeshEntDim> localCoord) const {
    return geom.jacobianFromNodeValues(c.geometry, localCoord);
  }

  /** @brief the physical gradient of the field at a parametric coordinate */
  KOKKOS_INLINE_FUNCTION GradArray gradientFromCoefficients(
      Coefficients const &c, Kokkos::Array<Real, MeshEntDim> localCoord) const {
    return mapType.gradient(
        field.localGradientFromCoefficients(c.field, localCoord),
        jacobianFromCoefficients(c, localCoord));
  }

  /**
   * @brief evaluate the field at a single parametric coordinate
   *
   * Prefer coefficients and valueFromCoefficients when evaluating an element
   * at more than one point.
   */
  KOKKOS_INLINE_FUNCTION ValArray
  getValue(int ent, Kokkos::Array<Real, MeshEntDim> localCoord) const {
    return valueFromCoefficients(coefficients(ent), localCoord);
  }

  /** @brief Jacobians, from the geometry element */
  Kokkos::View<Real ***> getJacobians(Kokkos::View<Real **> localCoords,
                                      Kokkos::View<LO *> offsets) const {
    return geom.getJacobians(localCoords, offsets);
  }

  /** @brief Jacobian determinants, from the geometry element */
  template <typename Matrices>
  Kokkos::View<Real *> getJacobianDeterminants(Matrices const &J) const {
    return geom.getJacobianDeterminants(J);
  }
};

} // namespace MeshField

#endif
