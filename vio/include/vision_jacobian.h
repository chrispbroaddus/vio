
#pragma once

#include "Eigen/Dense"
#include "packages/feature_tracker/include/track_database.h"
#include "packages/imu_propagator/include/imu_propagator_utils.h"
#include <iostream>
#include <limits>
#include <vector>

namespace vio {

/// Transform a world point into camera coordinates.
/// y = Rcam'*(Rimu'*x - Rimu'*pimu) - Rcam'*pcam
template <typename T>
inline Eigen::Matrix<T, 3, 1> mapWorldPointToCamera(const Eigen::Quaternion<T>& imuToWorldRotation,
    const Eigen::Matrix<T, 3, 1>& imuPositionInWorld, const Eigen::Quaternion<T>& cameraToImuRotation,
    const Eigen::Matrix<T, 3, 1>& cameraPositionInImu, const Eigen::Matrix<T, 3, 1>& x) {
    return cameraToImuRotation.inverse() * (imuToWorldRotation.inverse() * x - imuToWorldRotation.inverse() * imuPositionInWorld)
        - cameraToImuRotation.inverse() * cameraPositionInImu;
}

/// Compute the jacobian w.r.t rotation.
///
/// Let y = R*expm(w)*x ~ R*(I+ssm(w))*x
///       = (R+R*ssm(w))*x = (R*x)+(R*ssm(w)*x)
///       = (R*x)-(R*ssm(x)*w), anti-commutative property of skew-symmetric matrix
/// dy/dw = -R*ssm(x)
///
/// \param rotation Rotation
/// \param translation Translation
/// \param worldPoint World point
/// \return Jacobian
template <typename T>
inline Eigen::Matrix<T, 3, 3> computeJacobianRotation(const Eigen::Quaternion<T>& rotation, const Eigen::Matrix<T, 3, 1>& worldPoint) {
    return -(rotation * imu_propagator::utils::skewSymmetricMatrix(worldPoint));
}

/// Compute the jacobian w.r.t rotation transposed.
///
/// Let y = (R*expm(w))'*x ~ (R*(I+ssm(w)))'*x
///       = (R+R*ssm(w))'*x = (R'+ssm(w)'*R')*x = (R'*x)+(ssm(w)'*R'*x)
///       = (R'*x)-(ssm(w)*R'*x) = (R'*x)+(ssm(R'*x)*w), anti-commutative property of skew-symmetric matrix
/// dy/dw = ssm(R'*x)
///
/// \param rotation Rotation
/// \param translation Translation
/// \param worldPoint World point
/// \return Jacobian
template <typename T>
inline Eigen::Matrix<T, 3, 3> computeJacobianRotationTranspose(
    const Eigen::Quaternion<T>& rotation, const Eigen::Matrix<T, 3, 1>& worldPoint) {
    return imu_propagator::utils::skewSymmetricMatrix(rotation.inverse() * worldPoint);
}

/// Compute the projection of a point onto the image.
/// \param V 3d point
/// \return 2d projected point
template <typename T> inline Eigen::Matrix<T, 2, 1> perspectiveProjection(const Eigen::Matrix<T, 3, 1>& V) {
    if (V(2) <= std::numeric_limits<T>::epsilon()) {
        throw std::runtime_error("Z component too small");
    }
    return Eigen::Matrix<T, 2, 1>(V(0) / V(2), V(1) / V(2));
}

/// Compute the jacobian of the perspective projection.
/// U = (x/z, y/z)
/// Let V = (x,y,z)
/// DU/DV = [1/z, 0, -x/z^2;
///          0, 1/z, -y/z^2];
template <typename T> inline Eigen::Matrix<T, 2, 3> jacobianPerspectiveProjection(const Eigen::Matrix<T, 3, 1>& V) {
    const T V2squared = V(2) * V(2);
    if (V2squared <= std::numeric_limits<T>::epsilon()) {
        throw std::runtime_error("Z component too small");
    }
    const T oneOverV2 = 1 / V(2);
    Eigen::Matrix<T, 2, 3> jacobian;
    jacobian << oneOverV2, 0, -V(0) / V2squared, 0, oneOverV2, -V(1) / V2squared;
    return jacobian;
}

/// \addtogroup VIO Jacobians
/// @{

/// The VIO jacobians are derived below:
///
///     Rcam - Camera to IMU rotation (calibration)
///     pcam - Position of camera w.r.t IMU (calibration)
///     Rimu - IMU to world rotation
///     pimu - Position of IMU w.r.t. world
///     x    - 3D point in the world
///
/// m = project(Rcam'*(Rimu'*x - Rimu'*pimu) - Rcam'*pcam)
///   = project(Rcam'*Rimu'*x - Rcam'*Rimu'*pimu - Rcam'*pcam)
/// dm/dRimu = dm/du*du/dRimu = dm/du * ((Rcam'*ssm(Rimu'*x) - Rcam'*ssm(Rimu'*pimu))) where dm/du is the jacobian of pinhole projection.
/// dm/dpimu = dm/du*du/dpimu = dm/du * (-Rcam'*Rimu')
/// dm/dRcam = dm/du*du/dRcam = dm/du * (ssm(Rcam'*Rimu'*x) - ssm(Rcam'*Rimu'*pimu) - ssm(Rcam'*pcam)
/// dm/pcam = dm/du*du/pcam = dm/du * (-Rcam')
/// dm/dx = dm/du*du/pcam = dm/du * (Rcam'*Rimu')

template <typename T>
inline Eigen::Matrix<T, 2, 3> computeJacobianImuToWorldRotation(const Eigen::Quaternion<T>& imuToWorldRotation,
    const Eigen::Matrix<T, 3, 1>& imuToWorldPosition, const Eigen::Quaternion<T>& cameraToImuRotation,
    const Eigen::Matrix<T, 3, 1>& cameraToImuPosition, const Eigen::Matrix<T, 3, 1>& worldPoint) {
    const Eigen::Matrix<T, 3, 3> cameraToImuRotationMatrix = cameraToImuRotation.toRotationMatrix();
    const Eigen::Matrix<T, 3, 3> jacobianImuToWorldRotation
        = cameraToImuRotationMatrix.transpose() * imu_propagator::utils::skewSymmetricMatrix(imuToWorldRotation.inverse() * worldPoint)
        - cameraToImuRotationMatrix.transpose()
            * imu_propagator::utils::skewSymmetricMatrix(imuToWorldRotation.inverse() * imuToWorldPosition);
    return jacobianPerspectiveProjection(
               mapWorldPointToCamera(imuToWorldRotation, imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint))
        * jacobianImuToWorldRotation;
}
template <typename T>
inline Eigen::Matrix<T, 2, 3> computeJacobianImuToWorldPosition(const Eigen::Quaternion<T>& imuToWorldRotation,
    const Eigen::Matrix<T, 3, 1>& imuToWorldPosition, const Eigen::Quaternion<T>& cameraToImuRotation,
    const Eigen::Matrix<T, 3, 1>& cameraToImuPosition, const Eigen::Matrix<T, 3, 1>& worldPoint) {
    const Eigen::Matrix<T, 3, 3> jacobianImuToWorldPosition
        = -(cameraToImuRotation.toRotationMatrix().transpose() * imuToWorldRotation.toRotationMatrix().transpose());
    return jacobianPerspectiveProjection(
               mapWorldPointToCamera(imuToWorldRotation, imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint))
        * jacobianImuToWorldPosition;
}
template <typename T>
inline Eigen::Matrix<T, 2, 3> computeJacobianCameraToImuRotation(const Eigen::Quaternion<T>& imuToWorldRotation,
    const Eigen::Matrix<T, 3, 1>& imuToWorldPosition, const Eigen::Quaternion<T>& cameraToImuRotation,
    const Eigen::Matrix<T, 3, 1>& cameraToImuPosition, const Eigen::Matrix<T, 3, 1>& worldPoint) {
    const Eigen::Matrix<T, 3, 3> jacobianCameraToImuRotation
        = imu_propagator::utils::skewSymmetricMatrix(cameraToImuRotation.inverse() * imuToWorldRotation.inverse() * worldPoint)
        - imu_propagator::utils::skewSymmetricMatrix(cameraToImuRotation.inverse() * imuToWorldRotation.inverse() * imuToWorldPosition)
        - imu_propagator::utils::skewSymmetricMatrix(cameraToImuRotation.inverse() * cameraToImuPosition);
    return jacobianPerspectiveProjection(
               mapWorldPointToCamera(imuToWorldRotation, imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint))
        * jacobianCameraToImuRotation;
}
template <typename T>
inline Eigen::Matrix<T, 2, 3> computeJacobianCameraToImuPosition(const Eigen::Quaternion<T>& imuToWorldRotation,
    const Eigen::Matrix<T, 3, 1>& imuToWorldPosition, const Eigen::Quaternion<T>& cameraToImuRotation,
    const Eigen::Matrix<T, 3, 1>& cameraToImuPosition, const Eigen::Matrix<T, 3, 1>& worldPoint) {
    const Eigen::Matrix<T, 3, 3> jacobianCameraToImuPosition = -cameraToImuRotation.toRotationMatrix().transpose();
    return jacobianPerspectiveProjection(
               mapWorldPointToCamera(imuToWorldRotation, imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint))
        * jacobianCameraToImuPosition;
}
template <typename T>
inline Eigen::Matrix<T, 2, 3> computeJacobianWorldPoint(const Eigen::Quaternion<T>& imuToWorldRotation,
    const Eigen::Matrix<T, 3, 1>& imuToWorldPosition, const Eigen::Quaternion<T>& cameraToImuRotation,
    const Eigen::Matrix<T, 3, 1>& cameraToImuPosition, const Eigen::Matrix<T, 3, 1>& worldPoint) {
    const Eigen::Matrix<T, 3, 3> jacobianWorldPoint = (cameraToImuRotation.inverse() * imuToWorldRotation.inverse()).toRotationMatrix();
    return jacobianPerspectiveProjection(
               mapWorldPointToCamera(imuToWorldRotation, imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint))
        * jacobianWorldPoint;
}

/// @}

/// Compute the left null space of a matrix.
template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> computeLeftNullSpace(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& X) {
    const Eigen::HouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > qr(X);
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Q = qr.householderQ();
    return Q.rightCols(X.rows() - X.cols());
}

/// Represents a measurement in the image.
template <typename T> struct ReprojectionMeasurement {
    ReprojectionMeasurement(const Eigen::Quaternion<T>& imuToWorldRotation, const Eigen::Matrix<T, 3, 1>& imuToWorldPosition,
        const Eigen::Matrix<T, 2, 1>& calibratedImagePoint, const int64_t cloneId)
        : m_imuToWorldRotation(imuToWorldRotation)
        , m_imuToWorldPosition(imuToWorldPosition)
        , m_calibratedImagePoint(calibratedImagePoint)
        , m_cloneId(cloneId) {}
    ~ReprojectionMeasurement() = default;

    const Eigen::Quaternion<T> m_imuToWorldRotation;
    const Eigen::Matrix<T, 3, 1> m_imuToWorldPosition;
    const Eigen::Matrix<T, 2, 1> m_calibratedImagePoint;

    const int64_t m_cloneId;
};

/// Compute the nullified residual and jacobian from a collection of 2d image measurements for a single 3d track point.
/// The nullification process removes the world point from the state by projecting the residual and remaining state jacobian
/// onto the left null space of the world point jacobian.
///
/// r(i) = Hx(i)*dX + Hf(i)*dp + n(i) where i is the ith track (collection of 2d measurements)
/// L'*r(i) = L'*Hx(i))*dX + L'*n(i) where L is the left null space of Hf
template <typename T>
void computeTrackJacobian(Eigen::Matrix<T, Eigen::Dynamic, 1>& leftNullSpaceProjectedResidual,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& leftNullSpaceProjectedJacobian,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& leftNullSpace, Eigen::Matrix<T, Eigen::Dynamic, 1>& stackedResidual,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& stackedJacobiansRandomVariables,
    const std::vector<ReprojectionMeasurement<T> >& measurements, const Eigen::Quaternion<T>& cameraToImuRotation,
    const Eigen::Matrix<T, 3, 1>& cameraToImuPosition, const Eigen::Matrix<T, 3, 1>& worldPoint) {

    constexpr size_t calibration_state_dim = 6;
    constexpr size_t clone_state_dim = 6;
    constexpr size_t world_point_dim = 3;
    constexpr size_t residual_dim = 2;
    const size_t numRandomVariables = calibration_state_dim + clone_state_dim * measurements.size();

    if (measurements.size() < 2) {
        throw std::runtime_error("there must be at least 2 measurements/clones");
    }

    stackedResidual = Eigen::Matrix<T, Eigen::Dynamic, 1>(residual_dim * measurements.size());
    stackedJacobiansRandomVariables
        = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(residual_dim * measurements.size(), numRandomVariables);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> stackedJacobianWorldPoint(residual_dim * measurements.size(), world_point_dim);

    for (size_t i = 0; i < measurements.size(); i++) {
        const ReprojectionMeasurement<T>& measurement = measurements[i];

        // Compute residual
        const Eigen::Matrix<T, 2, 1> projectedImagePoint
            = mapWorldPointToCamera(
                  measurement.m_imuToWorldRotation, measurement.m_imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint)
                  .hnormalized();

        const Eigen::Matrix<T, 2, 1> imagePointResidual = projectedImagePoint - measurement.m_calibratedImagePoint;

        // Compute the derivatives of the reprojection error for each free variable
        const Eigen::Matrix<T, 2, 3> jacobianImuToWorldRotation = computeJacobianImuToWorldRotation(
            measurement.m_imuToWorldRotation, measurement.m_imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint);
        const Eigen::Matrix<T, 2, 3> jacobianImuToWorldPosition = computeJacobianImuToWorldPosition(
            measurement.m_imuToWorldRotation, measurement.m_imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint);
        const Eigen::Matrix<T, 2, 3> jacobianCameraToImuRotation = computeJacobianCameraToImuRotation(
            measurement.m_imuToWorldRotation, measurement.m_imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint);
        const Eigen::Matrix<T, 2, 3> jacobianCameraToImuPosition = computeJacobianCameraToImuPosition(
            measurement.m_imuToWorldRotation, measurement.m_imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint);
        const Eigen::Matrix<T, 2, 3> jacobianWorldPoint = computeJacobianWorldPoint(
            measurement.m_imuToWorldRotation, measurement.m_imuToWorldPosition, cameraToImuRotation, cameraToImuPosition, worldPoint);

        // Stackup residual
        stackedResidual.template segment<2>(i * residual_dim) = imagePointResidual;

        // Stackup the jacobian terms for the calibration and IMU clones in a block formation.
        //
        // stackedJacobiansRandomVariables = [Jcalibration0, Jimu0, 0,     ....;
        //                                    Jcalibration1, 0,     Jimu1, ....;
        //                                    ...];
        stackedJacobiansRandomVariables.template block<2, 3>(i * residual_dim, 0) = jacobianCameraToImuRotation;
        stackedJacobiansRandomVariables.template block<2, 3>(i * residual_dim, 3) = jacobianCameraToImuPosition;
        stackedJacobiansRandomVariables.template block<2, 3>(i * residual_dim, calibration_state_dim + i * clone_state_dim)
            = jacobianImuToWorldRotation;
        stackedJacobiansRandomVariables.template block<2, 3>(i * residual_dim, calibration_state_dim + i * clone_state_dim + 3)
            = jacobianImuToWorldPosition;

        // Stackup the world point jacobian
        stackedJacobianWorldPoint.template block<2, 3>(i * residual_dim, 0) = jacobianWorldPoint;
    }

    leftNullSpace = computeLeftNullSpace(stackedJacobianWorldPoint);

    leftNullSpaceProjectedResidual = leftNullSpace.transpose() * stackedResidual;
    leftNullSpaceProjectedJacobian = leftNullSpace.transpose() * stackedJacobiansRandomVariables;
}
/// Parameter reduced function from above.
template <typename T>
void computeTrackJacobian(Eigen::Matrix<T, Eigen::Dynamic, 1>& leftNullSpaceProjectedResidual,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& leftNullSpaceProjectedJacobian,
    const std::vector<ReprojectionMeasurement<T> >& measurements, const Eigen::Quaternion<T>& cameraToImuRotation,
    const Eigen::Matrix<T, 3, 1>& cameraToImuPosition, const Eigen::Matrix<T, 3, 1>& worldPoint) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> leftNullSpace;
    Eigen::Matrix<double, Eigen::Dynamic, 1> stackedResidual;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> stackedJacobiansRandomVariables;
    computeTrackJacobian(leftNullSpaceProjectedResidual, leftNullSpaceProjectedJacobian, leftNullSpace, stackedResidual,
        stackedJacobiansRandomVariables, measurements, cameraToImuRotation, cameraToImuPosition, worldPoint);
}
}
