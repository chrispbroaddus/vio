
#pragma once

#include "packages/imu_propagator/include/imu_propagator.h"
#include <iostream>

namespace vio {

/// Perform a Kalman Filter Update step.
/// \param xp State correction (not the actual state)
/// \param Pp Updated state covariance
/// \param P Current covariance
/// \param y Measurement residual
/// \param H Measurement jacobian
/// \param R Measurement covariance
template <typename T>
void updateKalmanFilter(Eigen::Matrix<T, Eigen::Dynamic, 1>& dx, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Pp,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P, const Eigen::Matrix<T, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& H, const Eigen::Matrix<T, Eigen::Dynamic, 1>& R) {
    const auto PtimesHtranspose = P * H.transpose();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S = H * PtimesHtranspose;
    S.diagonal() += R;
    const auto K = PtimesHtranspose * S.inverse();
    dx = K * y;
    Pp = (P - P * K * H);
}

/// Update a quaternion with an SO3 incremental rotation.
/// \param q Input quaternion
/// \param dq SO3 incremental rotation
/// \return Updated quaternion
template <typename T> inline Eigen::Quaternion<T> updateQuaternion(const Eigen::Quaternion<T>& q, const Eigen::Matrix<T, 3, 1>& dq) {
    return q * Eigen::Quaternion<T>(Eigen::AngleAxis<T>(dq.norm(), dq.normalized()));
}
}
