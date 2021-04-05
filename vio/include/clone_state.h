
#pragma once

#include "packages/imu_propagator/include/imu_propagator.h"
#include "packages/vio/include/kalman_filter_details.h"

namespace vio {
/// Represents the camera clones in the state
template <typename SCALAR> class CloneState {
public:
    static constexpr size_t state_dim = 6;

    static constexpr size_t quaterion_pos = 0;
    static constexpr size_t position_pos = 3;

    CloneState() {
        m_quaterion.setIdentity();
        m_position = { 0, 0, 0 };
    }
    /// Construct the clone state from the state variables.
    /// \param quaternion Quaternion from IMU to world
    /// \param position Position of IMU in world
    CloneState(const Eigen::Quaternion<SCALAR>& quaternion, const Eigen::Matrix<SCALAR, 3, 1>& position)
        : m_quaterion(quaternion)
        , m_position(position) {}

    /// Construct the clone state from a vector.
    /// \param x State vector
    CloneState(const Eigen::Matrix<SCALAR, state_dim, 1>& x) {
        const Eigen::Matrix<SCALAR, 3, 1> imaginary = x.template segment<3>(quaterion_pos);
        m_quaterion = imu_propagator::utils::imaginaryToQuaternion(imaginary);
        m_position = x.template segment<3>(position_pos);
    }
    ~CloneState() = default;

    /// Get the orientation from IMU to camera.
    /// \return Quaternion
    Eigen::Quaternion<SCALAR>& quaternion() { return m_quaterion; }
    const Eigen::Quaternion<SCALAR>& quaternion() const { return m_quaterion; }

    /// Get the position of the camera w.r.t. the IMU.
    /// \return Position vector
    Eigen::Matrix<SCALAR, 3, 1>& position() { return m_position; }
    const Eigen::Matrix<SCALAR, 3, 1>& position() const { return m_position; }

    /// Get a vector of the state variables.
    /// \return State vector
    Eigen::Matrix<SCALAR, state_dim, 1> vec() const {
        Eigen::Matrix<SCALAR, state_dim, 1> x;
        x.template segment<3>(quaterion_pos) = m_quaterion.vec();
        x.template segment<3>(position_pos) = m_position;
        return x;
    }

    /// Update the state with a delta vector.
    /// \param dx Delta vector
    void update(const Eigen::Matrix<SCALAR, state_dim, 1>& dx) {
        const Eigen::Matrix<SCALAR, 3, 1> dq = dx.template segment<3>(quaterion_pos);
        m_quaterion = updateQuaternion(m_quaterion, dq);
        m_position += dx.template segment<3>(position_pos);
    }

private:
    Eigen::Quaternion<SCALAR> m_quaterion;
    Eigen::Matrix<SCALAR, 3, 1> m_position;
};
}
