
#pragma once

#include "packages/imu_propagator/include/imu_propagator.h"
#include "packages/vio/include/kalman_filter_details.h"

namespace vio {
/// Represents the IMU state
template <typename SCALAR> class ImuState {
public:
    static constexpr size_t state_dim = 15;

    static constexpr size_t quaterion_pos = 0;
    static constexpr size_t gyro_bias_pos = 3;
    static constexpr size_t velocity_pos = 6;
    static constexpr size_t accel_bias_pos = 9;
    static constexpr size_t position_pos = 12;

    ImuState() {
        m_quaterion.setIdentity();
        m_gyroBias = { 0, 0, 0 };
        m_velocity = { 0, 0, 0 };
        m_accelBias = { 0, 0, 0 };
        m_position = { 0, 0, 0 };
    }
    /// Construct an IMU state from the variables.
    /// \param quaternion Quaternion from IMU to world
    /// \param gyroBias Gyroscope bias
    /// \param velocity World velocity
    /// \param accelBias Acclerometer bias
    /// \param position Position of IMU in world
    ImuState(const Eigen::Quaternion<SCALAR>& quaternion, const Eigen::Matrix<SCALAR, 3, 1>& gyroBias,
        const Eigen::Matrix<SCALAR, 3, 1>& velocity, const Eigen::Matrix<SCALAR, 3, 1>& accelBias,
        const Eigen::Matrix<SCALAR, 3, 1>& position)
        : m_quaterion(quaternion)
        , m_gyroBias(gyroBias)
        , m_velocity(velocity)
        , m_accelBias(accelBias)
        , m_position(position) {}
    /// Construct an IMU state from a vector.
    /// \param x State vector
    ImuState(const Eigen::Matrix<SCALAR, state_dim, 1>& x) {
        const Eigen::Matrix<SCALAR, 3, 1> imaginary = x.template segment<3>(quaterion_pos);
        m_quaterion = imu_propagator::utils::imaginaryToQuaternion(imaginary);
        m_gyroBias = x.template segment<3>(gyro_bias_pos);
        m_velocity = x.template segment<3>(velocity_pos);
        m_accelBias = x.template segment<3>(accel_bias_pos);
        m_position = x.template segment<3>(position_pos);
    }
    ~ImuState() = default;

    Eigen::Quaternion<SCALAR>& quaternion() { return m_quaterion; }
    Eigen::Matrix<SCALAR, 3, 1>& gyroBias() { return m_gyroBias; }
    Eigen::Matrix<SCALAR, 3, 1>& velocity() { return m_velocity; }
    Eigen::Matrix<SCALAR, 3, 1>& accelBias() { return m_accelBias; }
    Eigen::Matrix<SCALAR, 3, 1>& position() { return m_position; }

    const Eigen::Quaternion<SCALAR>& quaternion() const { return m_quaterion; }
    const Eigen::Matrix<SCALAR, 3, 1>& gyroBias() const { return m_gyroBias; }
    const Eigen::Matrix<SCALAR, 3, 1>& velocity() const { return m_velocity; }
    const Eigen::Matrix<SCALAR, 3, 1>& accelBias() const { return m_accelBias; }
    const Eigen::Matrix<SCALAR, 3, 1>& position() const { return m_position; }

    /// Get a vector of the state variables.
    /// \return State vector
    Eigen::Matrix<SCALAR, state_dim, 1> vec() const {
        Eigen::Matrix<SCALAR, state_dim, 1> x;
        x.template segment<3>(quaterion_pos) = m_quaterion.vec();
        x.template segment<3>(gyro_bias_pos) = m_gyroBias;
        x.template segment<3>(velocity_pos) = m_velocity;
        x.template segment<3>(accel_bias_pos) = m_accelBias;
        x.template segment<3>(position_pos) = m_position;
        return x;
    }

    /// Update the state with a delta vector.
    /// \param dx Delta vector
    void update(const Eigen::Matrix<SCALAR, state_dim, 1>& dx) {
        const Eigen::Matrix<SCALAR, 3, 1> dq = dx.template segment<3>(quaterion_pos);
        m_quaterion = updateQuaternion(m_quaterion, dq);
        m_gyroBias += dx.template segment<3>(gyro_bias_pos);
        m_velocity += dx.template segment<3>(velocity_pos);
        m_accelBias += dx.template segment<3>(accel_bias_pos);
        m_position += dx.template segment<3>(position_pos);
    }

private:
    Eigen::Quaternion<SCALAR> m_quaterion;
    Eigen::Matrix<SCALAR, 3, 1> m_gyroBias;
    Eigen::Matrix<SCALAR, 3, 1> m_velocity;
    Eigen::Matrix<SCALAR, 3, 1> m_accelBias;
    Eigen::Matrix<SCALAR, 3, 1> m_position;
};
}
