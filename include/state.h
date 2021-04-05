
#pragma once

#include "packages/vio/include/calibration_state.h"
#include "packages/vio/include/clone_state.h"
#include "packages/vio/include/imu_state.h"

#include <deque>

namespace vio {
/// Represents the full state of VIO
template <typename SCALAR> class State {
public:
    static constexpr size_t imu_state_pos = 0;
    static constexpr size_t calibration_state_pos = 15;
    static constexpr size_t clone_state_pos = 21;

    using imu_state = ImuState<SCALAR>;
    using calibration_state = CalibrationState<SCALAR>;
    using clone_state = CloneState<SCALAR>;

    static constexpr size_t initial_state_size = imu_state::state_dim + calibration_state::state_dim;

    State() = default;

    /// Construct a state from variables.
    /// \param imuState IMU state
    /// \param calibrationState Calibration state
    State(const imu_state& imuState, const calibration_state& calibrationState)
        : m_imuState(imuState)
        , m_calibrationState(calibrationState) {}
    /// Construct a state from a vector.
    /// \param x State vector
    /// \param numClones Number of clones in the state vector
    State(const Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>& x, const size_t numClones) {
        m_clones.resize(numClones);
        if (x.rows() != (long)size()) {
            throw std::runtime_error("input vector is wrong size");
        }
        m_imuState = imu_state(x.template segment<imu_state::state_dim>(imu_state_pos));
        m_calibrationState = calibration_state(x.template segment<calibration_state::state_dim>(calibration_state_pos));
        for (size_t i = 0; i < m_clones.size(); i++) {
            m_clones[i] = clone_state(x.segment(clone_state_pos + i * clone_state::state_dim, clone_state::state_dim));
        }
    }
    ~State() = default;

    /// Get the IMU state.
    /// \return IMU state
    imu_state& imu() { return m_imuState; }
    const imu_state& imu() const { return m_imuState; }

    /// Get the calibration state.
    /// \return Calibration state
    calibration_state& calibration() { return m_calibrationState; }
    const calibration_state& calibration() const { return m_calibrationState; }

    /// Clone the IMU
    void cloneImuState() { m_clones.push_back(clone_state(m_imuState.quaternion(), m_imuState.position())); }

    /// Get a list of the clones.
    /// \return Clone list
    std::deque<clone_state>& clones() { return m_clones; }
    const std::deque<clone_state>& clones() const { return m_clones; }

    /// Get the state vector length.
    /// \return Size of state vector
    size_t size() const { return imu_state::state_dim + calibration_state::state_dim + m_clones.size() * clone_state::state_dim; }

    /// Get a vector of the state variables.
    /// \return State vector
    Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> vec() const {
        Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> x(size());
        x.template segment<imu_state::state_dim>(imu_state_pos) = m_imuState.vec();
        x.template segment<calibration_state::state_dim>(calibration_state_pos) = m_calibrationState.vec();
        for (size_t i = 0; i < m_clones.size(); i++) {
            x.segment(clone_state_pos + i * clone_state::state_dim, clone_state::state_dim) = m_clones[i].vec();
        }
        return x;
    }

    /// Update the state with a delta vector.
    /// \param dx Delta vector
    void update(const Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>& dx) {
        const Eigen::Matrix<SCALAR, imu_state::state_dim, 1> imuDx = dx.template segment<imu_state::state_dim>(imu_state_pos);
        const Eigen::Matrix<SCALAR, calibration_state::state_dim, 1> calibrationDx
            = dx.template segment<calibration_state::state_dim>(calibration_state_pos);
        m_imuState.update(imuDx);
        m_calibrationState.update(calibrationDx);
        for (size_t i = 0; i < m_clones.size(); i++) {
            const Eigen::Matrix<SCALAR, clone_state::state_dim, 1> cloneDx
                = dx.segment(clone_state::state_dim, clone_state_pos + i * clone_state::state_dim);
            m_clones[i].update(cloneDx);
        }
    }

    /// Augment the state covariance with a new clone.
    /// \param covariance The input covariance
    /// \return The augmented covariance
    Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> augmentCovariance(
        const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>& covariance) {

        static_assert(clone_state::state_dim == 6, "function only works for specific layout");

        Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> A
            = Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>::Zero(covariance.rows() + clone_state::state_dim, covariance.cols());

        A.topLeftCorner(covariance.rows(), covariance.cols()).setIdentity();
        A.template block<3, 3>(A.rows() - (6 + clone_state::quaterion_pos), imu_state::quaterion_pos).setIdentity();
        A.template block<3, 3>(A.rows() - (6 - clone_state::position_pos), imu_state::position_pos).setIdentity();

        const auto newCovariance = A * covariance * A.transpose();

        return newCovariance;
    }

    /// Marginalize a clone in the covariance.
    /// \param covariance The input covariance
    /// \param numClones The number of clones in the covariance
    /// \param cloneIndex The index of the clone
    /// \return Covariance with the clone marginalized
    Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> marginalizeClone(
        const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>& covariance, const size_t cloneIndex) {

        if (cloneIndex >= m_clones.size()) {
            throw std::out_of_range("clone index our of range");
        }

        Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> newCovariance(
            covariance.rows() - clone_state::state_dim, covariance.cols() - clone_state::state_dim);

        const size_t m = clone_state_pos + clone_state::state_dim * cloneIndex;
        const size_t n = newCovariance.cols() - m;

        newCovariance.topLeftCorner(m, m) = covariance.topLeftCorner(m, m);
        newCovariance.topRightCorner(m, n) = covariance.topRightCorner(m, n);
        newCovariance.bottomRightCorner(n, n) = covariance.bottomRightCorner(n, n);
        newCovariance.bottomLeftCorner(n, m) = newCovariance.topRightCorner(m, n).transpose();

        return newCovariance;
    }

private:
    imu_state m_imuState;
    calibration_state m_calibrationState;
    std::deque<clone_state> m_clones;
};
}
