
#pragma once

#include "glog/logging.h"
#include "packages/feature_tracker/include/track_database.h"
#include "packages/imu_propagator/include/imu_database.h"
#include "packages/imu_propagator/include/imu_propagator.h"
#include "packages/vio/include/kalman_filter_details.h"
#include "packages/vio/include/state.h"
#include "packages/vio/include/vision_jacobian.h"

#include <algorithm>
#include <deque>

namespace vio {

using frame_id = int64_t;
using track_id = int64_t;
using point_type = feature_tracker::FeaturePoint<track_id>;
using frame_type = feature_tracker::Frame<frame_id, track_id, point_type>;
using track_type = feature_tracker::FeatureTrack<frame_id, track_id, double>;
using track_database_type = feature_tracker::TrackDatabase<frame_id, track_id, frame_type, track_type, point_type>;

using timestamp_type = double;
using imu_database_type = imu_propagator::ImuDatabase<timestamp_type, double>;
using imu_sample_type = imu_propagator::details::ImuSample<timestamp_type, double>;
using imu_propagator_type = imu_propagator::ImuPropagator<timestamp_type, double>;

using matrix_type = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

/// Implements the Visual Inertial Kalman Filter for position and orientation state estimation.
template <typename SCALAR> class VisualInertialExtendedKalmanFilter {
public:
    using state_type = State<SCALAR>;

    VisualInertialExtendedKalmanFilter(imu_propagator_type& imuPropagator, const size_t numLocalFramesInWindow,
        const timestamp_type timestamp, const State<SCALAR>& initialState,
        const Eigen::Matrix<SCALAR, state_type::initial_state_size, state_type::initial_state_size>& initialCovariance);
    ~VisualInertialExtendedKalmanFilter() = default;

    void predict(const imu_database_type& imuDatabase, const timestamp_type timestamp);
    void correct(const track_database_type& trackDatabase);

    /// Get the current filter time
    timestamp_type time() const { return m_timestamp; }

    /// Get the current state vector
    const state_type& state() const { return m_state; };

    /// Get the current state covariance
    const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>& covariance() const { return m_covariance; };

    /// Get the current list of frame id's in the estimator
    const std::deque<frame_id>& frames() const { return m_frameIds; }

private:
    /// List of frame ID's that are in the sliding window
    std::deque<frame_id> m_frameIds;

    /// Inertial propagator
    imu_propagator_type& m_imuPropagator;

    /// The maximum number of local frames that are allowed in the sliding window. Local frames are the N new adjacent frames.
    size_t m_numLocalFramesInWindow;

    /// Timestamp of the state
    timestamp_type m_timestamp;
    /// State vector
    state_type m_state;
    /// State covariance
    Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> m_covariance;

    /// Jacobian for visual correction
    Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> m_jacobian;
    /// Residual for visual correction
    Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> m_residual;
    /// Residual covariance
    Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> m_residualCovariance;

    void prepare(const track_database_type& trackDatabase);
    void solve();

    void clone(const track_database_type& trackDatabase);
    void marginalize();
};

template <typename SCALAR>
VisualInertialExtendedKalmanFilter<SCALAR>::VisualInertialExtendedKalmanFilter(imu_propagator_type& imuPropagator,
    const size_t numLocalFramesInWindow, const timestamp_type timestamp, const state_type& initialState,
    const Eigen::Matrix<SCALAR, state_type::initial_state_size, state_type::initial_state_size>& initialCovariance)
    : m_imuPropagator(imuPropagator)
    , m_numLocalFramesInWindow(numLocalFramesInWindow)
    , m_timestamp(timestamp)
    , m_state(initialState)
    , m_covariance(initialCovariance) {}

template <typename SCALAR>
void VisualInertialExtendedKalmanFilter<SCALAR>::predict(const imu_database_type& imuDatabase, const timestamp_type timestamp) {
    constexpr auto imu_state_dim = state_type::imu_state::state_dim;

    const Eigen::Matrix<SCALAR, imu_state_dim, imu_state_dim> imuCovariance = m_covariance.topLeftCorner(imu_state_dim, imu_state_dim);

    // Propagate the state forward
    const Eigen::Matrix<SCALAR, imu_state_dim, 1> nextImuState
        = m_imuPropagator.propagate(m_timestamp, m_state.imu().vec(), imuCovariance, imuDatabase, timestamp);
    m_state.imu() = ImuState<SCALAR>(nextImuState);

    matrix_type propagatedCovariance(m_covariance.rows(), m_covariance.cols());

    propagatedCovariance.topLeftCorner(imu_state_dim, imu_state_dim) = m_imuPropagator.covariance();
    propagatedCovariance.topRightCorner(imu_state_dim, m_covariance.cols() - imu_state_dim)
        = m_imuPropagator.jacobian() * m_covariance.topRightCorner(imu_state_dim, m_covariance.cols() - imu_state_dim);
    propagatedCovariance.bottomLeftCorner(m_covariance.rows() - imu_state_dim, imu_state_dim)
        = propagatedCovariance.topRightCorner(imu_state_dim, m_covariance.cols() - imu_state_dim).transpose();
    propagatedCovariance.bottomRightCorner(m_covariance.rows() - imu_state_dim, m_covariance.cols() - imu_state_dim)
        = m_covariance.bottomRightCorner(m_covariance.rows() - imu_state_dim, m_covariance.cols() - imu_state_dim);

    m_timestamp = timestamp;
    m_covariance = propagatedCovariance;
}

template <typename SCALAR> void VisualInertialExtendedKalmanFilter<SCALAR>::correct(const track_database_type& trackDatabase) {
    if (m_frameIds.size() < m_numLocalFramesInWindow) {
        clone(trackDatabase);
    } else {
        marginalize();
        clone(trackDatabase);
    }
    prepare(trackDatabase);
    solve();
}

template <typename SCALAR> void VisualInertialExtendedKalmanFilter<SCALAR>::prepare(const track_database_type& trackDatabase) {
    std::unordered_map<frame_id, size_t> frameIdToIndexHash;
    for (size_t i = 0; i < m_frameIds.size(); i++) {
        frameIdToIndexHash[m_frameIds[i]] = i;
    }

    const std::unordered_map<track_id, track_type>& tracks = trackDatabase.getTracks();

    // Calculate the size of the residual/jacobian
    size_t numResiduals = 0;
    for (const auto& track : tracks) {
        if (track.second.m_point.m_set) {
            numResiduals += 2 * m_frameIds.size() - 3; // (because we nullify the 3d point which reduces the dimensionality)
        }
    }
    m_residual.resize(numResiduals);
    m_jacobian.resize(numResiduals, state_type::imu_state::state_dim + state_type::calibration_state::state_dim
            + state_type::clone_state::state_dim * m_frameIds.size());

    size_t nextRowBlock = 0;
    for (const auto& track : tracks) {
        if (track.second.m_point.m_set) {
            std::vector<ReprojectionMeasurement<SCALAR> > measurements;
            for (size_t i = 0; i < m_frameIds.size(); i++) {
                const frame_type& frame = trackDatabase.getFrame(m_frameIds[i]);
                auto it = frame.m_points.find(track.second.m_trackId);
                if (it != frame.m_points.end()) {
                    const auto& featurePoint = frame.m_points.find(track.second.m_trackId)->second;
                    const Eigen::Matrix<SCALAR, 2, 1> calibratedPoint(
                        featurePoint.m_calibratedPoint.x(), featurePoint.m_calibratedPoint.y());
                    const ReprojectionMeasurement<SCALAR> measurement(
                        frame.m_ImuToWorldPose.m_quaternion, frame.m_ImuToWorldPose.m_position, calibratedPoint, frame.m_frameId);
                    measurements.push_back(measurement);
                }
            }

            Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> residual;
            Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> jacobian;

            computeTrackJacobian(residual, jacobian, measurements, m_state.calibration().quaternion(), m_state.calibration().position(),
                track.second.m_point.m_point);

            m_residual.template segment(nextRowBlock, residual.rows()) = residual;

            for (size_t i = 0; i < measurements.size(); i++) {
                const size_t& cloneIndex = frameIdToIndexHash[measurements[i].m_cloneId];
                m_jacobian.template block<2, 6>(nextRowBlock + 2 * i, state_type::calibration_state_pos)
                    = jacobian.template block<2, 6>(2 * i, 0);
                m_jacobian.template block<2, 6>(
                    nextRowBlock + 2 * i, state_type::clone_state_pos + state_type::clone_state::state_dim * cloneIndex)
                    = jacobian.template block<2, 6>(
                        2 * i, state_type::calibration_state::state_dim + state_type::clone_state::state_dim * i);
            }

            nextRowBlock += residual.rows();
        }
    }

    if (nextRowBlock != numResiduals) {
        throw std::runtime_error("didn't compute enough residuals");
    }
}

template <typename SCALAR> void VisualInertialExtendedKalmanFilter<SCALAR>::solve() {
    if (m_residual.rows() == 0) {
        LOG(WARNING) << "unable to update state because residual is empty";
        return;
    }

    Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> dx;
    Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> P;

    updateKalmanFilter(dx, P, m_covariance, m_residual, m_jacobian, m_residualCovariance);

    m_state.update(dx);
    m_covariance = P;
}

template <typename SCALAR> void VisualInertialExtendedKalmanFilter<SCALAR>::clone(const track_database_type& trackDatabase) {
    const std::unordered_map<frame_id, frame_type>& frames = trackDatabase.getFrames();

    if (frames.empty()) {
        throw std::runtime_error("there are no frames in the database");
    }

    std::vector<std::pair<double, frame_id> > frameInfo;
    for (auto& frame : frames) {
        frameInfo.push_back(std::make_pair(frame.second.m_timestamp, frame.first));
    }
    const auto maxElement = std::max_element(frameInfo.begin(), frameInfo.end());
    LOG(INFO) << "cloning frame id: " << maxElement->second;

    if (std::find_if(m_frameIds.begin(), m_frameIds.end(),
            [&maxElement](const frame_id& queryFrameId) { return queryFrameId == maxElement->second; })
        != m_frameIds.end()) {
        throw std::runtime_error("frame id already exists");
    }

    m_state.cloneImuState();
    m_covariance = m_state.augmentCovariance(m_covariance);

    m_frameIds.push_back(maxElement->second);
}

template <typename SCALAR> void VisualInertialExtendedKalmanFilter<SCALAR>::marginalize() {
    if (m_frameIds.empty()) {
        throw std::runtime_error("there are no frames to marginalize");
    }

    LOG(INFO) << "removing frame id: " << m_frameIds.front();

    m_state.clones().erase(m_state.clones().begin());
    m_covariance = m_state.marginalizeClone(m_covariance, 0);

    m_frameIds.pop_front();
}
}
