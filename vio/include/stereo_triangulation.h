#pragma once

#include "packages/calibration/include/kannala_brandt_distortion_model.h"
#include "packages/feature_tracker/include/stereo_feature_track_index.h"
#include "packages/triangulation/include/triangulation.h"

namespace vio {

/// Performs stereo triangulation for VIO
template <typename TRACK_DATABASE_T> class StereoTriangulation {
public:
    using track_index_type = TRACK_DATABASE_T;
    using scalar_type = typename track_index_type::point_type;

    /// Construct the stereo triangulation with some calibration transformations.
    ///
    /// Xworld = leftCameraToWorldTranslation * Xleft + leftCameraToWorldTranslation
    /// Xleft = rightCameraToLeftCameraRotation * Xright + rightCameraToLeftCameraTranslation
    ///
    StereoTriangulation(const Eigen::Matrix3d& rightCameraToLeftCameraRotation, const Eigen::Vector3d& rightCameraToLeftCameraTranslation)
        : m_rightCameraToLeftCameraRotation(rightCameraToLeftCameraRotation)
        , m_rightCameraToLeftCameraTranslation(rightCameraToLeftCameraTranslation) {}
    ~StereoTriangulation() = default;

    /// Triangulate the stereo matches. The 3D triangulated point is stored in the left track index in the world
    /// coordinate system using the rigid transformation (leftCameraToWorldRotation, leftCameraToWorldTranslation).
    /// \param leftTrackIndex Left track index which stores the triangulated point
    /// \param rightTrackIndex Right track index which is unmodified
    void triangulate(track_index_type& leftTrackIndex, const track_index_type& rightTrackIndex) {
        const auto& leftFrame = leftTrackIndex.getFrame(leftTrackIndex.getNewestFrameId());
        const auto& rightFrame = rightTrackIndex.getFrame(rightTrackIndex.getNewestFrameId());

        const Eigen::Matrix3d leftCameraToWorldRotation
            = leftFrame.m_ImuToWorldPose.m_quaternion.toRotationMatrix() * leftFrame.m_cameraToImuPose.m_quaternion.toRotationMatrix();
        const Eigen::Vector3d leftCameraToWorldTranslation
            = leftFrame.m_ImuToWorldPose.m_quaternion.toRotationMatrix() * leftFrame.m_cameraToImuPose.m_position
            + leftFrame.m_ImuToWorldPose.m_position;

        auto& leftTracks = leftTrackIndex.getTracks();
        for (auto& leftTrack : leftTracks) {
            if (leftTrack.second.m_stereoTrackId >= 0) {
                const auto& rightTrack = rightTrackIndex.getTrack(leftTrack.second.m_stereoTrackId);

                auto leftPointIter = leftFrame.m_points.find(leftTrack.second.m_trackId);
                if (leftPointIter == leftFrame.m_points.end()) {
                    throw std::runtime_error("Cannot find tracked point in left frame");
                }
                const auto& leftPoint = leftPointIter->second.m_calibratedPoint;
                auto rightPointIter = rightFrame.m_points.find(rightTrack.m_trackId);
                if (rightPointIter == rightFrame.m_points.end()) {
                    throw std::runtime_error("Cannot find tracked point in right frame");
                }
                const auto& rightPoint = rightPointIter->second.m_calibratedPoint;

                const Eigen::Vector2d m0(leftPoint.x(), leftPoint.y());
                const Eigen::Vector2d m1(rightPoint.x(), rightPoint.y());

                Eigen::Vector3d M;
                if (triangulation::triangulateDirectional(
                        M, m0, m_rightCameraToLeftCameraRotation, m_rightCameraToLeftCameraTranslation, m1)) {
                    const auto worldPoint = leftCameraToWorldRotation * M + leftCameraToWorldTranslation;
                    leftTrack.second.m_point = feature_tracker::Point3d<double>(worldPoint);
                }
            }
        }
    }

private:
    const Eigen::Matrix3d m_rightCameraToLeftCameraRotation;
    const Eigen::Vector3d m_rightCameraToLeftCameraTranslation;
};
}
