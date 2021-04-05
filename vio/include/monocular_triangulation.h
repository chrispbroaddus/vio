#pragma once

#include "packages/calibration/include/kannala_brandt_distortion_model.h"
#include "packages/feature_tracker/include/stereo_feature_track_index.h"
#include "packages/triangulation/include/triangulation.h"

namespace vio {

/// Performs monocular triangulation for VIO
template <typename TRACK_DATABASE_T> class MonocularTriangulation {
public:
    using track_index_type = TRACK_DATABASE_T;
    using scalar_type = typename track_index_type::point_type;

    /// Construct the monocular triangulation with some calibration transformations.
    ///
    /// Xworld = cameraToWorldTranslation * X + cameraToWorldTranslation
    ///
    MonocularTriangulation() {}
    ~MonocularTriangulation() = default;

    void triangulate(track_index_type& trackIndex) {
        const auto& currentFrame = trackIndex.getFrame(trackIndex.getNewestFrameId());

        const Eigen::Matrix3d currentCameraToWorldRotation = currentFrame.m_ImuToWorldPose.m_quaternion.toRotationMatrix()
            * currentFrame.m_cameraToImuPose.m_quaternion.toRotationMatrix();
        const Eigen::Vector3d currentCameraToWorldTranslation
            = currentFrame.m_ImuToWorldPose.m_quaternion.toRotationMatrix() * currentFrame.m_cameraToImuPose.m_position
            + currentFrame.m_ImuToWorldPose.m_position;

        auto& currentTracks = trackIndex.getTracks();
        for (auto& currentTrack : currentTracks) {
            if (currentTrack.second.m_stereoTrackId < 0) {

                auto currentPointIter = currentFrame.m_points.find(currentTrack.second.m_trackId);
                if (currentPointIter == currentFrame.m_points.end()) {
                    throw std::runtime_error("Cannot find tracked point in current frame");
                }
                const auto& currentPoint = currentPointIter->second.m_calibratedPoint;

                const auto& observingFrames = trackIndex.getObservingFrames(currentTrack.second.m_trackId);

                auto firstFrameIter
                    = std::min_element(observingFrames.begin(), observingFrames.end(), [&](const auto& frameID1, const auto& frameID2) {
                          return trackIndex.getFrame(frameID1).m_timestamp < trackIndex.getFrame(frameID2).m_timestamp;
                      });

                const auto& firstFrame = trackIndex.getFrame(*firstFrameIter);

                const Eigen::Matrix3d firstCameraToWorldRotation = firstFrame.m_ImuToWorldPose.m_quaternion.toRotationMatrix()
                    * firstFrame.m_cameraToImuPose.m_quaternion.toRotationMatrix();
                const Eigen::Vector3d firstCameraToWorldTranslation
                    = firstFrame.m_ImuToWorldPose.m_quaternion.toRotationMatrix() * firstFrame.m_cameraToImuPose.m_position
                    + firstFrame.m_ImuToWorldPose.m_position;

                const Eigen::Matrix3d firstToCurrentRotation = currentCameraToWorldRotation.transpose() * firstCameraToWorldRotation;
                const Eigen::Vector3d firstToCurrentTranslation = currentCameraToWorldRotation.transpose() * firstCameraToWorldTranslation
                    - currentCameraToWorldRotation.transpose() * currentCameraToWorldTranslation;

                auto firstPointIter = firstFrame.m_points.find(currentTrack.second.m_trackId);
                if (firstPointIter == firstFrame.m_points.end()) {
                    throw std::runtime_error("Cannot find tracked point in previous frame");
                }
                const auto& firstPoint = firstPointIter->second.m_calibratedPoint;

                const Eigen::Vector2d m0(currentPoint.x(), currentPoint.y());
                const Eigen::Vector2d m1(firstPoint.x(), firstPoint.y());

                Eigen::Vector3d M;
                if (triangulation::triangulateDirectional(M, m0, firstToCurrentRotation, firstToCurrentTranslation, m1)) {
                    const auto worldPoint = currentCameraToWorldRotation * M + currentCameraToWorldTranslation;
                    currentTrack.second.m_point = feature_tracker::Point3d<double>(worldPoint);
                }
            }
        }
    }
};
}
