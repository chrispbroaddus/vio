#include "packages/feature_tracker/include/stereo_feature_track_index.h"
#include "packages/vio/include/stereo_triangulation.h"

#include "Eigen/Eigen"
#include "packages/vio/include/stereo_triangulation.h"
#include "gtest/gtest.h"

using namespace feature_tracker;

TEST(StereoTriangulationTest, oneGoodTrack) {

    feature_tracker::details::stereo_feature_track_index_type leftDatabase(3);
    feature_tracker::details::stereo_feature_track_index_type rightDatabase(3);

    Eigen::Matrix3d leftK;
    Eigen::Vector4d leftDistortion;
    leftK << 771.4914, 0, 1244.208, 0, 770.1611, 1077.472, 0, 0, 1;
    leftDistortion << 0.01117972, 0.04504434, -0.05763411, 0.02156141;
    calibration::KannalaBrandtRadialDistortionModel4<double> leftKb4Model(leftK, leftDistortion, 10, 0);
    Eigen::Matrix3d rightK;
    Eigen::Vector4d rightDistortion;
    rightK << 761.5459, 0, 1257.009, 0, 760.4103, 1050.432, 0, 0, 1;
    rightDistortion << 0.08616139, -0.08189862, 0.01098505, -0.0004719968;
    calibration::KannalaBrandtRadialDistortionModel4<double> rightKb4Model(rightK, rightDistortion, 10, 0);

    Eigen::Matrix3d leftToImuRotation;
    double theta = 15.0 * M_PI / 180.0;
    leftToImuRotation << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
    Eigen::Vector3d leftToImuTranslation(0.1, 0.0, 0.0);

    Eigen::Matrix3d imuToWorldRotation;
    theta = 30.0 * M_PI / 180.0;
    imuToWorldRotation << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
    Eigen::Vector3d imuToWorldTranslation(0., 0.1, 0.1);

    Eigen::Matrix3d leftToWorldRotation = imuToWorldRotation * leftToImuRotation;
    Eigen::Vector3d leftToWorldTranslation = imuToWorldRotation * leftToImuTranslation + imuToWorldTranslation;

    Eigen::Matrix3d rightToLeftRotation;
    theta = 10.0 * M_PI / 180.0;
    rightToLeftRotation << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
    Eigen::Vector3d rightToLeftTranslation(0.1, 0.05, 0.1);

    Eigen::Vector3d point3dLeftCamera(0.2, -0.1, 1);
    Eigen::Vector3d point3dWorld = leftToWorldRotation * point3dLeftCamera + leftToWorldTranslation;

    Eigen::Vector2d point2dL = leftKb4Model.project(point3dLeftCamera.hnormalized().homogeneous());
    Eigen::Vector2d point2dR = rightKb4Model.project(
        (rightToLeftRotation.transpose() * point3dLeftCamera - rightToLeftRotation.transpose() * rightToLeftTranslation)
            .hnormalized()
            .homogeneous());

    Eigen::Vector2d pointCalibratedL = leftKb4Model.unproject(point2dL).hnormalized();
    Eigen::Vector2d pointCalibratedR = rightKb4Model.unproject(point2dR).hnormalized();

    std::vector<feature_detectors::FeaturePoint> featurePoints;
    feature_detectors::FeaturePoint pixelPoint;
    feature_detectors::FeaturePoint calibratedPoint;
    pixelPoint.set_x(point2dL(0));
    pixelPoint.set_y(point2dL(1));
    calibratedPoint.set_x(pointCalibratedL(0));
    calibratedPoint.set_y(pointCalibratedL(1));
    featurePoints.push_back(pixelPoint);
    std::vector<std::pair<int, int> > matches;
    feature_tracker::Pose<double> leftToImuPose(Eigen::Quaternion<double>(leftToImuRotation), leftToImuTranslation);
    feature_tracker::Pose<double> imuToWorldPose(Eigen::Quaternion<double>(imuToWorldRotation), imuToWorldTranslation);

    leftDatabase.addMatches(featurePoints, matches);
    leftDatabase.getFrame(leftDatabase.getNewestFrameId()).m_cameraToImuPose = leftToImuPose;
    leftDatabase.getFrame(leftDatabase.getNewestFrameId()).m_ImuToWorldPose = imuToWorldPose;
    leftDatabase.getFrame(leftDatabase.getNewestFrameId()).m_points[0].m_calibratedPoint = calibratedPoint;
    auto& leftTracks = leftDatabase.getTracks();
    EXPECT_EQ(1, leftTracks.size());
    for (auto& track : leftTracks) {
        track.second.m_stereoTrackId = 0;
    }

    featurePoints.clear();
    pixelPoint.set_x(point2dR(0));
    pixelPoint.set_y(point2dR(1));
    calibratedPoint.set_x(pointCalibratedR(0));
    calibratedPoint.set_y(pointCalibratedR(1));
    featurePoints.push_back(pixelPoint);
    feature_tracker::Pose<double> rightToImuPose(Eigen::Quaternion<double>(leftToImuRotation * rightToLeftRotation),
        leftToImuRotation * rightToLeftTranslation + leftToImuTranslation);
    rightDatabase.addMatches(featurePoints, matches);
    rightDatabase.getFrame(rightDatabase.getNewestFrameId()).m_cameraToImuPose = rightToImuPose;
    rightDatabase.getFrame(rightDatabase.getNewestFrameId()).m_ImuToWorldPose = imuToWorldPose;
    rightDatabase.getFrame(rightDatabase.getNewestFrameId()).m_points[0].m_calibratedPoint = calibratedPoint;

    vio::StereoTriangulation<feature_tracker::details::stereo_feature_track_index_type> stereoTriangulation(
        rightToLeftRotation, rightToLeftTranslation);
    stereoTriangulation.triangulate(leftDatabase, rightDatabase);

    const auto& tracks = leftDatabase.getTracks();
    for (const auto& track : tracks) {
        EXPECT_EQ(true, track.second.m_point.m_set);
        EXPECT_NEAR(point3dWorld(0), track.second.m_point.m_point(0), 1.0e-6);
        EXPECT_NEAR(point3dWorld(1), track.second.m_point.m_point(1), 1.0e-6);
        EXPECT_NEAR(point3dWorld(2), track.second.m_point.m_point(2), 1.0e-6);
    }
}
