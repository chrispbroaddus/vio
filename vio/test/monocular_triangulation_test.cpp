#include "packages/vio/include/monocular_triangulation.h"
#include "packages/calibration/include/linear_camera_model.h"
#include "packages/feature_tracker/include/stereo_feature_track_index.h"

#include "Eigen/Eigen"
#include "gtest/gtest.h"

using namespace feature_tracker;

TEST(MonocularTriangulationTest, oneGoodTrack) {

    feature_tracker::details::stereo_feature_track_index_type database(3);

    Eigen::Matrix3d K;
    Eigen::Vector4d distortionCoef;
    K << 771.4914, 0, 1244.208, 0, 770.1611, 1077.472, 0, 0, 1;
    distortionCoef << 0.01117972, 0.04504434, -0.05763411, 0.02156141;
    calibration::KannalaBrandtRadialDistortionModel4<double> kb4Model(K, distortionCoef, 10, 0);

    Eigen::Matrix3d currentToImuRotation;
    double theta = 15.0 * M_PI / 180.0;
    currentToImuRotation << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
    Eigen::Vector3d currentToImuTranslation(0.1, 0.0, 0.0);

    Eigen::Matrix3d currentImuToWorldRotation;
    theta = 30.0 * M_PI / 180.0;
    currentImuToWorldRotation << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
    Eigen::Vector3d currentImuToWorldTranslation(0., 0.1, 0.1);

    Eigen::Matrix3d currentToWorldRotation = currentImuToWorldRotation * currentToImuRotation;
    Eigen::Vector3d currentToWorldTranslation = currentImuToWorldRotation * currentToImuTranslation + currentImuToWorldTranslation;

    Eigen::Matrix3d firstToImuRotation;
    theta = 14.0 * M_PI / 180.0;
    firstToImuRotation << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
    Eigen::Vector3d firstToImuTranslation(0.15, 0.0, 0.0);

    Eigen::Matrix3d firstImuToWorldRotation;
    theta = 40.0 * M_PI / 180.0;
    firstImuToWorldRotation << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
    Eigen::Vector3d firstImuToWorldTranslation(0., 0.2, 0.1);

    Eigen::Matrix3d firstToWorldRotation = firstImuToWorldRotation * firstToImuRotation;
    Eigen::Vector3d firstToWorldTranslation = firstImuToWorldRotation * firstToImuTranslation + firstImuToWorldTranslation;

    const Eigen::Matrix3d firstToCurrentRotation = currentToWorldRotation.transpose() * firstToWorldRotation;
    const Eigen::Vector3d firstToCurrentTranslation
        = currentToWorldRotation.transpose() * firstToWorldTranslation - currentToWorldRotation.transpose() * currentToWorldTranslation;

    Eigen::Vector3d point3dCurrentCamera(0.2, -0.1, 1);
    Eigen::Vector3d point3dWorld = currentToWorldRotation * point3dCurrentCamera + currentToWorldTranslation;

    Eigen::Vector2d point2dF1 = kb4Model.project(
        (firstToCurrentRotation.transpose() * point3dCurrentCamera - firstToCurrentRotation.transpose() * firstToCurrentTranslation)
            .hnormalized()
            .homogeneous());
    Eigen::Vector2d point2dF2 = kb4Model.project(point3dCurrentCamera.hnormalized().homogeneous());

    Eigen::Vector2d pointCalibratedF1 = kb4Model.unproject(point2dF1).hnormalized();
    Eigen::Vector2d pointCalibratedF2 = kb4Model.unproject(point2dF2).hnormalized();

    std::vector<feature_detectors::FeaturePoint> featurePoints;
    feature_detectors::FeaturePoint pixelPoint;
    feature_detectors::FeaturePoint calibratedPoint;
    pixelPoint.set_x(point2dF1(0));
    pixelPoint.set_y(point2dF1(1));
    calibratedPoint.set_x(pointCalibratedF1(0));
    calibratedPoint.set_y(pointCalibratedF1(1));
    featurePoints.push_back(pixelPoint);
    std::vector<std::pair<int, int> > matches;
    feature_tracker::Pose<double> firstToImuPose(Eigen::Quaternion<double>(firstToImuRotation), firstToImuTranslation);
    feature_tracker::Pose<double> firstImuToWorldPose(Eigen::Quaternion<double>(firstImuToWorldRotation), firstImuToWorldTranslation);

    database.addMatches(featurePoints, matches);
    database.getFrame(database.getNewestFrameId()).m_cameraToImuPose = firstToImuPose;
    database.getFrame(database.getNewestFrameId()).m_ImuToWorldPose = firstImuToWorldPose;
    database.getFrame(database.getNewestFrameId()).m_timestamp = 0.1;
    database.getFrame(database.getNewestFrameId()).m_points[0].m_calibratedPoint = calibratedPoint;

    matches.push_back(std::make_pair<int, int>(0, 0));
    featurePoints.clear();
    pixelPoint.set_x(point2dF2(0));
    pixelPoint.set_y(point2dF2(1));
    calibratedPoint.set_x(pointCalibratedF2(0));
    calibratedPoint.set_y(pointCalibratedF2(1));
    featurePoints.push_back(pixelPoint);
    database.addMatches(featurePoints, matches);
    feature_tracker::Pose<double> currentToImuPose(Eigen::Quaternion<double>(currentToImuRotation), currentToImuTranslation);
    feature_tracker::Pose<double> currentImuToWorldPose(Eigen::Quaternion<double>(currentImuToWorldRotation), currentImuToWorldTranslation);
    database.getFrame(database.getNewestFrameId()).m_cameraToImuPose = currentToImuPose;
    database.getFrame(database.getNewestFrameId()).m_ImuToWorldPose = currentImuToWorldPose;
    database.getFrame(database.getNewestFrameId()).m_timestamp = 0.2;
    database.getFrame(database.getNewestFrameId()).m_points[0].m_calibratedPoint = calibratedPoint;

    vio::MonocularTriangulation<feature_tracker::details::stereo_feature_track_index_type> monocularTriangulation;
    monocularTriangulation.triangulate(database);

    const auto& tracks = database.getTracks();
    EXPECT_EQ(1, tracks.size());
    for (const auto& track : tracks) {
        EXPECT_EQ(true, track.second.m_point.m_set);
        EXPECT_NEAR(point3dWorld(0), track.second.m_point.m_point(0), 1.0e-6);
        EXPECT_NEAR(point3dWorld(1), track.second.m_point.m_point(1), 1.0e-6);
        EXPECT_NEAR(point3dWorld(2), track.second.m_point.m_point(2), 1.0e-6);
    }
}
