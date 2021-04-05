
#include "packages/vio/include/state.h"
#include "gtest/gtest.h"

using namespace vio;

TEST(State, parameterConstructorTest) {
    const ImuState<double> imuState(Eigen::Quaternion<double>(1, 0, 0, 0), Eigen::Matrix<double, 3, 1>(1, 2, 3),
        Eigen::Matrix<double, 3, 1>(4, 5, 6), Eigen::Matrix<double, 3, 1>(7, 8, 9), Eigen::Matrix<double, 3, 1>(10, 11, 12));
    const CalibrationState<double> calibrationState(Eigen::Quaternion<double>(1, 0, 0, 0), Eigen::Matrix<double, 3, 1>(1, 2, 3));

    const State<double> state(imuState, calibrationState);

    EXPECT_EQ(0, state.imu().quaternion().x());
    EXPECT_EQ(0, state.imu().quaternion().x());
    EXPECT_EQ(0, state.imu().quaternion().y());
    EXPECT_EQ(1, state.imu().quaternion().w());

    EXPECT_EQ(0, state.calibration().quaternion().x());
    EXPECT_EQ(0, state.calibration().quaternion().x());
    EXPECT_EQ(0, state.calibration().quaternion().y());
    EXPECT_EQ(1, state.calibration().quaternion().w());
}

TEST(State, vectorConstructorTest) {
    const ImuState<double> imuState(Eigen::Quaternion<double>(1, 0, 0, 0), Eigen::Matrix<double, 3, 1>(1, 2, 3),
        Eigen::Matrix<double, 3, 1>(4, 5, 6), Eigen::Matrix<double, 3, 1>(7, 8, 9), Eigen::Matrix<double, 3, 1>(10, 11, 12));
    const CalibrationState<double> calibrationState(Eigen::Quaternion<double>(1, 0, 0, 0), Eigen::Matrix<double, 3, 1>(1, 2, 3));

    State<double> state(imuState, calibrationState);
    state.cloneImuState();

    const State<double> nextState(state.vec(), state.clones().size());

    EXPECT_EQ(0, nextState.imu().quaternion().x());
    EXPECT_EQ(0, nextState.imu().quaternion().x());
    EXPECT_EQ(0, nextState.imu().quaternion().y());
    EXPECT_EQ(1, nextState.imu().quaternion().w());

    EXPECT_EQ(0, nextState.calibration().quaternion().x());
    EXPECT_EQ(0, nextState.calibration().quaternion().x());
    EXPECT_EQ(0, nextState.calibration().quaternion().y());
    EXPECT_EQ(1, nextState.calibration().quaternion().w());

    EXPECT_EQ(0, nextState.clones()[0].quaternion().x());
    EXPECT_EQ(0, nextState.clones()[0].quaternion().x());
    EXPECT_EQ(0, nextState.clones()[0].quaternion().y());
    EXPECT_EQ(1, nextState.clones()[0].quaternion().w());
}
