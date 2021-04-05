
#include "packages/vio/include/calibration_state.h"
#include "gtest/gtest.h"

using namespace vio;

TEST(CalibrationState, parameterConstructorTest) {
    const Eigen::Quaternion<double> quaternion(1, 0, 0, 0);
    const Eigen::Matrix<double, 3, 1> position(10, 20, 30);
    const CalibrationState<double> state(quaternion, position);

    EXPECT_EQ(quaternion.x(), state.quaternion().x());
    EXPECT_EQ(quaternion.y(), state.quaternion().y());
    EXPECT_EQ(quaternion.z(), state.quaternion().z());
    EXPECT_EQ(quaternion.w(), state.quaternion().w());

    EXPECT_EQ(position(0), state.position()(0));
    EXPECT_EQ(position(1), state.position()(1));
    EXPECT_EQ(position(2), state.position()(2));
}

TEST(CalibrationState, vectorConstructorTest) {
    const Eigen::Quaternion<double> quaternion(1, 0, 0, 0);
    const Eigen::Matrix<double, 3, 1> position(10, 20, 30);
    const CalibrationState<double> state(quaternion, position);

    const CalibrationState<double> nextState(state.vec());

    EXPECT_EQ(quaternion.x(), nextState.quaternion().x());
    EXPECT_EQ(quaternion.y(), nextState.quaternion().y());
    EXPECT_EQ(quaternion.z(), nextState.quaternion().z());
    EXPECT_EQ(quaternion.w(), nextState.quaternion().w());

    EXPECT_EQ(position(0), nextState.position()(0));
    EXPECT_EQ(position(1), nextState.position()(1));
    EXPECT_EQ(position(2), nextState.position()(2));
}
