
#include "packages/vio/include/imu_state.h"
#include "gtest/gtest.h"

using namespace vio;

TEST(ImuState, parameterConstructorTest) {
    const Eigen::Quaternion<double> quaternion(1, 0, 0, 0);
    const Eigen::Matrix<double, 3, 1> gyroBias(0, 1, 2);
    const Eigen::Matrix<double, 3, 1> velocity(100, 101, 102);
    const Eigen::Matrix<double, 3, 1> accelBias(1, 2, 3);
    const Eigen::Matrix<double, 3, 1> position(10, 20, 30);
    const ImuState<double> state(quaternion, gyroBias, velocity, accelBias, position);

    EXPECT_EQ(quaternion.x(), state.quaternion().x());
    EXPECT_EQ(quaternion.y(), state.quaternion().y());
    EXPECT_EQ(quaternion.z(), state.quaternion().z());
    EXPECT_EQ(quaternion.w(), state.quaternion().w());

    EXPECT_EQ(gyroBias(0), state.gyroBias()(0));
    EXPECT_EQ(gyroBias(1), state.gyroBias()(1));
    EXPECT_EQ(gyroBias(2), state.gyroBias()(2));

    EXPECT_EQ(velocity(0), state.velocity()(0));
    EXPECT_EQ(velocity(1), state.velocity()(1));
    EXPECT_EQ(velocity(2), state.velocity()(2));

    EXPECT_EQ(accelBias(0), state.accelBias()(0));
    EXPECT_EQ(accelBias(1), state.accelBias()(1));
    EXPECT_EQ(accelBias(2), state.accelBias()(2));

    EXPECT_EQ(position(0), state.position()(0));
    EXPECT_EQ(position(1), state.position()(1));
    EXPECT_EQ(position(2), state.position()(2));
}

TEST(ImuState, vectorConstructorTest) {
    const Eigen::Quaternion<double> quaternion(1, 0, 0, 0);
    const Eigen::Matrix<double, 3, 1> gyroBias(0, 1, 2);
    const Eigen::Matrix<double, 3, 1> velocity(100, 101, 102);
    const Eigen::Matrix<double, 3, 1> accelBias(1, 2, 3);
    const Eigen::Matrix<double, 3, 1> position(10, 20, 30);
    const ImuState<double> state(quaternion, gyroBias, velocity, accelBias, position);

    const ImuState<double> nextState(state.vec());

    EXPECT_EQ(quaternion.x(), nextState.quaternion().x());
    EXPECT_EQ(quaternion.y(), nextState.quaternion().y());
    EXPECT_EQ(quaternion.z(), nextState.quaternion().z());
    EXPECT_EQ(quaternion.w(), nextState.quaternion().w());

    EXPECT_EQ(gyroBias(0), nextState.gyroBias()(0));
    EXPECT_EQ(gyroBias(1), nextState.gyroBias()(1));
    EXPECT_EQ(gyroBias(2), nextState.gyroBias()(2));

    EXPECT_EQ(velocity(0), nextState.velocity()(0));
    EXPECT_EQ(velocity(1), nextState.velocity()(1));
    EXPECT_EQ(velocity(2), nextState.velocity()(2));

    EXPECT_EQ(accelBias(0), nextState.accelBias()(0));
    EXPECT_EQ(accelBias(1), nextState.accelBias()(1));
    EXPECT_EQ(accelBias(2), nextState.accelBias()(2));

    EXPECT_EQ(position(0), nextState.position()(0));
    EXPECT_EQ(position(1), nextState.position()(1));
    EXPECT_EQ(position(2), nextState.position()(2));
}

TEST(ImuState, update) {
    ImuState<double> imuState;

    Eigen::Matrix<double, 15, 1> dx;
    dx(0) = .01;
    dx(1) = -.02;
    dx(2) = .03;
    dx(3) = 0;
    dx(4) = 0;
    dx(5) = 0;
    dx(6) = 0;
    dx(7) = 0;
    dx(8) = 0;
    dx(9) = 0;
    dx(10) = 0;
    dx(11) = 0;
    dx(12) = 0;
    dx(13) = 0;
    dx(14) = 0;

    imuState.update(dx);

    EXPECT_NEAR(dx(0) / 2, imuState.quaternion().x(), 1e-6);
    EXPECT_NEAR(dx(1) / 2, imuState.quaternion().y(), 1e-6);
    EXPECT_NEAR(dx(2) / 2, imuState.quaternion().z(), 1e-6);
}
