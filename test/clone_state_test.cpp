
#include "packages/vio/include/clone_state.h"
#include "gtest/gtest.h"

using namespace vio;

TEST(CloneState, parameterConstructorTest) {
    const Eigen::Quaternion<double> quaternion(1, 0, 0, 0);
    const Eigen::Matrix<double, 3, 1> position(10, 20, 30);
    const CloneState<double> state(quaternion, position);

    EXPECT_EQ(quaternion.x(), state.quaternion().x());
    EXPECT_EQ(quaternion.y(), state.quaternion().y());
    EXPECT_EQ(quaternion.z(), state.quaternion().z());
    EXPECT_EQ(quaternion.w(), state.quaternion().w());

    EXPECT_EQ(position(0), state.position()(0));
    EXPECT_EQ(position(1), state.position()(1));
    EXPECT_EQ(position(2), state.position()(2));
}

TEST(CloneState, vectorConstructorTest) {
    const Eigen::Quaternion<double> quaternion(1, 0, 0, 0);
    const Eigen::Matrix<double, 3, 1> position(10, 20, 30);
    const CloneState<double> state(quaternion, position);

    const CloneState<double> nextState(state.vec());

    EXPECT_EQ(quaternion.x(), nextState.quaternion().x());
    EXPECT_EQ(quaternion.y(), nextState.quaternion().y());
    EXPECT_EQ(quaternion.z(), nextState.quaternion().z());
    EXPECT_EQ(quaternion.w(), nextState.quaternion().w());

    EXPECT_EQ(position(0), nextState.position()(0));
    EXPECT_EQ(position(1), nextState.position()(1));
    EXPECT_EQ(position(2), nextState.position()(2));
}

TEST(CloneState, update) {
    CloneState<double> cloneState;

    Eigen::Matrix<double, 6, 1> dx;
    dx(0) = 0.01;
    dx(1) = -0.02;
    dx(2) = 0.03;
    dx(3) = 1;
    dx(4) = 2;
    dx(5) = 3;

    cloneState.update(dx);

    EXPECT_NEAR(0.00499971, cloneState.quaternion().x(), 1e-6);
    EXPECT_NEAR(-0.00999942, cloneState.quaternion().y(), 1e-6);
    EXPECT_NEAR(0.0149991, cloneState.quaternion().z(), 1e-6);
    EXPECT_NEAR(1, cloneState.position()(0), 1e-6);
    EXPECT_NEAR(2, cloneState.position()(1), 1e-6);
    EXPECT_NEAR(3, cloneState.position()(2), 1e-6);
}
