
#include "packages/vio/include/visual_inertial_extended_kalman_filter.h"
#include "gtest/gtest.h"

using namespace vio;

using vio_ekf_type = VisualInertialExtendedKalmanFilter<double>;

TEST(VisualInertialExtendedKalmanFilter, constructor) {
    const imu_propagator_type::noise_type gyroWhiteNoise(0, 0, 0);
    const imu_propagator_type::noise_type gyroBiasNoise(0, 0, 0);
    const imu_propagator_type::noise_type accelWhiteNoise(0, 0, 0);
    const imu_propagator_type::noise_type accelBiasNoise(0, 0, 0);
    const imu_propagator::details::gravity_type<double> gravity(0, 0, 0);
    imu_propagator_type propagator(gyroWhiteNoise, gyroBiasNoise, accelWhiteNoise, accelBiasNoise, gravity);

    constexpr auto imu_state_size = vio_ekf_type::state_type::initial_state_size;
    constexpr double timestamp = 0;
    vio_ekf_type::state_type state;
    Eigen::Matrix<double, imu_state_size, imu_state_size> covariance = Eigen::Matrix<double, imu_state_size, imu_state_size>::Zero();

    constexpr size_t numLocalFramesInWindow = 2;

    vio_ekf_type ekf(propagator, numLocalFramesInWindow, timestamp, state, covariance);
}

TEST(VisualInertialExtendedKalmanFilter, predict) {
    const imu_propagator_type::noise_type gyroWhiteNoise(0, 0, 0);
    const imu_propagator_type::noise_type gyroBiasNoise(0, 0, 0);
    const imu_propagator_type::noise_type accelWhiteNoise(0.1, 0, 0);
    const imu_propagator_type::noise_type accelBiasNoise(0, 0, 0);
    const imu_propagator::details::gravity_type<double> gravity(0, 0, 0);
    imu_propagator_type propagator(gyroWhiteNoise, gyroBiasNoise, accelWhiteNoise, accelBiasNoise, gravity);

    constexpr auto imu_state_size = vio_ekf_type::state_type::initial_state_size;
    constexpr double timestamp = 10;
    vio_ekf_type::state_type state;
    Eigen::Matrix<double, imu_state_size, imu_state_size> covariance = Eigen::Matrix<double, imu_state_size, imu_state_size>::Zero();

    constexpr size_t numLocalFramesInWindow = 2;

    vio_ekf_type ekf(propagator, numLocalFramesInWindow, timestamp, state, covariance);

    imu_database_type imuDatabase(10);

    const imu_sample_type::gyro_type gyro1(0, 0, 0);
    const imu_sample_type::accel_type accel1(1, 0, 0);
    const imu_sample_type sample1(10, gyro1, accel1);

    const imu_sample_type::gyro_type gyro2(0, 0, 0);
    const imu_sample_type::accel_type accel2(1, 0, 0);
    const imu_sample_type sample2(11, gyro2, accel2);

    const imu_sample_type::gyro_type gyro3(0, 0, 0);
    const imu_sample_type::accel_type accel3(1, 0, 0);
    const imu_sample_type sample3(12, gyro3, accel3);

    imuDatabase.addImuSample(sample1);
    imuDatabase.addImuSample(sample2);
    imuDatabase.addImuSample(sample3);

    ekf.predict(imuDatabase, 11);

    EXPECT_DOUBLE_EQ(0, ekf.state().imu().quaternion().x());
    EXPECT_DOUBLE_EQ(0, ekf.state().imu().quaternion().y());
    EXPECT_DOUBLE_EQ(0, ekf.state().imu().quaternion().z());

    EXPECT_DOUBLE_EQ(0.5, ekf.state().imu().position()[0]);
    EXPECT_DOUBLE_EQ(0, ekf.state().imu().position()[1]);
    EXPECT_DOUBLE_EQ(0, ekf.state().imu().position()[2]);

    // Ensure that the covariance for the (vx,px) components are non-zero
    EXPECT_TRUE(ekf.covariance()(6, 6) > 0);
    EXPECT_TRUE(ekf.covariance()(12, 12) > 0);
    EXPECT_TRUE(ekf.covariance()(6, 12) > 0);
    EXPECT_TRUE(ekf.covariance()(12, 6) > 0);
}

TEST(VisualInertialExtendedKalmanFilter, correct) {
    const imu_propagator_type::noise_type gyroWhiteNoise(0, 0, 0);
    const imu_propagator_type::noise_type gyroBiasNoise(0, 0, 0);
    const imu_propagator_type::noise_type accelWhiteNoise(0, 0, 0);
    const imu_propagator_type::noise_type accelBiasNoise(0, 0, 0);
    const imu_propagator::details::gravity_type<double> gravity(0, 0, 0);
    imu_propagator_type propagator(gyroWhiteNoise, gyroBiasNoise, accelWhiteNoise, accelBiasNoise, gravity);

    constexpr auto imu_state_size = vio_ekf_type::state_type::initial_state_size;
    constexpr double timestamp = 10;
    vio_ekf_type::state_type state;
    Eigen::Matrix<double, imu_state_size, imu_state_size> covariance = Eigen::Matrix<double, imu_state_size, imu_state_size>::Zero();

    constexpr size_t numLocalFramesInWindow = 2;

    vio_ekf_type ekf(propagator, numLocalFramesInWindow, timestamp, state, covariance);
    EXPECT_EQ(21, ekf.covariance().rows());
    EXPECT_EQ(21, ekf.covariance().cols());

    track_database_type trackDatabase;

    ASSERT_THROW(ekf.correct(trackDatabase), std::runtime_error);
    EXPECT_EQ(0, ekf.state().clones().size());

    trackDatabase.insertFrame(track_database_type::frame_type(100, 1));
    ekf.correct(trackDatabase);
    EXPECT_EQ(1, ekf.frames().size());
    EXPECT_EQ(1, ekf.state().clones().size());
    EXPECT_EQ(27, ekf.covariance().rows());
    EXPECT_EQ(27, ekf.covariance().cols());

    trackDatabase.insertFrame(track_database_type::frame_type(101, 2));
    ekf.correct(trackDatabase);
    EXPECT_EQ(2, ekf.frames().size());
    EXPECT_EQ(2, ekf.state().clones().size());
    EXPECT_EQ(33, ekf.covariance().rows());
    EXPECT_EQ(33, ekf.covariance().cols());

    trackDatabase.insertFrame(track_database_type::frame_type(102, 2));
    ekf.correct(trackDatabase);
    EXPECT_EQ(2, ekf.frames().size());
    EXPECT_EQ(2, ekf.state().clones().size());
    EXPECT_EQ(33, ekf.covariance().rows());
    EXPECT_EQ(33, ekf.covariance().cols());
}
