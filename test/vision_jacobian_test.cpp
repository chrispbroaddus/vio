
#include "packages/vio/include/vision_jacobian.h"
#include "gtest/gtest.h"

using namespace vio;

/// y = R*x
template <typename T> inline Eigen::Matrix<T, 3, 1> applyRotation(const Eigen::Matrix<T, 3, 3>& rotation, const Eigen::Matrix<T, 3, 1>& x) {
    return rotation * x;
}

/// y = R'*x
template <typename T>
inline Eigen::Matrix<T, 3, 1> applyRotationTranspose(const Eigen::Matrix<T, 3, 3>& rotation, const Eigen::Matrix<T, 3, 1>& x) {
    return rotation.transpose() * x;
}

/// Test that y = R*(I+ssm(w)) ~ R*expm(w) which is computed using Eigen
TEST(skewSymmetric, smallAngleAssumption) {
    const Eigen::Matrix<double, 3, 1> axis(.01, -.01, .02);
    const Eigen::Matrix<double, 3, 3> rotation
        = Eigen::Quaternion<double>(Eigen::AngleAxis<double>(axis.norm(), axis.normalized())).toRotationMatrix();
    const Eigen::Matrix<double, 3, 3> approxRotation
        = Eigen::Matrix<double, 3, 3>::Identity() + imu_propagator::utils::skewSymmetricMatrix(axis);

    EXPECT_NEAR(rotation(0, 0), approxRotation(0, 0), 1e-3);
    EXPECT_NEAR(rotation(0, 1), approxRotation(0, 1), 1e-3);
    EXPECT_NEAR(rotation(0, 2), approxRotation(0, 2), 1e-3);
    EXPECT_NEAR(rotation(1, 0), approxRotation(1, 0), 1e-3);
    EXPECT_NEAR(rotation(1, 1), approxRotation(1, 1), 1e-3);
    EXPECT_NEAR(rotation(1, 2), approxRotation(1, 2), 1e-3);
    EXPECT_NEAR(rotation(2, 0), approxRotation(2, 0), 1e-3);
    EXPECT_NEAR(rotation(2, 1), approxRotation(2, 1), 1e-3);
    EXPECT_NEAR(rotation(2, 2), approxRotation(2, 2), 1e-3);
}

TEST(computeJacobianRotation, verifyJacobian) {
    const Eigen::Matrix<double, 3, 1> axis(4, -1, .3);

    const Eigen::Quaternion<double> quaternion(Eigen::AngleAxis<double>(axis.norm(), axis.normalized()));
    const Eigen::Vector3d worldPoint(10, -50, 100);
    const Eigen::Matrix<double, 3, 3> jacobian = computeJacobianRotation(quaternion, worldPoint);

    const Eigen::Matrix<double, 3, 1> w(.01, -.01, .02);
    const Eigen::Quaternion<double> deltaQuaternion(Eigen::AngleAxis<double>(w.norm(), w.normalized()));
    const Eigen::Matrix<double, 3, 1> deltaRotatedWorldPoint
        = applyRotation((quaternion.toRotationMatrix() * deltaQuaternion.toRotationMatrix()).eval(), worldPoint);
    const Eigen::Matrix<double, 3, 1> rotatedWorldPoint = applyRotation(quaternion.toRotationMatrix(), worldPoint);
    const Eigen::Matrix<double, 3, 1> deltaRotatedWorldPointApprox = rotatedWorldPoint + jacobian * deltaQuaternion.vec() * 2;

    EXPECT_NEAR(deltaRotatedWorldPoint(0), deltaRotatedWorldPointApprox(0), 1e-2);
    EXPECT_NEAR(deltaRotatedWorldPoint(1), deltaRotatedWorldPointApprox(1), 1e-2);
    EXPECT_NEAR(deltaRotatedWorldPoint(2), deltaRotatedWorldPointApprox(2), 1e-2);
}

TEST(computeJacobianRotationTranspose, verifyJacobian) {
    const Eigen::Matrix<double, 3, 1> axis(4, -1, .3);

    const Eigen::Quaternion<double> quaternion(Eigen::AngleAxis<double>(axis.norm(), axis.normalized()));
    const Eigen::Vector3d worldPoint(10, -50, 100);
    const Eigen::Matrix<double, 3, 3> jacobian = computeJacobianRotationTranspose(quaternion, worldPoint);

    const Eigen::Matrix<double, 3, 1> w(.01, -.01, .02);
    const Eigen::Quaternion<double> deltaQuaternion(Eigen::AngleAxis<double>(w.norm(), w.normalized()));
    const Eigen::Matrix<double, 3, 1> deltaRotatedWorldPoint
        = applyRotationTranspose((quaternion.toRotationMatrix() * deltaQuaternion.toRotationMatrix()).eval(), worldPoint);
    const Eigen::Matrix<double, 3, 1> rotatedWorldPoint = applyRotationTranspose(quaternion.toRotationMatrix(), worldPoint);
    const Eigen::Matrix<double, 3, 1> deltaRotatedWorldPointApprox = rotatedWorldPoint + jacobian * deltaQuaternion.vec() * 2;

    EXPECT_NEAR(deltaRotatedWorldPoint(0), deltaRotatedWorldPointApprox(0), 5e-2);
    EXPECT_NEAR(deltaRotatedWorldPoint(1), deltaRotatedWorldPointApprox(1), 5e-2);
    EXPECT_NEAR(deltaRotatedWorldPoint(2), deltaRotatedWorldPointApprox(2), 5e-2);
}

TEST(projectPoint, centerPoint) {
    const Eigen::Quaternion<double> imuToWorldRotation(1, 0, 0, 0);
    const Eigen::Matrix<double, 3, 1> imuPositionInWorld(0, 0, 0);
    const Eigen::Quaternion<double> cameraToImuRotation(1, 0, 0, 0);
    const Eigen::Matrix<double, 3, 1> cameraPositionInImu(0, 0, 0);
    const Eigen::Matrix<double, 3, 1> worldPoint(0, 0, 100);
    const Eigen::Matrix<double, 3, 1> cameraPoint
        = mapWorldPointToCamera(imuToWorldRotation, imuPositionInWorld, cameraToImuRotation, cameraPositionInImu, worldPoint);
    EXPECT_DOUBLE_EQ(0, cameraPoint(0));
    EXPECT_DOUBLE_EQ(0, cameraPoint(1));
    EXPECT_DOUBLE_EQ(100, cameraPoint(2));
}

TEST(jacobianPerspectiveProjection, finiteDifference) {
    const double epsilon = 1e-8;
    const Eigen::Matrix<double, 3, 1> X(1, 2, 3);
    const Eigen::Matrix<double, 2, 3> jacobian = jacobianPerspectiveProjection(X);
    const Eigen::Matrix<double, 2, 1> m = perspectiveProjection(X);

    Eigen::Matrix<double, 2, 3> jacobianApprox;
    for (size_t i = 0; i < 3; i++) {
        Eigen::Matrix<double, 3, 1> dx = Eigen::Matrix<double, 3, 1>::Zero();
        dx(i) += epsilon;

        const Eigen::Matrix<double, 2, 1> mp = perspectiveProjection((X + dx).eval());
        const Eigen::Matrix<double, 2, 1> d = (mp - m) / epsilon;

        jacobianApprox.col(i) = d;
    }

    for (size_t row = 0; row < 2; row++) {
        for (size_t col = 0; col < 3; col++) {
            EXPECT_NEAR(jacobianApprox(row, col), jacobian(row, col), 1e-7);
        }
    }
}

TEST(computeJacobianImuToWorldRotation, finiteDifference) {
    const double epsilon = 1e-8;
    const Eigen::Matrix<double, 3, 1> imuToWorldRotationAxis(.1, -.1, .2);
    const Eigen::Quaternion<double> imuToWorldRotation(
        Eigen::AngleAxis<double>(imuToWorldRotationAxis.norm(), imuToWorldRotationAxis.normalized()));
    const Eigen::Matrix<double, 3, 1> imuPositionInWorld(10, 100, 200);

    const Eigen::Matrix<double, 3, 1> cameraToImuRotationAxis(.01, -.01, .02);
    const Eigen::Quaternion<double> cameraToImuRotation(
        Eigen::AngleAxis<double>(cameraToImuRotationAxis.norm(), cameraToImuRotationAxis.normalized()));
    const Eigen::Matrix<double, 3, 1> cameraPositionInImu(1, 2, 3);

    const Eigen::Matrix<double, 3, 1> worldPoint(5, 10, 1000);

    const Eigen::Matrix<double, 2, 1> imagePoint = perspectiveProjection(
        mapWorldPointToCamera(imuToWorldRotation, imuPositionInWorld, cameraToImuRotation, cameraPositionInImu, worldPoint));

    const Eigen::Matrix<double, 2, 3> jacobianImuToWorldRotation
        = computeJacobianImuToWorldRotation(imuToWorldRotation, imuPositionInWorld, cameraToImuRotation, cameraPositionInImu, worldPoint);
    const Eigen::Matrix<double, 2, 3> jacobianImuToWorldPosition
        = computeJacobianImuToWorldPosition(imuToWorldRotation, imuPositionInWorld, cameraToImuRotation, cameraPositionInImu, worldPoint);
    const Eigen::Matrix<double, 2, 3> jacobianCameraToImuRotation
        = computeJacobianCameraToImuRotation(imuToWorldRotation, imuPositionInWorld, cameraToImuRotation, cameraPositionInImu, worldPoint);
    const Eigen::Matrix<double, 2, 3> jacobianCameraToImuPosition
        = computeJacobianCameraToImuPosition(imuToWorldRotation, imuPositionInWorld, cameraToImuRotation, cameraPositionInImu, worldPoint);
    const Eigen::Matrix<double, 2, 3> jacobianWorldPoint
        = computeJacobianWorldPoint(imuToWorldRotation, imuPositionInWorld, cameraToImuRotation, cameraPositionInImu, worldPoint);

    Eigen::Matrix<double, 2, 3> jacobianImuToWorldRotationApprox;
    Eigen::Matrix<double, 2, 3> jacobianImuToWorldPositionApprox;
    Eigen::Matrix<double, 2, 3> jacobianCameraToImuRotationApprox;
    Eigen::Matrix<double, 2, 3> jacobianCameraToImuPositionApprox;
    Eigen::Matrix<double, 2, 3> jacobianWorldPointApprox;

    for (size_t i = 0; i < 3; i++) {
        {
            Eigen::Matrix<double, 3, 1> w = Eigen::Matrix<double, 3, 1>::Zero();
            w(i) += epsilon;

            const Eigen::Quaternion<double> deltaQuaternion(Eigen::AngleAxis<double>(w.norm(), w.normalized()));
            const Eigen::Matrix<double, 2, 1> imagePointPerturbed = perspectiveProjection(mapWorldPointToCamera(
                imuToWorldRotation * deltaQuaternion, imuPositionInWorld, cameraToImuRotation, cameraPositionInImu, worldPoint));
            const Eigen::Matrix<double, 2, 1> d = (imagePointPerturbed - imagePoint) / epsilon;

            jacobianImuToWorldRotationApprox.col(i) = d;
        }

        {
            Eigen::Matrix<double, 3, 1> deltaPosition = Eigen::Matrix<double, 3, 1>::Zero();
            deltaPosition(i) += epsilon;

            const Eigen::Matrix<double, 2, 1> imagePointPerturbed = perspectiveProjection(mapWorldPointToCamera(
                imuToWorldRotation, (imuPositionInWorld + deltaPosition).eval(), cameraToImuRotation, cameraPositionInImu, worldPoint));
            const Eigen::Matrix<double, 2, 1> d = (imagePointPerturbed - imagePoint) / epsilon;

            jacobianImuToWorldPositionApprox.col(i) = d;
        }

        {
            Eigen::Matrix<double, 3, 1> w = Eigen::Matrix<double, 3, 1>::Zero();
            w(i) += epsilon;

            const Eigen::Quaternion<double> deltaQuaternion(Eigen::AngleAxis<double>(w.norm(), w.normalized()));
            const Eigen::Matrix<double, 2, 1> imagePointPerturbed = perspectiveProjection(mapWorldPointToCamera(
                imuToWorldRotation, imuPositionInWorld, cameraToImuRotation * deltaQuaternion, cameraPositionInImu, worldPoint));
            const Eigen::Matrix<double, 2, 1> d = (imagePointPerturbed - imagePoint) / epsilon;

            jacobianCameraToImuRotationApprox.col(i) = d;
        }

        {
            Eigen::Matrix<double, 3, 1> deltaPosition = Eigen::Matrix<double, 3, 1>::Zero();
            deltaPosition(i) += epsilon;

            const Eigen::Matrix<double, 2, 1> imagePointPerturbed = perspectiveProjection(mapWorldPointToCamera(
                imuToWorldRotation, imuPositionInWorld, cameraToImuRotation, (cameraPositionInImu + deltaPosition).eval(), worldPoint));
            const Eigen::Matrix<double, 2, 1> d = (imagePointPerturbed - imagePoint) / epsilon;

            jacobianCameraToImuPositionApprox.col(i) = d;
        }

        {
            Eigen::Matrix<double, 3, 1> deltaPosition = Eigen::Matrix<double, 3, 1>::Zero();
            deltaPosition(i) += epsilon;

            const Eigen::Matrix<double, 2, 1> imagePointPerturbed = perspectiveProjection(mapWorldPointToCamera(
                imuToWorldRotation, imuPositionInWorld, cameraToImuRotation, cameraPositionInImu, (worldPoint + deltaPosition).eval()));
            const Eigen::Matrix<double, 2, 1> d = (imagePointPerturbed - imagePoint) / epsilon;

            jacobianWorldPointApprox.col(i) = d;
        }
    }

    for (size_t row = 0; row < 2; row++) {
        for (size_t col = 0; col < 3; col++) {
            EXPECT_NEAR(jacobianImuToWorldRotationApprox(row, col), jacobianImuToWorldRotation(row, col), 1e-7);
            EXPECT_NEAR(jacobianImuToWorldPositionApprox(row, col), jacobianImuToWorldPosition(row, col), 1e-7);
            EXPECT_NEAR(jacobianCameraToImuRotationApprox(row, col), jacobianCameraToImuRotation(row, col), 1e-7);
            EXPECT_NEAR(jacobianCameraToImuPositionApprox(row, col), jacobianCameraToImuPosition(row, col), 1e-7);
            EXPECT_NEAR(jacobianWorldPointApprox(row, col), jacobianWorldPoint(row, col), 1e-7);
        }
    }
}

TEST(computeLeftNullSpace, smallMatrix) {
    std::srand(1234);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::MatrixXd::Random(10, 3);
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> L = computeLeftNullSpace(A);

    EXPECT_EQ(10, L.rows());
    EXPECT_EQ(7, L.cols());

    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X = L.transpose() * A;

    for (long row = 0; row < X.rows(); row++) {
        for (long col = 0; col < X.cols(); col++) {
            EXPECT_NEAR(0, X(row, col), 1e-10);
        }
    }
}

TEST(computeTrackJacobian, threeObservations) {
    const double epsilon = 1e-8;

    const Eigen::Matrix<double, 3, 1> cameraToImuRotationAngle(.1, -.2, .3);
    const Eigen::Quaternion<double> cameraToImuRotation(
        Eigen::AngleAxis<double>(cameraToImuRotationAngle.norm(), cameraToImuRotationAngle.normalized()));
    const Eigen::Matrix<double, 3, 1> cameraToImuPosition(0, 0, 0);
    const Eigen::Matrix<double, 3, 1> worldPoint(10, 10, 1000);

    std::vector<ReprojectionMeasurement<double> > measurements;

    const Eigen::Matrix<double, 3, 1> imuToWorldRotationAngle1(-.2, .3, -.4);
    const Eigen::Quaternion<double> imuToWorldRotation1(
        Eigen::AngleAxis<double>(imuToWorldRotationAngle1.norm(), imuToWorldRotationAngle1.normalized()));
    const Eigen::Matrix<double, 3, 1> imuToWorldPosition1(1, 2, 3);
    const Eigen::Matrix<double, 2, 1> calibratedImagePoint1(0, 0);
    measurements.push_back(ReprojectionMeasurement<double>(imuToWorldRotation1, imuToWorldPosition1, calibratedImagePoint1, 1));

    const Eigen::Matrix<double, 3, 1> imuToWorldRotationAngle2(.3, .4, -.5);
    const Eigen::Quaternion<double> imuToWorldRotation2(
        Eigen::AngleAxis<double>(imuToWorldRotationAngle2.norm(), imuToWorldRotationAngle2.normalized()));
    const Eigen::Matrix<double, 3, 1> imuToWorldPosition2(4, 5, 6);
    const Eigen::Matrix<double, 2, 1> calibratedImagePoint2(0, 0);
    measurements.push_back(ReprojectionMeasurement<double>(imuToWorldRotation2, imuToWorldPosition2, calibratedImagePoint2, 2));

    const Eigen::Matrix<double, 3, 1> imuToWorldRotationAngle3(.4, .5, -.6);
    const Eigen::Quaternion<double> imuToWorldRotation3(
        Eigen::AngleAxis<double>(imuToWorldRotationAngle3.norm(), imuToWorldRotationAngle3.normalized()));
    const Eigen::Matrix<double, 3, 1> imuToWorldPosition3(7, 8, 9);
    const Eigen::Matrix<double, 2, 1> calibratedImagePoint3(0, 0);
    measurements.push_back(ReprojectionMeasurement<double>(imuToWorldRotation3, imuToWorldPosition3, calibratedImagePoint3, 3));

    Eigen::Matrix<double, Eigen::Dynamic, 1> residual;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobian;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> leftNullSpace;
    Eigen::Matrix<double, Eigen::Dynamic, 1> stackedResidual;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> stackedJacobiansRandomVariables;

    computeTrackJacobian(residual, jacobian, leftNullSpace, stackedResidual, stackedJacobiansRandomVariables, measurements,
        cameraToImuRotation, cameraToImuPosition, worldPoint);

    std::cout << jacobian << std::endl << std::endl;

    Eigen::Matrix<double, 3, 3> jacobianCameraToImuRotationApprox;
    Eigen::Matrix<double, 3, 3> jacobianCameraToImuPositionApprox;
    Eigen::Matrix<double, 3, 3> jacobianImuToWorldRotation1Approx;
    Eigen::Matrix<double, 3, 3> jacobianImuToWorldPosition1Approx;
    Eigen::Matrix<double, 3, 3> jacobianImuToWorldRotation2Approx;
    Eigen::Matrix<double, 3, 3> jacobianImuToWorldPosition2Approx;
    Eigen::Matrix<double, 3, 3> jacobianImuToWorldRotation3Approx;
    Eigen::Matrix<double, 3, 3> jacobianImuToWorldPosition3Approx;

    for (size_t i = 0; i < 3; i++) {
        {
            Eigen::Matrix<double, 3, 1> w = Eigen::Matrix<double, 3, 1>::Zero();
            w(i) += epsilon;

            const Eigen::Quaternion<double> deltaQuaternion(Eigen::AngleAxis<double>(w.norm(), w.normalized()));

            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedJacobian;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedLeftNullSpace;
            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedStackedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedStackedJacobiansRandomVariables;
            computeTrackJacobian(perturbedResidual, perturbedJacobian, perturbedLeftNullSpace, perturbedStackedResidual,
                perturbedStackedJacobiansRandomVariables, measurements, cameraToImuRotation * deltaQuaternion, cameraToImuPosition,
                worldPoint);

            const Eigen::Matrix<double, 3, 1> d = (leftNullSpace.transpose() * perturbedStackedResidual - residual) / epsilon;

            jacobianCameraToImuRotationApprox.col(i) = d;
        }

        {
            Eigen::Matrix<double, 3, 1> deltaPosition = Eigen::Matrix<double, 3, 1>::Zero();
            deltaPosition(i) += epsilon;

            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedJacobian;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedLeftNullSpace;
            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedStackedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedStackedJacobiansRandomVariables;
            computeTrackJacobian(perturbedResidual, perturbedJacobian, perturbedLeftNullSpace, perturbedStackedResidual,
                perturbedStackedJacobiansRandomVariables, measurements, cameraToImuRotation, (cameraToImuPosition + deltaPosition).eval(),
                worldPoint);

            const Eigen::Matrix<double, 3, 1> d = (leftNullSpace.transpose() * perturbedStackedResidual - residual) / epsilon;

            jacobianCameraToImuPositionApprox.col(i) = d;
        }

        {
            std::vector<ReprojectionMeasurement<double> > measurementsCopy;

            Eigen::Matrix<double, 3, 1> w = Eigen::Matrix<double, 3, 1>::Zero();
            w(i) += epsilon;

            const Eigen::Quaternion<double> deltaQuaternion(Eigen::AngleAxis<double>(w.norm(), w.normalized()));
            measurementsCopy.push_back(ReprojectionMeasurement<double>(measurements[0].m_imuToWorldRotation * deltaQuaternion,
                measurements[0].m_imuToWorldPosition, measurements[0].m_calibratedImagePoint, measurements[0].m_cloneId));
            measurementsCopy.push_back(measurements[1]);
            measurementsCopy.push_back(measurements[2]);

            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedJacobian;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedLeftNullSpace;
            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedStackedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedStackedJacobiansRandomVariables;
            computeTrackJacobian(perturbedResidual, perturbedJacobian, perturbedLeftNullSpace, perturbedStackedResidual,
                perturbedStackedJacobiansRandomVariables, measurementsCopy, cameraToImuRotation, cameraToImuPosition, worldPoint);

            const Eigen::Matrix<double, 3, 1> d = (leftNullSpace.transpose() * perturbedStackedResidual - residual) / epsilon;

            jacobianImuToWorldRotation1Approx.col(i) = d;
        }

        {
            std::vector<ReprojectionMeasurement<double> > measurementsCopy;

            Eigen::Matrix<double, 3, 1> deltaPosition = Eigen::Matrix<double, 3, 1>::Zero();
            deltaPosition(i) += epsilon;

            measurementsCopy.push_back(ReprojectionMeasurement<double>(measurements[0].m_imuToWorldRotation,
                (measurements[0].m_imuToWorldPosition + deltaPosition).eval(), measurements[0].m_calibratedImagePoint,
                measurements[0].m_cloneId));
            measurementsCopy.push_back(measurements[1]);
            measurementsCopy.push_back(measurements[2]);

            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedJacobian;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedLeftNullSpace;
            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedStackedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedStackedJacobiansRandomVariables;
            computeTrackJacobian(perturbedResidual, perturbedJacobian, perturbedLeftNullSpace, perturbedStackedResidual,
                perturbedStackedJacobiansRandomVariables, measurementsCopy, cameraToImuRotation, cameraToImuPosition, worldPoint);

            const Eigen::Matrix<double, 3, 1> d = (leftNullSpace.transpose() * perturbedStackedResidual - residual) / epsilon;

            jacobianImuToWorldPosition1Approx.col(i) = d;
        }

        {
            std::vector<ReprojectionMeasurement<double> > measurementsCopy;

            Eigen::Matrix<double, 3, 1> w = Eigen::Matrix<double, 3, 1>::Zero();
            w(i) += epsilon;

            const Eigen::Quaternion<double> deltaQuaternion(Eigen::AngleAxis<double>(w.norm(), w.normalized()));
            measurementsCopy.push_back(measurements[0]);
            measurementsCopy.push_back(ReprojectionMeasurement<double>(measurements[1].m_imuToWorldRotation * deltaQuaternion,
                measurements[1].m_imuToWorldPosition, measurements[1].m_calibratedImagePoint, measurements[1].m_cloneId));
            measurementsCopy.push_back(measurements[2]);

            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedJacobian;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedLeftNullSpace;
            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedStackedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedStackedJacobiansRandomVariables;
            computeTrackJacobian(perturbedResidual, perturbedJacobian, perturbedLeftNullSpace, perturbedStackedResidual,
                perturbedStackedJacobiansRandomVariables, measurementsCopy, cameraToImuRotation, cameraToImuPosition, worldPoint);

            const Eigen::Matrix<double, 3, 1> d = (leftNullSpace.transpose() * perturbedStackedResidual - residual) / epsilon;

            jacobianImuToWorldRotation2Approx.col(i) = d;
        }

        {
            std::vector<ReprojectionMeasurement<double> > measurementsCopy;

            Eigen::Matrix<double, 3, 1> deltaPosition = Eigen::Matrix<double, 3, 1>::Zero();
            deltaPosition(i) += epsilon;

            measurementsCopy.push_back(measurements[0]);
            measurementsCopy.push_back(ReprojectionMeasurement<double>(measurements[1].m_imuToWorldRotation,
                (measurements[1].m_imuToWorldPosition + deltaPosition).eval(), measurements[1].m_calibratedImagePoint,
                measurements[1].m_cloneId));
            measurementsCopy.push_back(measurements[2]);

            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedJacobian;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedLeftNullSpace;
            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedStackedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedStackedJacobiansRandomVariables;
            computeTrackJacobian(perturbedResidual, perturbedJacobian, perturbedLeftNullSpace, perturbedStackedResidual,
                perturbedStackedJacobiansRandomVariables, measurementsCopy, cameraToImuRotation, cameraToImuPosition, worldPoint);

            const Eigen::Matrix<double, 3, 1> d = (leftNullSpace.transpose() * perturbedStackedResidual - residual) / epsilon;

            jacobianImuToWorldPosition2Approx.col(i) = d;
        }

        {
            std::vector<ReprojectionMeasurement<double> > measurementsCopy;

            Eigen::Matrix<double, 3, 1> w = Eigen::Matrix<double, 3, 1>::Zero();
            w(i) += epsilon;

            const Eigen::Quaternion<double> deltaQuaternion(Eigen::AngleAxis<double>(w.norm(), w.normalized()));
            measurementsCopy.push_back(measurements[0]);
            measurementsCopy.push_back(measurements[1]);
            measurementsCopy.push_back(ReprojectionMeasurement<double>(measurements[2].m_imuToWorldRotation * deltaQuaternion,
                measurements[2].m_imuToWorldPosition, measurements[2].m_calibratedImagePoint, measurements[2].m_cloneId));

            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedJacobian;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedLeftNullSpace;
            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedStackedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedStackedJacobiansRandomVariables;
            computeTrackJacobian(perturbedResidual, perturbedJacobian, perturbedLeftNullSpace, perturbedStackedResidual,
                perturbedStackedJacobiansRandomVariables, measurementsCopy, cameraToImuRotation, cameraToImuPosition, worldPoint);

            const Eigen::Matrix<double, 3, 1> d = (leftNullSpace.transpose() * perturbedStackedResidual - residual) / epsilon;

            jacobianImuToWorldRotation3Approx.col(i) = d;
        }

        {
            std::vector<ReprojectionMeasurement<double> > measurementsCopy;

            Eigen::Matrix<double, 3, 1> deltaPosition = Eigen::Matrix<double, 3, 1>::Zero();
            deltaPosition(i) += epsilon;

            measurementsCopy.push_back(measurements[0]);
            measurementsCopy.push_back(measurements[1]);
            measurementsCopy.push_back(ReprojectionMeasurement<double>(measurements[2].m_imuToWorldRotation,
                (measurements[2].m_imuToWorldPosition + deltaPosition).eval(), measurements[2].m_calibratedImagePoint,
                measurements[2].m_cloneId));

            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedJacobian;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedLeftNullSpace;
            Eigen::Matrix<double, Eigen::Dynamic, 1> perturbedStackedResidual;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> perturbedStackedJacobiansRandomVariables;
            computeTrackJacobian(perturbedResidual, perturbedJacobian, perturbedLeftNullSpace, perturbedStackedResidual,
                perturbedStackedJacobiansRandomVariables, measurementsCopy, cameraToImuRotation, cameraToImuPosition, worldPoint);

            const Eigen::Matrix<double, 3, 1> d = (leftNullSpace.transpose() * perturbedStackedResidual - residual) / epsilon;

            jacobianImuToWorldPosition3Approx.col(i) = d;
        }
    }

    Eigen::MatrixXd jacobianApprox = Eigen::MatrixXd::Zero(jacobian.rows(), jacobian.cols());

    jacobianApprox.template block<3, 3>(0, 0) = jacobianCameraToImuRotationApprox;
    jacobianApprox.template block<3, 3>(0, 3) = jacobianCameraToImuPositionApprox;
    jacobianApprox.template block<3, 3>(0, 6) = jacobianImuToWorldRotation1Approx;
    jacobianApprox.template block<3, 3>(0, 9) = jacobianImuToWorldPosition1Approx;
    jacobianApprox.template block<3, 3>(0, 12) = jacobianImuToWorldRotation2Approx;
    jacobianApprox.template block<3, 3>(0, 15) = jacobianImuToWorldPosition2Approx;
    jacobianApprox.template block<3, 3>(0, 18) = jacobianImuToWorldRotation3Approx;
    jacobianApprox.template block<3, 3>(0, 21) = jacobianImuToWorldPosition3Approx;

    EXPECT_EQ(3, residual.rows());
    EXPECT_EQ(1, residual.cols());
    EXPECT_EQ(3, jacobian.rows());
    EXPECT_EQ(24, jacobian.cols());

    for (long row = 0; row < jacobian.rows(); row++) {
        for (long col = 0; col < jacobian.cols(); col++) {
            EXPECT_NEAR(jacobianApprox(row, col), jacobian(row, col), 1e-7);
        }
    }
}
