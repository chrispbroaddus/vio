
#include "packages/vio/include/kalman_filter_details.h"
#include "packages/math/linear_algebra/decompositions.h"
#include "packages/vio/include/state.h"
#include "gtest/gtest.h"

using namespace vio;

template <typename T> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> rawMatrixToEigen(const T* A, const size_t rows, const size_t cols) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X(rows, cols);
    for (auto row = 0; row < X.rows(); row++) {
        for (auto col = 0; col < X.cols(); col++) {
            X(row, col) = A[row * X.cols() + col];
        }
    }
    return X;
}

template <typename T> Eigen::Matrix<T, Eigen::Dynamic, 1> rawVectorToEigen(const T* x, const size_t length) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> X(length);
    for (auto i = 0; i < X.rows(); i++) {
        X(i) = x[i];
    }
    return X;
}

TEST(updateKalmanFilter, compareToMatlab) {
    const double P[] = { 0.124872758719812849, 0.956935924070684063, 0.759327383131096267, 0.463260578593719163, 0.163569909784993195,
        0.024434016050373986, 0.935730872784880430, 0.740648064978614240, 0.212163205254934373, 0.665987216411110583, 0.290185265130727177,
        0.457886333854366834, 0.743688341487325988, 0.098518737688108371, 0.894389375354242788, 0.317520582899226356, 0.240478396832084607,
        0.105920416732765310, 0.823574473927838557, 0.516558208351270420, 0.653690133966475129, 0.763897944286478281, 0.681560430470315670,
        0.175009737382079589, 0.702702306950475308 };
    const double H[] = { 0.153590376619400226, 0.546449439903068490, 0.921097255892197486, 0.715212514785840137, 0.635661388861376908,
        0.953457069886247677, 0.398880752383198978, 0.794657885388753149, 0.642060828433852260, 0.950894415378135238, 0.540884081241476466,
        0.415093386613046622, 0.577394196706648710, 0.419048293624883050, 0.443964155018810369, 0.679733898210466925, 0.180737760254794377,
        0.440035595760253639, 0.390762082204174521, 0.060018819779475985, 0.036563018048452856, 0.255386740488050767, 0.257613736712437702,
        0.816140102875322682, 0.866749896999318703, 0.809203851293793353, 0.020535774658184569, 0.751946393867449991, 0.317427863655849629,
        0.631188734269011231, 0.748618871776197126, 0.923675612620407205, 0.228669482105501420, 0.814539772900651271, 0.355073651878848984,
        0.120187017987080647, 0.653699889008252932, 0.064187087391898601, 0.789073514938958276, 0.997003271606647701, 0.525045164762608763,
        0.932613572048564099, 0.767329510776574408, 0.852263890343845643, 0.224171498983127160, 0.325833628763249172, 0.163512368527525598,
        0.671202185356535530, 0.505636617571756153, 0.652451072968614931 };
    const double R[] = { 0.604990641908259352, 0.604990641908259352, 0.604990641908259352, 0.604990641908259352, 0.604990641908259352,
        0.604990641908259352, 0.604990641908259352, 0.604990641908259352, 0.604990641908259352, 0.604990641908259352 };
    const double y[] = { 0.770954220673924495, 0.042659855935048729, 0.378186137050218862, 0.704339624483367621, 0.729513045504646906,
        0.224277070664514411, 0.269054731773365030, 0.673031165004118970, 0.477492197726861245, 0.623716412667442488 };
    const double desiredDx[] = { -0.0102572331874636, 0.2258985999620680, 0.2756348789173075, 0.3707639153116442, -0.0731666302289696 };
    const double desiredPp[] = { 0.0688226513241250, 0.4421902244427486, 0.2108208897767408, -0.1956712167978680, -0.4499909532202627,
        -0.2863295821251346, 0.5780908897975418, 0.0768597574892530, -0.3420446102897404, -0.0485578358149369, -0.3045681568617397,
        0.2418943789643704, 0.0490534772237653, -0.3161278910730731, 0.2653088722734315, -0.0338584700689653, -0.1279242680401528,
        -0.1207259381453507, 0.3088378096118185, 0.0460819359519238, -0.0202188451427547, 0.2167210773812465, -0.1187731591315194,
        -0.4070611040084732, 0.2664962706693654 };

    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenP = rawMatrixToEigen(P, 5, 5);
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenH = rawMatrixToEigen(H, 10, 5);
    const Eigen::Matrix<double, Eigen::Dynamic, 1> eigenR = rawVectorToEigen(R, 10);
    const Eigen::Matrix<double, Eigen::Dynamic, 1> eigeny = rawVectorToEigen(y, 10);

    const Eigen::Matrix<double, Eigen::Dynamic, 1> eigenDesiredDx = rawVectorToEigen(desiredDx, 5);
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenDesiredPp = rawMatrixToEigen(desiredPp, 5, 5);

    Eigen::Matrix<double, Eigen::Dynamic, 1> eigendx;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenPp;

    updateKalmanFilter(eigendx, eigenPp, eigenP, eigeny, eigenH, eigenR);

    for (int i = 0; i < eigenDesiredDx.rows(); i++) {
        EXPECT_NEAR(eigendx(i), eigenDesiredDx(i), 1e-10) << "failed at index: " << i;
    }

    for (int row = 0; row < eigenDesiredPp.rows(); row++) {
        for (int col = 0; col < eigenDesiredPp.cols(); col++) {
            EXPECT_NEAR(eigenPp(row, col), eigenDesiredPp(row, col), 1e-10) << "failed at index: " << row << ", " << col;
        }
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> eigendxQR;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenPpQR;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenPQR = rawMatrixToEigen(P, 5, 5);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenHQR = rawMatrixToEigen(H, 10, 5);
    Eigen::Matrix<double, Eigen::Dynamic, 1> eigenRQR = rawVectorToEigen(R, 5);
    Eigen::Matrix<double, Eigen::Dynamic, 1> eigenyQR = rawVectorToEigen(y, 10);

    linear_algebra::applyQRBothSides(eigenHQR, eigenyQR);

    updateKalmanFilter(eigendxQR, eigenPpQR, eigenPQR, eigenyQR, eigenHQR, eigenRQR);

    for (size_t i = 0; i < (size_t)eigenDesiredDx.rows(); i++) {
        EXPECT_NEAR(eigendxQR(i), eigenDesiredDx(i), 1e-10) << "failed at index: " << i;
    }
    for (size_t row = 0; row < (size_t)eigenDesiredPp.rows(); row++) {
        for (size_t col = 0; col < (size_t)eigenDesiredPp.cols(); col++) {
            EXPECT_NEAR(eigenPpQR(row, col), eigenDesiredPp(row, col), 1e-10) << "failed at index: " << row << ", " << col;
        }
    }
}

TEST(updateKalmanFilter, update) {
    using matrix_type = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_type = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    constexpr double dt = 1;

    vector_type x(2);
    matrix_type P(2, 2);
    matrix_type F(2, 2);
    matrix_type Q(2, 2);
    matrix_type H(1, 2);
    vector_type R(1);
    R.setIdentity();
    vector_type z(1);

    x << 0, 0; // (p,v)
    P << 1e10, 0, 0, 1e10;
    F << 1, dt, 0, 1;
    Q << 1e-10, 0, 0, 1e-10;
    H << 1, 0;
    R = 1e-10 * R;

    for (size_t i = 0; i < 100; i++) {
        z << 1 + 2 * i;

        // Predict
        const matrix_type xpp = F * x;
        const matrix_type Ppp = F * P * F.transpose() + Q;
        x = xpp;
        P = Ppp;

        // Correct
        const vector_type y = z - H * x;
        vector_type dx;
        matrix_type Pp;
        updateKalmanFilter(dx, Pp, P, y, H, R);
        x += dx;
        P = Pp;
    }

    EXPECT_DOUBLE_EQ(199, x(0));
    EXPECT_NEAR(2, x(1), 0.1);
}

TEST(augmentCovariance, augmentCovariance) {
    State<double> state;
    state.cloneImuState();
    state.cloneImuState();

    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(21 + 6, 21 + 6);

    for (long i = 0; i < covariance.rows(); i++) {
        covariance(i, i) = i + 1;
    }

    const auto newCovariance = state.augmentCovariance(covariance);

    EXPECT_EQ(1, newCovariance(0, 0));
    EXPECT_EQ(21, newCovariance(20, 20));

    EXPECT_EQ(1, newCovariance(27, 27));
    EXPECT_EQ(2, newCovariance(28, 28));
    EXPECT_EQ(3, newCovariance(29, 29));
    EXPECT_EQ(13, newCovariance(30, 30));
    EXPECT_EQ(14, newCovariance(31, 31));
    EXPECT_EQ(15, newCovariance(32, 32));
}

TEST(marginalizeClone, marginalizeCloneAtIndex0) {
    State<double> state;
    state.cloneImuState();
    state.cloneImuState();

    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(21 + 2 * 6, 21 + 2 * 6);

    for (long i = 0; i < covariance.rows(); i++) {
        covariance(i, i) = i + 1;
    }
    covariance(0, covariance.cols() - 1) = 10;
    covariance(covariance.rows() - 1, 0) = 10;

    const auto newCovariance = state.marginalizeClone(covariance, 0);

    EXPECT_EQ(1, newCovariance(0, 0));
    EXPECT_EQ(21, newCovariance(20, 20));

    EXPECT_EQ(28, newCovariance(21, 21));
    EXPECT_EQ(29, newCovariance(22, 22));
    EXPECT_EQ(30, newCovariance(23, 23));
    EXPECT_EQ(31, newCovariance(24, 24));
    EXPECT_EQ(32, newCovariance(25, 25));
    EXPECT_EQ(33, newCovariance(26, 26));

    EXPECT_EQ(newCovariance(0, newCovariance.rows() - 1), 10);

    for (long row = 0; row < newCovariance.rows(); row++) {
        for (long col = 0; col < newCovariance.cols(); col++) {
            EXPECT_EQ(newCovariance(row, col), newCovariance(col, row));
        }
    }
}

TEST(marginalizeClone, marginalizeCloneAtIndex1) {
    State<double> state;
    state.cloneImuState();
    state.cloneImuState();

    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(21 + 2 * 6, 21 + 2 * 6);

    for (long i = 0; i < covariance.rows(); i++) {
        covariance(i, i) = i + 1;
    }
    covariance(0, covariance.cols() - 7) = 10;
    covariance(covariance.rows() - 7, 0) = 10;

    const auto newCovariance = state.marginalizeClone(covariance, 1);

    EXPECT_EQ(1, newCovariance(0, 0));
    EXPECT_EQ(21, newCovariance(20, 20));

    EXPECT_EQ(22, newCovariance(21, 21));
    EXPECT_EQ(23, newCovariance(22, 22));
    EXPECT_EQ(24, newCovariance(23, 23));
    EXPECT_EQ(25, newCovariance(24, 24));
    EXPECT_EQ(26, newCovariance(25, 25));
    EXPECT_EQ(27, newCovariance(26, 26));

    EXPECT_EQ(newCovariance(0, newCovariance.rows() - 1), 10);

    for (long row = 0; row < newCovariance.rows(); row++) {
        for (long col = 0; col < newCovariance.cols(); col++) {
            EXPECT_EQ(newCovariance(row, col), newCovariance(col, row));
        }
    }
}

// Indirect Kalman Filter for 3D Attitude Estimation, Nikolas Trawny and Stergios I. Roumeliotis
// Equation 70, Page 8
TEST(angleAxis, compareToApprox) {
    const Eigen::Matrix<double, 3, 1> w(.01, -.02, .03);
    const Eigen::Matrix<double, 3, 1> axis = w.normalized();
    const Eigen::Quaternion<double> q(Eigen::AngleAxis<double>(w.norm(), axis));
    const Eigen::Quaternion<double> qApprox(1, w(0) / 2, w(1) / 2, w(2));

    EXPECT_NEAR(q.x(), qApprox.x(), 0.1);
    EXPECT_NEAR(q.y(), qApprox.y(), 0.1);
    EXPECT_NEAR(q.z(), qApprox.z(), 0.1);
    EXPECT_NEAR(q.w(), qApprox.w(), 0.1);
}

TEST(updateQuaternion, verifyUpdate) {
    const Eigen::Quaternion<double> quaternion(1, 0, 0, 0);
    const Eigen::Matrix<double, 3, 1> quaternionDx(.01, -.02, .03);
    const Eigen::Quaternion<double> q = updateQuaternion(quaternion, quaternionDx);

    EXPECT_NEAR(quaternionDx(0) / 2, q.x(), 1e-6);
    EXPECT_NEAR(quaternionDx(1) / 2, q.y(), 1e-6);
    EXPECT_NEAR(quaternionDx(2) / 2, q.z(), 1e-6);
}
