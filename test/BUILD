load("//tools:cpp_compile_flags.bzl", "COPTS")

cc_test(
    name = "kalman_filter_details_test",
    srcs = [
        "calibration_state_test.cpp",
        "clone_state_test.cpp",
        "imu_state_test.cpp",
        "kalman_filter_details_test.cpp",
        "state_test.cpp",
        "vision_jacobian_test.cpp",
        "visual_inertial_extended_kalman_filter_test.cpp",
    ],
    copts = COPTS,
    deps = [
        "//packages/math/linear_algebra",
        "//packages/vio",
        "@gtest//:main",
    ],
)

cc_test(
    name = "triangulation_test",
    srcs = [
        "monocular_triangulation_test.cpp",
        "stereo_trianguation_test.cpp",
    ],
    deps = [
        "//packages/vio",
        "@gtest//:main",
    ],
)
