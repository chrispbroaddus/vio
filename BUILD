load("//tools:cpp_compile_flags.bzl", "COPTS")

cc_library(
    name = "vio",
    srcs = [],
    hdrs = [
        "include/calibration_state.h",
        "include/clone_state.h",
        "include/imu_state.h",
        "include/kalman_filter_details.h",
        "include/monocular_triangulation.h",
        "include/state.h",
        "include/stereo_triangulation.h",
        "include/vision_jacobian.h",
        "include/visual_inertial_extended_kalman_filter.h",
    ],
    copts = COPTS,
    visibility = ["//visibility:public"],
    deps = [
        "//external:eigen",
        "//external:glog",
        "//packages/feature_tracker",
        "//packages/imu_propagator",
        "//packages/triangulation",
    ],
)
