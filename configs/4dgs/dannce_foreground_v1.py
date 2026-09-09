"""Mouse-only 4D-GS settings for the masked 80-frame DANNCE pilot."""

ModelHiddenParams = dict(
    kplanes_config={
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 16,
        "resolution": [64, 64, 64, 80],
    },
    multires=[1, 2],
    defor_depth=0,
    net_width=128,
    plane_tv_weight=0.0002,
    time_smoothness_weight=0.001,
    l1_time_planes=0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    no_dr=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,
)

OptimizationParams = dict(
    dataloader=True,
    iterations=10000,
    batch_size=1,
    coarse_iterations=1000,
    densify_from_iter=200,
    pruning_from_iter=200,
    densify_until_iter=8000,
    opacity_threshold_coarse=0.005,
    opacity_threshold_fine_init=0.005,
    opacity_threshold_fine_after=0.005,
)
