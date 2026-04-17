from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import (
    DeclareLaunchArgument,
)

from launch.substitutions import (
    LaunchConfiguration
)

def generate_launch_description():

    pkg_uncertainty_nav = FindPackageShare("uncertainty_nav")
    pkg_gazebo_ros      = FindPackageShare("gazebo_ros")

    env_arg = DeclareLaunchArgument(
        "env", default_value="A",
        description="Environment: A (training, moderate noise) or B (test, high noise)"
    )
    policy_arg = DeclareLaunchArgument(
        "policy", default_value="ensemble",
        description="Policy type: ensemble | vanilla | lstm | gru | large_mlp"
    )
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint", default_value="",
        description="Absolute path to policy .pt checkpoint file"
    )
    unc_threshold_arg = DeclareLaunchArgument(
        "uncertainty_threshold", default_value="0.3",
        description="Epistemic uncertainty threshold for cautious behavior"
    )
    caution_scale_arg = DeclareLaunchArgument(
        "caution_scale", default_value="0.5",
        description="Action scale factor when uncertainty > threshold"
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="true",
        description="Launch RViz2"
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true"
    )

    use_rviz       = LaunchConfiguration("use_rviz")
    use_sim_time   = LaunchConfiguration("use_sim_time")

