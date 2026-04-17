from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare



def generate_launch_description():

    pkg_uncertainty_nav = FindPackageShare("uncertainty_nav")
    pkg_gazebo_ros      = FindPackageShare("gazebo_ros")
