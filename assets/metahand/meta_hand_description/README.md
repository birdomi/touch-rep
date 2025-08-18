This is a ROS package that generates URDFs for variants of the MetaHand. It supports left and right hands, as well as both original Digit and Digit 360. It has been tested with ROS Noetic.

# Install

Make sure this repo is inside of the *src* folder of a catkin workspace. For example, this file could be located at *[ws_root]/src/GUM/gum/meta_hand_description/README.md*

In the root of the workspace, run ```catkin_make``` to build URDFs for the hand. The resulting URDFs can be found at *[ws_root]/devel/share/meta_hand_description/robots*

# Test

Make sure to source the workspace:

```
source [ws_root]/devel/setup.bash
```

The following command will load URDF and launch rviz:

```
roslaunch meta_hand_description test_hand.launch
```

There are also arguments 'side' and 'fingertip' that can take arguments 'left/right' and 'digit/digitv2' respectively. For example:

```
roslaunch meta_hand_description test_hand.launch side:=right fingertip:=digitv2
```
