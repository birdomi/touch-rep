# ROS2 node for Sparsh-X inputs topic publication

## Overview

This ROS2 node processes and formats data streams from one or multiple D360 tactile sensors in real-time, preparing them for use with the Sparsh-X encoder. The formatted data is published to ROS2 topics in a structure compatible with downstream tasks such as robot control policies.

This node serves as a critical bridge between the raw D360 tactile sensor data and higher-level control algorithms, ensuring that tactile information is properly formated for using Sparsh-X encoder. Launch this node after D360 sensors are streaming data and before starting any policy deployment on the robot.

## Real-time deployment

Note: this assumes the D360 topics are already being published.

Deploy the Sparsh-x data node running the bash script:

```bash
bash run_sparsh_input_nodes.sh < LIST OF D360 DEVICES> 
```
For instance, for all D360's mounted in the Allegro hand:
```bash
bash run_sparsh_input_nodes.sh d360_0 d360_1 d360_2 d360_3 
```

The above example if for working with 4 D360s. In case of testing with only one:
```bash
bash run_sparsh_input_nodes.sh d360_<ID>
```


## For debugging 

These instructions help you emulate D360 sensor data streaming during policy deployment development and debugging:

1. Play a recorded rosbag file containing D360 sensor data:
```bash
ros2 bag play rosbag2_2025_02_18-19_30_20/rosbag2_2025_02_18-19_30_20_0.mcap
```

2. Run the Sparsh-X data node to process the recorded data:
```bash
bash run_sparsh_input_nodes.sh d360_0 d360_1 d360_2 d360_3
```

3. Now you can launch your downstream task script:
```bash
python my_policy_node.py 
```
