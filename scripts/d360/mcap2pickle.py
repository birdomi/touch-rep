"""
Sample script of MCAP to Pickle Converter for D360 Dataset

This script converts multiple MCAP (ROS2 bag) files to pickle format for easier data processing.
It extracts specified topics from D360 sensors, Allegro hand, Franka robot, and other sources.

Input:
    - MCAP files in the specified directory (./rosbags_dataset/)
    - Configuration of which data types to extract (images, audio, IMU, pressure, robot states)

Output:
    - Structured pickle files in the output directory (./dataset_extracted/data.pickle)
    - Each bag is processed into its own subdirectory with extracted data
"""

from scripts.d360.mcap_utils import save_to_pickle
import os
from glob import glob

allegro_topics = ["joint_states", "joint_cmd"] # Topics to extract for the Allegro hand
franka_topics = None # Topics to extract for the Franka robot, set to None if not needed
extra_topics = None # Additional topics to extract (eg realsense topics), set to None if not needed

path_bags = f"./rosbags_dataset/" # Change this to your local path where the MCAP files are stored
path_output_base = f"./dataset_extracted/" # Change this to your local path where you want to save the extracted data


for path_rosbag in glob(path_bags + "/*"):
    rosbag_name = path_rosbag.split("/")[-1]

    file_input = f"{path_rosbag}/{rosbag_name}_0.mcap"
    path_output = f"{path_output_base}/{rosbag_name}/" # Update to the desired output path tree structure
    os.makedirs(path_output, exist_ok=True)

    print(f"Processing {rosbag_name}")
    save_to_pickle(
        file=file_input,
        file_out=path_output,
        image=True, # Set to True if you want to extract images from D360 topics in the MCAP file
        audio=True, # Set to True if you want to extract audio from D360 topics in the MCAP file
        imu=True, # Set to True if you want to extract IMU data from D360 topics in the MCAP file
        pressure=True, # Set to True if you want to extract pressure data from D360 topics in the MCAP file
        allegro=allegro_topics,
        franka=franka_topics,
        extra_topics=extra_topics,
    )
    print(f"Done - {file_input} \n")
