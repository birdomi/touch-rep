cmake_minimum_required(VERSION 3.5)
project(meta_hand_description)


find_package(ament_cmake REQUIRED)


install(DIRECTORY
  meshes
  robots
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
