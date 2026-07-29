cmake_minimum_required(VERSION 3.5)
project(obigo_launch)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

install(
  PROGRAMS
    scripts/planning_sim_empty_occupancy_grid.py
    scripts/planning_sim_localization_state.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
  rviz
  config_files
)
