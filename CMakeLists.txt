cmake_minimum_required(VERSION 3.16)
project(segDAC_sim)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})
add_executable(binaryDAC binary_DAC.cpp)
  target_link_libraries( binaryDAC
   
    Eigen3::Eigen
    ${ROOT_LIBRARIES}

  )
add_executable(thermDAC therm_DAC.cpp)
  target_link_libraries(thermDAC
   
     Eigen3::Eigen
     ${ROOT_LIBRARIES}

  )
add_executable(segDAC Segment_DAC.cpp)
  target_link_libraries(segDAC
   
    Eigen3::Eigen
    ${ROOT_LIBRARIES}
  )
add_executable(segAll Segment_all_DAC.cpp)
  target_link_libraries(segAll
    Eigen3::Eigen
    ${ROOT_LIBRARIES}
  )
