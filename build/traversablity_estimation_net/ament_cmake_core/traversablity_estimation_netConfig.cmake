# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_traversablity_estimation_net_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED traversablity_estimation_net_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(traversablity_estimation_net_FOUND FALSE)
  elseif(NOT traversablity_estimation_net_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(traversablity_estimation_net_FOUND FALSE)
  endif()
  return()
endif()
set(_traversablity_estimation_net_CONFIG_INCLUDED TRUE)

# output package information
if(NOT traversablity_estimation_net_FIND_QUIETLY)
  message(STATUS "Found traversablity_estimation_net: 0.0.0 (${traversablity_estimation_net_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'traversablity_estimation_net' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${traversablity_estimation_net_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(traversablity_estimation_net_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_dependencies-extras.cmake")
foreach(_extra ${_extras})
  include("${traversablity_estimation_net_DIR}/${_extra}")
endforeach()
