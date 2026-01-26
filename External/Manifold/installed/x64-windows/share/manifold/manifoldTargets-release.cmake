#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "manifold::manifold" for configuration "Release"
set_property(TARGET manifold::manifold APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(manifold::manifold PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/manifold.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/manifold.dll"
  )

list(APPEND _cmake_import_check_targets manifold::manifold )
list(APPEND _cmake_import_check_files_for_manifold::manifold "${_IMPORT_PREFIX}/lib/manifold.lib" "${_IMPORT_PREFIX}/bin/manifold.dll" )

# Import target "manifold::manifoldc" for configuration "Release"
set_property(TARGET manifold::manifoldc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(manifold::manifoldc PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/manifoldc.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "manifold::manifold"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/manifoldc.dll"
  )

list(APPEND _cmake_import_check_targets manifold::manifoldc )
list(APPEND _cmake_import_check_files_for_manifold::manifoldc "${_IMPORT_PREFIX}/lib/manifoldc.lib" "${_IMPORT_PREFIX}/bin/manifoldc.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
