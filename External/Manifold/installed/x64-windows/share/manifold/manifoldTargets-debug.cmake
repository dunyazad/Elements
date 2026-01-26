#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "manifold::manifold" for configuration "Debug"
set_property(TARGET manifold::manifold APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(manifold::manifold PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/manifold.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/manifold.dll"
  )

list(APPEND _cmake_import_check_targets manifold::manifold )
list(APPEND _cmake_import_check_files_for_manifold::manifold "${_IMPORT_PREFIX}/debug/lib/manifold.lib" "${_IMPORT_PREFIX}/debug/bin/manifold.dll" )

# Import target "manifold::manifoldc" for configuration "Debug"
set_property(TARGET manifold::manifoldc APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(manifold::manifoldc PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/manifoldc.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "manifold::manifold"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/manifoldc.dll"
  )

list(APPEND _cmake_import_check_targets manifold::manifoldc )
list(APPEND _cmake_import_check_files_for_manifold::manifoldc "${_IMPORT_PREFIX}/debug/lib/manifoldc.lib" "${_IMPORT_PREFIX}/debug/bin/manifoldc.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
