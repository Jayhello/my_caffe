find_path(LMDB_INCLUDE_DIR NAMES  lmdb.h PATHS "$ENV{LMDB_DIR}/include")
find_library(LMDB_LIBRARIES NAMES lmdb   PATHS "$ENV{LMDB_DIR}/lib" )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LMDB DEFAULT_MSG LMDB_INCLUDE_DIR LMDB_LIBRARIES)

if(LMDB_FOUND)
    message(STATUS "Found lmdb    (include: ${LMDB_INCLUDE_DIR}, library: ${LMDB_LIBRARIES})")
    mark_as_advanced(LMDB_INCLUDE_DIR LMDB_LIBRARIES)
endif()