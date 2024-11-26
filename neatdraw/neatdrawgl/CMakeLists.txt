CPMAddPackage("gh:MumblyJuergens/glube#main")

set(neatdrawgl_src
    "src/neatdrawgl.cpp"
)

set(neatdrawgl_headers
    "include/neat/neatdrawgl.hpp"
)

add_library(neatdrawgl ${neatdrawgl_src} ${neatdrawgl_headers})
add_library(neat::neatdrawgl ALIAS neatdrawgl)
target_compile_features(neatdrawgl PUBLIC cxx_std_23)
target_link_libraries(neatdrawgl PUBLIC neat::neat mj::mj glm::glm glube)
MJFullCompilerWarnings(neatdrawgl)
MJLibrarySetup(neatdrawgl INCLUDE_DIR "include" EXPORT_HEADER_PREFIX "neat/neatdrawgl" HEADERS ${neatdrawgl_headers})
# if (CMAKE_BUILD_TYPE STREQUAL "Release")
#     MJSanitize(neatdrawgl ON OFF ON OFF OFF)
# endif()