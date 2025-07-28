cmake_minimum_required(VERSION 3.26)

# Collect all source files
file(GLOB SOURCE_FILES "*.cc")

# Define executable
add_executable(text_generation llama_main.cc ${SOURCE_FILES})

# Link RBLN runtime
find_package(rbln CONFIG REQUIRED)
target_link_libraries(text_generation rbln::rbln_runtime)

# Add header files directory
target_include_directories(text_generation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
