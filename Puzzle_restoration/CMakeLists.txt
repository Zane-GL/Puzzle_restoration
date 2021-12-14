cmake_minimum_required(VERSION 3.14)

# opencv_work为工程输出的exe名字

project(opencv_work CXX)

# msys64 可以用pkg-config进行管理

find_package(PkgConfig REQUIRED)

# PKGS_DEP是用户自定义的cmake模块名，REQUIRED后面是要导入的模块名称（以空格分割）

pkg_check_modules(PKGS_DEP REQUIRED opencv4)

# C++标准是c++11

set(CMAKE_CXX_STANDARD 11)

# 让工程输出的可执行文件在当前目录下

set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_CURRENT_SOURCE_DIR})



# 将PkgConfig搜索到的编译参数应用到当前工程

FOREACH(flag "${PKGS_DEP_CFLAGS}")

    set(EXTRA_CFLAGS "${EXTRA_CFLAGS} ${flag}")

ENDFOREACH(flag)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS}")



# 将当前目录下的include文件夹添加为c++搜索头文件的默认目录

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)





add_executable(${PROJECT_NAME} main.cpp)

# 将PkgConfig搜索到库名应用到当前目录

target_link_libraries(${PROJECT_NAME} ${PKGS_DEP_LDFLAGS})