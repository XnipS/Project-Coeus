cmake_minimum_required(VERSION 3.5)

project(Coeus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_subdirectory(dependencies/csv-parser)


add_library(Coeus SHARED
	include/core.h
	include/coeus.h
	include/neuron.h
	include/csv.h

	src/csv.cpp
	src/coeus.cpp
)

target_include_directories(Coeus PUBLIC dependencies/eigen)

add_executable(Coeus-App src/main.cpp)
#target_link_libraries(Coeus csv)
target_link_libraries(Coeus-App Coeus)

#message(STATUS "Adding PCHs")
#target_precompile_headers(Coeus PRIVATE include/pch.h)