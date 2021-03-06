cmake_minimum_required(VERSION 3.13)
project(imgui_gl2)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
    message("OpenGL found")
    message ("OPENGL_LIBRARIES: " ${OPENGL_LIBRARIES})
endif()

find_package(GLUT REQUIRED)
if(GLUT_FOUND)
    message("GLUT found")
    message ("GLUT_glut_LIBRARY: " ${GLUT_glut_LIBRARY})
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)



include_directories(3rd/imgui 3rd/imgui/backends 3rd/implot)
set(ImGui_Glut_src 3rd/imgui/imgui.cpp 3rd/imgui/imgui_demo.cpp 3rd/imgui/imgui_draw.cpp 3rd/imgui/imgui_widgets.cpp
        3rd/imgui/backends/imgui_impl_glut.cpp 3rd/imgui/backends/imgui_impl_opengl2.cpp
        3rd/implot/implot.cpp 3rd/implot/implot_items.cpp)



add_executable(imgui_covariance_demo main.cpp ${ImGui_Glut_src} )
target_link_libraries(imgui_covariance_demo ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} Eigen3::Eigen)

add_executable(ekf_slam ekf_slam.cpp ${ImGui_Glut_src} )
target_link_libraries(ekf_slam ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} Eigen3::Eigen)

add_executable(ekf_localization ekf_localization.cpp ${ImGui_Glut_src} )
target_link_libraries(ekf_localization ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} Eigen3::Eigen)

add_executable(imgui_interpolation_demo interpolation.cpp ${ImGui_Glut_src} )
target_link_libraries(imgui_interpolation_demo ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} Eigen3::Eigen)