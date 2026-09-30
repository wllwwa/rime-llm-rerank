# rime-llm-rerank Linux/fcitx5 构建（2026-10-01 移植新增）。
#
# 作为 librime 的插件目录使用：librime 的 plugins/CMakeLists.txt 对
# plugins/ 下每个子目录 add_subdirectory，并按下列 PARENT_SCOPE 约定
# 包装出 librime-llm.so 安装到 /usr/lib/rime-plugins/：
#   plugin_name   = rime-llm        → 产物 librime-llm.so
#   plugin_objs   = 本目录对象库     （rime_llm.cpp + llm_module.cc + 内嵌 Lua）
#   plugin_deps   = llama.cpp 静态库 + ${rime_library}
#   plugin_modules= "llm"           → librime 模块名（空注册，见 cpp/llm_module.cc）
# 产物同时导出 luaopen_rime_llm，由 librime-lua 侧 require/loadlib 加载。
# Windows 构建仍用 cpp/CMakeLists.txt（独立 DLL），与本文件互不相干。
cmake_minimum_required(VERSION 3.10)
project(rime-llm LANGUAGES C CXX)

set(LLAMA_ROOT "$ENV{HOME}/Desktop/Github/llama.cpp" CACHE PATH
    "llama.cpp 根目录（需已构建 build/src/libllama.a 等静态库）")

foreach(_lib src/libllama.a
             ggml/src/libggml.a
             ggml/src/libggml-base.a
             ggml/src/libggml-cpu.a)
  if(NOT EXISTS "${LLAMA_ROOT}/build/${_lib}")
    message(FATAL_ERROR
      "llama.cpp 静态库缺失: ${LLAMA_ROOT}/build/${_lib}\n"
      "请先构建 llama.cpp（CPU）:\n"
      "  cmake -B build -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF\n"
      "  cmake --build build --target llama")
  endif()
endforeach()

find_package(Threads REQUIRED)
find_package(OpenMP QUIET)

add_library(rime-llm-objs OBJECT
  cpp/rime_llm.cpp
  cpp/llm_module.cc)

# 内嵌 Lua 5.4（与上游 cpp/CMakeLists.txt 同一 glob 口径；librime 的插件
# 加载是 RTLD_LOCAL，lua_* 符号不可跨库解析，必须自带一份）
file(GLOB LUA_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lua/l*.c")
list(FILTER LUA_SRCS EXCLUDE REGEX "lua\\.c$")
list(FILTER LUA_SRCS EXCLUDE REGEX "luac\\.c$")
target_sources(rime-llm-objs PRIVATE ${LUA_SRCS})

target_include_directories(rime-llm-objs PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lua"
  "${LLAMA_ROOT}/include"
  "${LLAMA_ROOT}/ggml/include")

set_target_properties(rime-llm-objs PROPERTIES
  C_STANDARD 11
  CXX_STANDARD 17
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(rime-llm-objs PRIVATE -O2)

add_library(llama_ext STATIC IMPORTED GLOBAL)
set_target_properties(llama_ext PROPERTIES
  IMPORTED_LOCATION "${LLAMA_ROOT}/build/src/libllama.a")
add_library(ggml_ext STATIC IMPORTED GLOBAL)
set_target_properties(ggml_ext PROPERTIES
  IMPORTED_LOCATION "${LLAMA_ROOT}/build/ggml/src/libggml.a")
add_library(ggml_base_ext STATIC IMPORTED GLOBAL)
set_target_properties(ggml_base_ext PROPERTIES
  IMPORTED_LOCATION "${LLAMA_ROOT}/build/ggml/src/libggml-base.a")
add_library(ggml_cpu_ext STATIC IMPORTED GLOBAL)
set_target_properties(ggml_cpu_ext PROPERTIES
  IMPORTED_LOCATION "${LLAMA_ROOT}/build/ggml/src/libggml-cpu.a")

set(plugin_name rime-llm PARENT_SCOPE)
set(plugin_objs $<TARGET_OBJECTS:rime-llm-objs> PARENT_SCOPE)
set(plugin_deps
  llama_ext
  ggml_ext
  ggml_base_ext
  ggml_cpu_ext
  Threads::Threads
  ${OpenMP_CXX_LIBRARIES}
  ${rime_library}
  PARENT_SCOPE)
set(plugin_modules "llm" PARENT_SCOPE)
