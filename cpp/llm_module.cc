// librime 模块注册（空模块，2026-10-01 Linux 移植新增）。
// librime 外部插件加载器按文件名 librime-llm.so → 模块名 "llm" 查找，
// 找不到会在每次启动时告警；注册空模块让扫描静默通过。
// 本库的真实功能是 Lua C 扩展 luaopen_rime_llm，由 librime-lua 侧
// require/loadlib 加载，不经 librime 模块机制（Windows 构建不走本文件）。
#include <rime_api.h>

static void rime_llm_initialize() {}
static void rime_llm_finalize() {}

RIME_REGISTER_MODULE(llm)
