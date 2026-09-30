// librime 模块注册（空模块，2026-10-01 Linux 移植新增）。
// librime 外部插件加载器按文件名 librime-llm.so → 模块名 "llm" 查找，
// 找不到会在每次启动时告警；注册空模块让扫描静默通过。
// 本库的真实功能是 Lua C 扩展 luaopen_rime_llm，由 librime-lua 侧
// require/loadlib 加载，不经 librime 模块机制（Windows 构建不走本文件）。
#include <rime_api.h>

// 定义在 rime_llm.cpp：停掉预习 worker 并 join。
// finalize 是唯一能覆盖「fcitx5 退出」与「lucky_bench 退出」的公共时机——worker
// 若正持锁解码而进程在此刻收尾，就是 use-after-free。原实现每次 prepare 起一个
// detached 线程、无人回收，离线基准的退出阶段实测偶发 core dump。
void rime_llm_shutdown();

static void rime_llm_initialize() {}
static void rime_llm_finalize() { rime_llm_shutdown(); }

RIME_REGISTER_MODULE(llm)
