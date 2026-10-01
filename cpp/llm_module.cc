// librime 模块注册（空模块，2026-10-01 Linux 移植新增）。
// librime 外部插件加载器按文件名 librime-llm.so → 模块名 "llm" 查找，
// 找不到会在每次启动时告警；注册空模块让扫描静默通过。
// 本库的真实功能是 Lua C 扩展 luaopen_rime_llm，由 librime-lua 侧
// require/loadlib 加载，不经 librime 模块机制（Windows 构建不走本文件）。
//
// 这里**刻意不挂 finalize**：librime 会在一个进程里走多轮 initialize/finalize
// （fcitx5 启动期的部署/重载就会），任何"在 finalize 里停掉后台线程"的做法都会
// 在下一轮里静默失效——真机踩过：预解码 worker 被停掉后再也没起来，31 次预习
// 请求全部作废。预习 worker 因此设计为随进程结束而消失（同上游 ctxlink 线程）。
#include <rime_api.h>

static void rime_llm_initialize() {}
static void rime_llm_finalize() {}

RIME_REGISTER_MODULE(llm)
