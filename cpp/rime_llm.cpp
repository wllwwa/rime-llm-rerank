/*
 * rime_llm.cpp — RIME LLM 候选重排 C++ 插件 (CPU)
 * 编译: cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
 *        cmake --build build --config Release
 * Lua:  require("rime_llm") → llm.score(ctx, cands)
 */

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <objbase.h>
#include <oleauto.h>
// UIA 客户端接口（IUIAutomationTextPattern 等）在 Client.h，TextUnit 枚举在 Core.h
#include <UIAutomationClient.h>
#include <UIAutomationCore.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#pragma comment(lib, "UIAutomationCore.lib")
#else
// Linux/fcitx5 移植（2026-10-01）：无 COM/UIA，仅需 POSIX 文件 API
#include <sys/stat.h>
#include <cstdio>
#include <time.h>   // 日志时间戳: clock_gettime / localtime_r（2026-10-01 加）
#endif

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

#include "llama.h"

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <algorithm>
#include <regex>

// ============================================================
// 配置默认值
// ============================================================
// 默认模型路径 = RIME 用户目录根\Qwen...gguf（2026-08-31 用户澄清定案：
// "用户文件夹"= 小狼毫右键的用户文件夹 %APPDATA%\Rime——方案配置所在
// 处，模型直接放根目录、不套子文件夹；8-27 曾误用 %USERPROFILE%\
// gguf_models\。自定义位置用 schema model_path 显式指向。便携模式小狼毫
// 的用户目录不在 %APPDATA%\Rime，此类部署需显式配置 model_path）
static std::string default_model_path() {
#ifdef _WIN32
  const char *ad = getenv("APPDATA");
  if (ad && *ad)
    return std::string(ad) + "\\Rime\\Qwen3.5-0.8B-Q4_K_M.gguf";
#else
  // Linux：默认 RIME 用户目录根（fcitx5 = $XDG_DATA_HOME/fcitx5/rime）
  const char *xdg = getenv("XDG_DATA_HOME");
  if (xdg && *xdg)
    return std::string(xdg) + "/fcitx5/rime/Qwen3.5-0.8B-Q4_K_M.gguf";
  const char *home = getenv("HOME");
  if (home && *home)
    return std::string(home) + "/.local/share/fcitx5/rime/Qwen3.5-0.8B-Q4_K_M.gguf";
#endif
  return "Qwen3.5-0.8B-Q4_K_M.gguf";
}
static std::string  g_model_path      = default_model_path();
static int          g_min_tokens      = 1;
static int          g_max_ctx_tokens  = 10; // tok=10 准确率 93.4%，10→17 收益仅 +1.1pp 但延迟翻倍
static int          g_n_threads       = 4;  // 默认=GGML 默认; 可用 bench_threads 实测后配置
static int          g_n_ctx           = 128; // KV: 11 seqs × (ctx 10 + cand 2) = 132, 64 溢出
                                              //（2026-09-11 对齐源码版 llm_filter.cc 同款修正：
                                              //  64 时 score 的 S2 恒 ~100ms vs 128 的 ~40ms，
                                              //  真机 DLL 日志实证）
static int          g_n_seq_max       = 12;  // 模板 seq 0 + 最多 11 worker seq
// GPU offload 层数（2026-10-02，见 notes/tests/LLM重排GPU后端门槛测试报告.md）：
// -1 = 有可用 GPU 时全部层进显存（llama.cpp 默认）；0 = 纯 CPU；N = 前 N 层。
// 只在模型加载时生效，改动走 unload/reload（同 n_threads）。
static int          g_n_gpu_layers    = -1;

// CE 位置权重（2026-09-16 采纳，首项归一规范形；与 (1.15,1.30,0.70) 排序等价、
// ce2 判别力最强（同时携带上下文契合 + 词内连贯），ce3 已深入词内部、上下文
// 影响衰减属噪声源降权；4+ token 词只算前三项——外推证伪删除（λ 0→1 端到端
// 命中率单调递减，λ=0 最优；旧 λ=0.6 标定用"与全量 CE 一致率"口径，与端
// 指标分道扬镳）。胜出变体 conf 91.82→92.13（+0.31pp，平台非尖峰）。
static constexpr double kCeW1 = 1.0, kCeW2 = 1.13, kCeW3 = 0.61;

// ============================================================
// 模型状态
// ============================================================
static llama_model        * g_model   = nullptr;
static llama_context      * g_ctx     = nullptr;
static const llama_vocab  * g_vocab   = nullptr;
static std::mutex           g_mutex;
static std::atomic<bool>    g_loaded{false};
static std::atomic<bool>    g_loading{false};
static std::vector<double>      g_last_scores;
static std::vector<std::string> g_last_cands;

// ============================================================
// 预解码状态：prepare() 在 commit 后异步执行 Step 1 + KV copy
// score() 检测 ctx 一致时直接跳到 Step 2
// ============================================================
static std::vector<llama_token> g_prep_ctx;     // 预解码的上下文 token
static std::vector<float>       g_prep_logits;  // ctx_last logits
static bool                      g_prep_ready = false; // prepare() 已完成
static std::atomic<int>         g_prep_seq{0};  // 请求序列号，跳过过期请求
static long                      g_seq0_gen = 0; // seq0 KV 代次: 任何覆盖 seq0 的 decode 递增
static long                      g_prep_gen = 0; // prepare 完成时的代次

// ---- 预习请求调度（2026-10-01 重做）----------------------------------------
// 原实现：每次 prepare 起一个 detached 线程，抢不到 g_mutex 就**直接丢弃**该请求
// （log "SKIP score in progress"）。实测下来被丢弃的那个 ctx 恰恰是紧接着 score
// 需要的：真人日志里 SKIP 占预习请求的 15%，被丢弃后 score 只能自己同步解码
// （S1=70~90ms）。离线基准用 burst 语料复现得更极端：丢弃率 66%、87% 的评分走
// 同步 S1。
// 现实现：**单个常驻 worker + 只保留最新请求的槽 + 代次校验**，worker 抢不到锁
// 就退避重试而不是丢弃（旧请求被新请求取代才算作废，这是正确语义）。
// 另外给 score 让路：worker 在"即将开始一次长 decode"前检查有无 score 在等锁/
// 持锁，有就让路——score 是人等着的路径，预习只是投机，绝不能让新调度器给
// score 加延迟（这是必须守住的约束）。
// 这三个**故意泄漏**（绑定到 new 出来的对象，不是命名空间作用域对象）：常驻 worker
// 会永久停在 g_prep_cv 上等新请求，而 glibc 的 pthread_cond_destroy 会先广播、再
// **等所有等待者离开**才返回——被唤醒的 worker 一看 g_pending_valid 仍为假，又回去
// 等，于是 exit() 里的静态析构永远等不到它，进程挂死（离线基准实测：日志停在最后
// 一条评分，进程数分钟不退出；真机上等价于 `fcitx5 -r` 卡住）。
// 泄漏掉就根本没有析构，退出路径不再碰这把锁/这个 cv；worker 需要的内存也全都有效，
// 不会出现"线程还在跑、对象已被回收"的静态析构顺序问题。
static std::mutex               &g_prep_mu = *new std::mutex;        // 保护待预习槽
static std::condition_variable  &g_prep_cv = *new std::condition_variable;
static std::vector<llama_token> &g_pending_ctx = *new std::vector<llama_token>; // 最新待预习 ctx（覆盖式）
static std::atomic<int>         g_pending_seq{0}; // 槽内 ctx 对应的请求号
static bool                      g_pending_valid = false;
static std::atomic<bool>        g_prep_worker_started{false};
static std::atomic<int>         g_score_waiting{0}; // 正在等锁的 score 数
static std::atomic<int>         g_score_active{0};  // 持锁执行的 score 数
static std::atomic<int>         g_prep_superseded{0}; // 被更新取代的请求数（诊断）

// ============================================================
// 轻量日志
// 目录: Lua 侧设置 log_dir (RIME 用户目录); 未设置则回退 %TEMP%
// ============================================================
static std::string g_log_dir;
// 日志总开关（2026-10-02）：配置面是 schema 的 llm_rerank/log_enabled，
// 由 lua 经 cpp.log_enabled 传进来（与 log_dir 同路，见 lua_newindex）。
// 默认 true = 与加开关前的行为完全一致；置 false 后所有 log_msg
// （prepare/score/上下文探测等）直接返回，连 fopen 都不做。
// lua 侧那份 rime_llm_events.txt / plog 行由同一个配置项在 lua 自己的
// 判断里关掉（见 llm_filter.lua / llm_processor.lua）。
static std::atomic<bool> g_log_enabled{true};

// 本地时间戳 "HH:MM:SS.mmm"（2026-10-01 加：原日志无时间，
// 无法与 events 文件对齐，也无法还原 prepare/score 的先后时序）
static void now_stamp(char * out, size_t n) {
#ifdef _WIN32
    SYSTEMTIME st;
    GetLocalTime(&st);
    snprintf(out, n, "%02d:%02d:%02d.%03d",
             st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tmv;
    localtime_r(&ts.tv_sec, &tmv);
    snprintf(out, n, "%02d:%02d:%02d.%03d",
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec,
             (int)(ts.tv_nsec / 1000000L));
#endif
}

static void log_msg(const char * fmt, ...) {
    // 总开关：关掉后连 vsnprintf/fopen 都不做（llm_rerank/log_enabled）
    if (!g_log_enabled.load(std::memory_order_relaxed)) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
#ifdef _WIN32
    char path[MAX_PATH];
    if (!g_log_dir.empty() && g_log_dir.size() < MAX_PATH - 32) {
        strcpy_s(path, g_log_dir.c_str());
        strcat_s(path, "\\rime_llm_log.txt");
    } else {
        GetTempPathA(sizeof(path), path);
        strcat_s(path, sizeof(path), "rime_llm_log.txt");
    }
    // 日志轮转（2026-09-04）：>5MB 改名 .old（单文件轮转）防无限增长；
    // 改名失败（被占等罕见）则继续追加，不丢日志
    WIN32_FILE_ATTRIBUTE_DATA fa;
    if (GetFileAttributesExA(path, GetFileExInfoStandard, &fa) &&
        ((((unsigned long long)fa.nFileSizeHigh << 32) |
          (unsigned long long)fa.nFileSizeLow) > 5ull * 1024 * 1024)) {
        std::string oldp = std::string(path) + ".old";
        DeleteFileA(oldp.c_str());
        MoveFileA(path, oldp.c_str());
    }
#else
    char path[512];
    if (!g_log_dir.empty() && g_log_dir.size() < sizeof(path) - 32)
        snprintf(path, sizeof(path), "%s/rime_llm_log.txt", g_log_dir.c_str());
    else
        snprintf(path, sizeof(path), "/tmp/rime_llm_log.txt");
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 5 * 1024 * 1024) {
        std::string oldp = std::string(path) + ".old";
        remove(oldp.c_str());
        rename(path, oldp.c_str());
    }
#endif
    FILE * f = fopen(path, "a");
    if (f) {
        char tbuf[32];
        now_stamp(tbuf, sizeof(tbuf));
        fprintf(f, "%s %s\n", tbuf, buf);
        fclose(f);
    }
}


// ============================================================
// 光标上文旁路（2026-09-10 插件版全功能：三层上文的前两层）
//   COM = Word/WPS 文字文档模型（前台 OpusApp 门控；Kwps→Word 双 ProGID，
//         WPS 双注册 Word CLSID 但以文档打开为条件，Kwps 优先无歧义）
//   UIA = UIA TextPattern 焦点元素前文（通用现代应用：Chromium 系/记事本
//         实测 8-13ms；读的是"当时焦点元素"，消费时校验前台进程一致）
//   历史 = lua 侧 commit_history 兜底（llm_processor.get_context）
// 实验依据与真机数据：memory wps-context-investigation.md 十一/十二节
//（探针 scripts/probe_wps_ctx.cpp / probe_uia_ctx.cpp，源码版 llm_filter.cc
//  comctx 同源实现已真机验证）。单后台线程 STA：2s 周期 + kick（commit
// 后 / 编辑键 / 点击，kick 唤醒统一延迟 ~50ms 再读）刷新；编辑键/点击
// 经 edit_reset 先失效再 kick（2026-09-12 信号层统一——快照制对非
// commit 编辑事件原本是盲区，旧快照 2.5s 新鲜窗内冒充真文）。
// lua 调用线程经 mutex 快照消费，COM/UIA 调用绝不在按键同步路径。
// ============================================================
#ifdef _WIN32

namespace ctxlink {

static std::mutex g_mu;
static std::string g_com_text;                 // UTF-8
static unsigned long long g_com_stamp = 0;
static std::wstring g_com_title;               // 发布时的前台标题（消费端指纹）
static std::string g_uia_text;                 // UTF-8
static unsigned long long g_uia_stamp = 0;
static DWORD g_uia_pid = 0;                    // 快照来源焦点元素进程
static std::wstring g_uia_title;               // 发布时的前台标题（消费端指纹）

static std::mutex g_cv_mu;
static std::condition_variable g_cv;
static std::atomic<bool> g_kick{false};
static std::atomic<bool> g_started{false};
static std::atomic<unsigned long long> g_last_kick_ms{0};  // 打字活跃期判定

// 前台 Office 判定（2026-09-11 扩展演示）：OpusApp 同属 MS Word（开发代号
// Opus）与 WPS 文字（复刻 Word 窗口体系）；PP12FrameClass 同属 MS
// PowerPoint（2010+）与 WPS 演示（实测 12.1.0.28505）→ 两个类名圈定
// "文字处理器 + 演示文档"。表格 XLMAIN（复刻 Excel）刻意不含——单元格
// 编辑态 COM 盲区（Office 系通病：编辑态对象模型挂起、无 Selection.Start
// 等价物），输入法打字恰在盲区，落 UIA/历史兜底。候选窗不抢前台。
enum class OfficeKind { NONE, WRITER, PPT };
static OfficeKind foreground_office() {
    HWND fg = GetForegroundWindow();
    if (!fg) return OfficeKind::NONE;
    wchar_t cls[64];
    if (GetClassNameW(fg, cls, 64) <= 0) return OfficeKind::NONE;
    if (wcscmp(cls, L"OpusApp") == 0) return OfficeKind::WRITER;
    if (wcscmp(cls, L"PP12FrameClass") == 0) return OfficeKind::PPT;
    return OfficeKind::NONE;
}
static DWORD foreground_pid() {
    HWND fg = GetForegroundWindow();
    DWORD pid = 0;
    if (fg) GetWindowThreadProcessId(fg, &pid);
    return pid;
}

// ---- COM 读链（源码版 llm_filter.cc comctx 同源）----
static HRESULT disp_invoke(IDispatch * obj, const wchar_t * name, WORD flags,
                           DISPPARAMS * pdp, VARIANT * ret) {
    DISPID dispid = 0;
    HRESULT hr = obj->GetIDsOfNames(IID_NULL, (LPOLESTR *)&name, 1,
                                    LOCALE_USER_DEFAULT, &dispid);
    if (FAILED(hr)) return hr;
    VariantInit(ret);
    return obj->Invoke(dispid, IID_NULL, LOCALE_USER_DEFAULT, flags, pdp,
                       ret, nullptr, nullptr);
}
static HRESULT disp_get(IDispatch * obj, const wchar_t * name, VARIANT * ret) {
    DISPPARAMS dp;
    ZeroMemory(&dp, sizeof(dp));
    return disp_invoke(obj, name, DISPATCH_PROPERTYGET, &dp, ret);
}

// BSTR → UTF-8（COM 两读链与 UIA 读链共用；失败时 out 不变返回 false）
static bool bstr_to_utf8(BSTR s, std::string & out) {
    if (!s) return false;
    int n = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0,
                                nullptr, nullptr);
    if (n <= 0) return false;
    out.resize(n - 1);
    if (n > 1)
        WideCharToMultiByte(CP_UTF8, 0, s, -1, &out[0], n,
                            nullptr, nullptr);
    return true;
}

static void wlower(wchar_t * s) {
    for (; *s; ++s)
        if (*s >= L'A' && *s <= L'Z') *s = *s + 32;
}

// 文档名 ⊂ 前台标题校验（WPS 统一标签页防串档，2026-09-11 真机两轮定案）：
// WPS 多组件标签共用单一框架窗口，前台类名恒为 OpusApp（实测演示标签激活
// 仍 OpusApp、标题变"演示文稿1 - WPS Office"）——类名门控区分不了活动
// 组件。帧标题随活动标签文档名变，组件模型的 ActiveDocument.Name /
// ActivePresentation.Name 固定指本组件自己的活动文档：名字（全名与去
// 扩展名两种形态）都不在前台标题中 = 活动标签是另一组件（或前台是微软
// Office 而后台 WPS 同开），本组件读出的文本不属于当前光标处，判
// TAB_IDLE 落下一层。锚点不可用 Window.Caption——实测带修改标记尾缀
// " *" 且无扩展名剥不掉（"文字文稿1 *" ⊄ "文字文稿1 - WPS Office"）恒
// 误杀。名字取不到/去扩展名后过短（单字符易误配）→ 保守放行。
static bool name_matches_fg_title(const VARIANT & vname) {
    if (vname.vt != VT_BSTR || !vname.bstrVal || !*vname.bstrVal)
        return true;
    wchar_t full[128] = {0}, stem[128] = {0};
    const wchar_t * base = wcsrchr(vname.bstrVal, L'\\');
    base = base ? base + 1 : vname.bstrVal;
    wcsncpy_s(full, base, _TRUNCATE);
    wcsncpy_s(stem, base, _TRUNCATE);
    wchar_t * dot = wcsrchr(stem, L'.');
    if (dot && dot != stem) *dot = 0;
    if (wcslen(stem) < 2) return true;
    HWND fg = GetForegroundWindow();
    wchar_t title[256] = {0};
    if (!fg || GetWindowTextW(fg, title, 256) <= 0)
        return true;  // 标题取不到（罕见），保守放行
    wlower(full);
    wlower(stem);
    wlower(title);
    return wcsstr(title, full) != nullptr || wcsstr(title, stem) != nullptr;
}

enum class ComRead { OK, NO_DOC, DEAD, TAB_IDLE };

// 活动标签判定（不依赖文件名，2026-09-11 三轮真机定案）：文字组件视图
// 窗口（_WwB 类，ActiveWindow.Hwnd）的根窗口 == 前台窗口 && 视图可见
// = 文字标签/文字窗口活动。统一标签页：视图随活动标签显隐（实测即时
// 翻转）；多独立窗口：视图都可见，但只有活动窗口的根 == 前台（他窗/
// 他应用前台时根不等——微软 Word 前台场景由此天然排除，无需名字）。
// 句柄取不到 = UNKNOWN（回落名字校验兜底）。
enum class TabActive { YES, NO, UNKNOWN };
static TabActive writer_tab_active(IDispatch * app_w) {
    if (!app_w) return TabActive::UNKNOWN;
    VARIANT vwin, vh;
    VariantInit(&vwin); VariantInit(&vh);
    long long h = 0;
    if (SUCCEEDED(disp_get(app_w, L"ActiveWindow", &vwin)) &&
        vwin.vt == VT_DISPATCH && vwin.pdispVal &&
        SUCCEEDED(disp_get(vwin.pdispVal, L"Hwnd", &vh))) {
        if (vh.vt == VT_I4)      h = vh.lVal;
        else if (vh.vt == VT_I8) h = vh.llVal;
        else if (vh.vt == VT_R8) h = (long long)vh.dblVal;
    }
    VariantClear(&vwin); VariantClear(&vh);
    if (!h) return TabActive::UNKNOWN;
    HWND root = GetAncestor((HWND)(LONG_PTR)h, GA_ROOT);
    HWND fg = GetForegroundWindow();
    if (!root || !fg) return TabActive::UNKNOWN;
    if (root != fg) return TabActive::NO;
    return IsWindowVisible((HWND)(LONG_PTR)h) ? TabActive::YES : TabActive::NO;
}

static ComRead com_read_chain(IDispatch * app, std::string & out,
                              TabActive wta) {
    VARIANT vwin, vsel, vstart, vdoc, vrng, vtxt, vname;
    VariantInit(&vwin); VariantInit(&vsel); VariantInit(&vstart);
    VariantInit(&vdoc); VariantInit(&vrng); VariantInit(&vtxt);
    VariantInit(&vname);
    auto vt_ok = [](const VARIANT & v) {
        return v.vt == VT_DISPATCH && v.pdispVal;
    };
    HRESULT hr;
    auto cleanup = [&]() {
        VariantClear(&vwin); VariantClear(&vsel); VariantClear(&vstart);
        VariantClear(&vdoc); VariantClear(&vrng); VariantClear(&vtxt);
        VariantClear(&vname);
    };
    // 串档拦截①：视图窗口判定非活动（统一标签页切走/他窗前台）——
    // 不依赖文件名，同基名文档也能正确区分
    if (wta == TabActive::NO) {
        cleanup();
        return ComRead::TAB_IDLE;
    }
    if (FAILED(hr = disp_get(app, L"ActiveWindow", &vwin)) || !vt_ok(vwin)) {
        cleanup();
        if (hr == DISP_E_EXCEPTION || hr == DISP_E_MEMBERNOTFOUND)
            return ComRead::NO_DOC;
        return ComRead::DEAD;
    }
    // ActiveDocument 前移：名字校验（串档拦截②兜底）与下方 Range 共用
    if (FAILED(hr = disp_get(app, L"ActiveDocument", &vdoc)) ||
        !vt_ok(vdoc)) {
        cleanup();
        if (hr == DISP_E_EXCEPTION || hr == DISP_E_MEMBERNOTFOUND)
            return ComRead::NO_DOC;
        return ComRead::DEAD;
    }
    if (wta == TabActive::UNKNOWN) {
        // 视图句柄取不到时回落名字校验（YES 时有 root==fg 硬判据，跳过
        // 免受标题格式差异影响——Caption 轮误杀的教训）
        disp_get(vdoc.pdispVal, L"Name", &vname);
        if (!name_matches_fg_title(vname)) {
            cleanup();
            return ComRead::TAB_IDLE;
        }
    }
    if (FAILED(hr = disp_get(vwin.pdispVal, L"Selection", &vsel)) ||
        !vt_ok(vsel) ||
        FAILED(hr = disp_get(vsel.pdispVal, L"Start", &vstart)) ||
        vstart.vt != VT_I4) {
        cleanup();
        if (hr == DISP_E_EXCEPTION || hr == DISP_E_MEMBERNOTFOUND)
            return ComRead::NO_DOC;
        return ComRead::DEAD;
    }
    long pos = vstart.lVal, from = pos - 64;
    if (from < 0) from = 0;
    ComRead res = ComRead::OK;
    do {
        VARIANT args[2];  // rgvark 反序：末参数 (pos) 在前
        VariantInit(&args[0]); args[0].vt = VT_I4; args[0].lVal = pos;
        VariantInit(&args[1]); args[1].vt = VT_I4; args[1].lVal = from;
        DISPPARAMS dp;
        ZeroMemory(&dp, sizeof(dp));
        dp.rgvarg = args; dp.cArgs = 2;
        if (FAILED(hr = disp_invoke(vdoc.pdispVal, L"Range",
                                    DISPATCH_METHOD | DISPATCH_PROPERTYGET,
                                    &dp, &vrng)) || !vt_ok(vrng)) {
            res = (hr == DISP_E_EXCEPTION) ? ComRead::NO_DOC : ComRead::DEAD;
            break;
        }
        if (FAILED(hr = disp_get(vrng.pdispVal, L"Text", &vtxt)) ||
            vtxt.vt != VT_BSTR) {
            res = (hr == DISP_E_EXCEPTION) ? ComRead::NO_DOC : ComRead::DEAD;
            break;
        }
        bstr_to_utf8(vtxt.bstrVal, out);
    } while (false);
    cleanup();
    return res;
}

// ---- 演示读链（WPP/PowerPoint，2026-09-11 本机实测验证）----
// app.ActiveWindow.Selection.Type==3(文本编辑) 时：start = TextRange.Start
//（框内 1-based 偏移）；前文 = ShapeRange.TextFrame.TextRange.Characters(
// max(1,start-64), min(64, start-1)).Text。Type!=3 → NO_DOC 语义
//（无文本编辑：形状/幻灯片选择态）。表格 XLMAIN 不在此链（编辑态盲区）。
static ComRead ppt_read_chain(IDispatch * app, std::string & out,
                              TabActive wta) {
    VARIANT vwin, vsel, vtype, vstart, vshape, vframe, vrng, vtxt;
    VARIANT vpres, vname;
    VariantInit(&vwin); VariantInit(&vsel); VariantInit(&vtype);
    VariantInit(&vstart); VariantInit(&vshape); VariantInit(&vframe);
    VariantInit(&vrng); VariantInit(&vtxt);
    VariantInit(&vpres); VariantInit(&vname);
    auto vt_ok = [](const VARIANT & v) {
        return v.vt == VT_DISPATCH && v.pdispVal;
    };
    HRESULT hr;
    auto cleanup = [&]() {
        VariantClear(&vwin); VariantClear(&vsel); VariantClear(&vtype);
        VariantClear(&vstart); VariantClear(&vshape); VariantClear(&vframe);
        VariantClear(&vrng); VariantClear(&vtxt);
        VariantClear(&vpres); VariantClear(&vname);
    };
    // 串档拦截①：文字视图判定活动 = 活动标签必是文字（演示/表格都不是）
    if (wta == TabActive::YES) {
        cleanup();
        return ComRead::TAB_IDLE;
    }
    if (FAILED(hr = disp_get(app, L"ActiveWindow", &vwin)) || !vt_ok(vwin)) {
        cleanup();
        if (hr == DISP_E_EXCEPTION) return ComRead::NO_DOC;
        return ComRead::DEAD;
    }
    // 串档拦截②：演示无窗口句柄（WPP Window.Hwnd/HWND 均空，实测），
    // 名字校验是演示链主判据——挡表格标签/微软 Office 前台；同基名文档
    // 场景由拦截①的 wta 覆盖
    if (FAILED(hr = disp_get(app, L"ActivePresentation", &vpres)) ||
        !vt_ok(vpres)) {
        cleanup();
        if (hr == DISP_E_EXCEPTION) return ComRead::NO_DOC;
        return ComRead::DEAD;
    }
    disp_get(vpres.pdispVal, L"Name", &vname);
    if (!name_matches_fg_title(vname)) {
        cleanup();
        return ComRead::TAB_IDLE;
    }
    if (FAILED(hr = disp_get(vwin.pdispVal, L"Selection", &vsel)) ||
        !vt_ok(vsel) ||
        FAILED(hr = disp_get(vsel.pdispVal, L"Type", &vtype)) ||
        vtype.vt != VT_I4) {
        cleanup();
        if (hr == DISP_E_EXCEPTION) return ComRead::NO_DOC;
        return ComRead::DEAD;
    }
    if (vtype.lVal != 3) {  // ppSelectionText = 3：非文本编辑态
        cleanup();
        return ComRead::NO_DOC;
    }
    ComRead res = ComRead::OK;
    do {
        if (FAILED(hr = disp_get(vsel.pdispVal, L"TextRange", &vrng)) ||
            !vt_ok(vrng) ||
            FAILED(hr = disp_get(vrng.pdispVal, L"Start", &vstart)) ||
            vstart.vt != VT_I4) {
            res = (hr == DISP_E_EXCEPTION) ? ComRead::NO_DOC : ComRead::DEAD;
            break;
        }
        long start = vstart.lVal;               // 1-based
        long len = start - 1 > 64 ? 64 : start - 1;  // 前文字数（截 64）
        if (len <= 0) { out.clear(); break; }   // 光标在框首：前文真空
        if (FAILED(hr = disp_get(vsel.pdispVal, L"ShapeRange", &vshape)) ||
            !vt_ok(vshape) ||
            FAILED(hr = disp_get(vshape.pdispVal, L"TextFrame", &vframe)) ||
            !vt_ok(vframe)) {
            res = (hr == DISP_E_EXCEPTION) ? ComRead::NO_DOC : ComRead::DEAD;
            break;
        }
        VARIANT vfull;
        VariantInit(&vfull);
        if (FAILED(hr = disp_get(vframe.pdispVal, L"TextRange", &vfull)) ||
            !vt_ok(vfull)) {
            VariantClear(&vfull);
            res = (hr == DISP_E_EXCEPTION) ? ComRead::NO_DOC : ComRead::DEAD;
            break;
        }
        // Characters(start_index, length)：rgvark 反序 [length, start_index]
        VARIANT args[2];
        VariantInit(&args[0]); args[0].vt = VT_I4; args[0].lVal = len;
        VariantInit(&args[1]); args[1].vt = VT_I4;
        args[1].lVal = start - len;             // 起始字符 = start-len（含尾部）
        DISPPARAMS dp;
        ZeroMemory(&dp, sizeof(dp));
        dp.rgvarg = args; dp.cArgs = 2;
        VariantClear(&vrng);  // 复用作出参：先释放首个 TextRange 引用
        if (FAILED(hr = disp_invoke(vfull.pdispVal, L"Characters",
                                    DISPATCH_METHOD | DISPATCH_PROPERTYGET,
                                    &dp, &vrng)) || !vt_ok(vrng)) {
            VariantClear(&vfull);
            res = (hr == DISP_E_EXCEPTION) ? ComRead::NO_DOC : ComRead::DEAD;
            break;
        }
        VariantClear(&vfull);
        if (FAILED(hr = disp_get(vrng.pdispVal, L"Text", &vtxt)) ||
            vtxt.vt != VT_BSTR) {
            res = (hr == DISP_E_EXCEPTION) ? ComRead::NO_DOC : ComRead::DEAD;
            break;
        }
        bstr_to_utf8(vtxt.bstrVal, out);
    } while (false);
    cleanup();
    return res;
}

// ---- UIA 读链（2026-09-11 瘦身：~10 个跨进程往返 → 6 个）----
// 省法：①GetPattern 失败即无 TextPattern（去掉 IsTextPatternAvailable
// 属性往返）；②selection range 直接 Clone 后把 Start 端点前移 64 字
//（数学上等价于 [doc.Start, sel.Start) 截尾 64——去掉 DocumentRange +
// 2×MoveEndpointByRange 共 3 个往返）。每次跨进程往返 1-3ms 且打断
// 目标应用 UI 线程，读链瘦身同时降两边开销（真机 A/B：ctxlink 活跃
// 期 score 均值 +36%，2026-09-11 排查）。
static bool uia_read(IUIAutomation * uia, std::string & out, DWORD & pid) {
    IUIAutomationElement * el = nullptr;
    if (FAILED(uia->GetFocusedElement(&el)) || !el)
        return false;
    bool ret = false;
    int elpid = 0;
    el->get_CurrentProcessId(&elpid);
    IUIAutomationTextPattern * tp = nullptr;
    if (SUCCEEDED(el->GetCurrentPattern(UIA_TextPatternId,
                                        (IUnknown **)&tp)) && tp) {
        IUIAutomationTextRangeArray * sel = nullptr;
        if (SUCCEEDED(tp->GetSelection(&sel)) && sel) {
            IUIAutomationTextRange * s0 = nullptr;
            int nsel = 0;
            sel->get_Length(&nsel);
            if (nsel > 0) sel->GetElement(0, &s0);
            sel->Release();
            if (s0) {
                IUIAutomationTextRange * r = nullptr;
                if (SUCCEEDED(s0->Clone(&r))) {
                    // r = [光标(或选区起点)-64 字, 光标]：Start 端点前移
                    int moved = 0;
                    r->MoveEndpointByUnit(TextPatternRangeEndpoint_Start,
                                          TextUnit_Character, -64, &moved);
                    BSTR txt = nullptr;
                    if (SUCCEEDED(r->GetText(-1, &txt)) && txt) {
                        ret = bstr_to_utf8(txt, out);
                        SysFreeString(txt);
                    }
                    r->Release();
                }
                s0->Release();
            }
        }
        tp->Release();
    }
    if (ret && elpid) pid = (DWORD)elpid;
    el->Release();
    return ret;
}

static void thread_proc() {
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    IUIAutomation * uia = nullptr;
    bool uia_ok = SUCCEEDED(CoCreateInstance(CLSID_CUIAutomation, nullptr,
                                             CLSCTX_INPROC_SERVER,
                                             IID_IUIAutomation,
                                             (void **)&uia));
    // 仅 WPS 系 ProgID（2026-09-11 定案）：COM 旁路只补 TSF/UIA 墙上唯一
    // 缺口=WPS（微软 Word/PPT 的 TSF/UIA 完整，无需 COM）。类名二义由
    // 附着成败天然消解：Kwps 附着成功=WPS 文字（用 COM）；失败=微软 Word
    // 或无文档（落 UIA/历史）——Word/PPT 的 ProgID 回落已撤（共存双注册
    // 误附风险随之归零）。
    CLSID clsid_kwps, clsid_kwpp;
    bool has_kwps = SUCCEEDED(CLSIDFromProgID(L"Kwps.Application",
                                              &clsid_kwps));
    bool has_kwpp = SUCCEEDED(CLSIDFromProgID(L"KWPP.Application",
                                              &clsid_kwpp));
    if (!uia_ok && !has_kwps && !has_kwpp) {
        CoUninitialize();  // 与 CoInitializeEx 配对（无任何通道，极罕见）
        return;
    }

    // 双组件持久附着 + 探读（2026-09-11 串档修复）：WPS 统一标签页的框架
    // 类名不随活动标签组件变（真机实测演示标签激活仍 OpusApp），旧"前台
    // 类名选单附着"设计失效——文字附着会一直读出文字文档全文冒充演示上
    // 文。改为 Kwps/KWPP 双附着，每轮按"前台类型对位优先、另一组件兜底"
    // 顺序读，由读链标题校验（TAB_IDLE）裁决哪个组件的活动文档才是当前
    // 光标处。DEAD（宿主退出）单独释放该组件下轮懒重附；双双无产出 →
    // 清缓存（消费端落 UIA/历史），未附着时 500ms 快速重试不变。
    IDispatch * app_w = nullptr;  // Kwps.Application（文字）
    IDispatch * app_p = nullptr;  // KWPP.Application（演示）
    bool logged_w = false, logged_p = false;
    bool com_pending = true;  // 前台 Office 且未附着 → 500ms 快速重试
                              //（文档打开瞬间 WPS 才注册 ROT，缩短发现间隙；
                              //  成功即回 2s，开销仅微秒级 ROT 查询）
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g_cv_mu);
            g_cv.wait_for(lk, std::chrono::milliseconds(
                                  g_kick.load() ? 1 : (com_pending ? 500 : 2000)));
        }
        bool woken_by_kick = g_kick.exchange(false);
        com_pending = false;
        // kick 统一延迟 ~50ms 再读（2026-09-12 信号层统一）：编辑键/点击
        // 的 kick 若立即读会赶在应用处理完退格/点击之前拿到旧文本；commit
        // kick 的文档更新同量级。下一键远晚于 100ms（人手速度），统一延迟
        // 无感。连续 kick 在 sleep 期间再次置位 g_kick，下一轮 1ms 即醒，
        // 自然合并不叠加。
        if (woken_by_kick) Sleep(50);

        // COM：前台 Office 才读（读端门控省后台空转；切标签/切窗由发布
        // 标题指纹在消费端即刻拦截，kick/周期兜住换新）
        if (has_kwps || has_kwpp) {
            OfficeKind fg = foreground_office();
            if (fg != OfficeKind::NONE) {
                // 文字附着先行：ActiveWindow.Hwnd 的视图窗口判定是活动
                // 标签主信号（无文字文档时 UNKNOWN 回落名字校验）
                if (has_kwps && !app_w) {
                    IUnknown * punk = nullptr;
                    if (SUCCEEDED(GetActiveObject(clsid_kwps, nullptr,
                                                  &punk)) &&
                        punk) {
                        punk->QueryInterface(IID_IDispatch, (void **)&app_w);
                        punk->Release();
                        if (app_w && !logged_w) {
                            log_msg("ctx probe: com attached (Kwps.Application)");
                            logged_w = true;
                        }
                    }
                }
                TabActive wta = writer_tab_active(app_w);
                if (wta == TabActive::NO && app_w) {
                    // 跨进程标签切换重附（2026-09-11 四轮真机定案）：新建
                    // 文档/演示标签各起新 wps.exe、各持帧窗口，仅活动标签
                    // 所属进程的帧可见——旧附着实例的视图 root≠前台恒 NO。
                    // 实测 GetActiveObject 返回最新激活实例（sees 跟随活动
                    // 标签切换），重附一次即取到活动实例，仍 NO 才真非活动
                    //（表格标签/他应用前台）。GetActiveObject 微秒级，每轮
                    // 至多重附一次。
                    app_w->Release();
                    app_w = nullptr;
                    if (has_kwps) {
                        IUnknown * punk = nullptr;
                        if (SUCCEEDED(GetActiveObject(clsid_kwps, nullptr,
                                                      &punk)) &&
                            punk) {
                            punk->QueryInterface(IID_IDispatch,
                                                 (void **)&app_w);
                            punk->Release();
                        }
                    }
                    wta = writer_tab_active(app_w);
                }
                OfficeKind order[2] = {
                    fg, (fg == OfficeKind::WRITER) ? OfficeKind::PPT
                                                   : OfficeKind::WRITER};
                bool published = false;
                for (int i = 0; i < 2 && !published; i++) {
                    bool is_p = (order[i] == OfficeKind::PPT);
                    if (is_p ? !has_kwpp : !has_kwps) continue;
                    IDispatch *& app = is_p ? app_p : app_w;
                    bool & logged = is_p ? logged_p : logged_w;
                    if (!app) {
                        IUnknown * punk = nullptr;
                        if (SUCCEEDED(GetActiveObject(
                                is_p ? clsid_kwpp : clsid_kwps, nullptr,
                                &punk)) &&
                            punk) {
                            punk->QueryInterface(IID_IDispatch, (void **)&app);
                            punk->Release();
                        }
                        if (app && !logged) {
                            log_msg("ctx probe: com attached (%s)",
                                    is_p ? "KWPP.Application"
                                         : "Kwps.Application");
                            logged = true;
                        }
                    }
                    if (!app) continue;
                    std::string txt;
                    ComRead r = is_p ? ppt_read_chain(app, txt, wta)
                                     : com_read_chain(app, txt, wta);
                    if (r == ComRead::OK) {
                        wchar_t wt[256] = {0};
                        HWND fh = GetForegroundWindow();
                        if (fh) GetWindowTextW(fh, wt, 256);
                        std::lock_guard<std::mutex> lk(g_mu);
                        g_com_text = txt;
                        g_com_stamp = GetTickCount64();
                        g_com_title = wt;  // 消费端指纹：标题变=切标签/切窗
                        published = true;
                    } else if (r == ComRead::DEAD) {
                        bool & logged = is_p ? logged_p : logged_w;
                        if (logged) {
                            log_msg("ctx probe: office app gone, "
                                    "will re-attach");
                            logged = false;
                        }
                        app->Release();
                        app = nullptr;
                    }
                    // NO_DOC（文档全关/演示非文本编辑态）与 TAB_IDLE（活动
                    // 标签非本组件）：保留附着，试另一组件
                }
                if (!published) {
                    std::lock_guard<std::mutex> lk(g_mu);
                    g_com_text.clear();
                    g_com_stamp = 0;
                    g_com_title.clear();
                }
            } else {
                std::lock_guard<std::mutex> lk(g_mu);
                g_com_text.clear();
                g_com_stamp = 0;  // 前台非 Office：COM 缓存即刻失效
                g_com_title.clear();
            }
            com_pending = (fg != OfficeKind::NONE) && !app_w && !app_p;
        }

        // UIA：读当前焦点元素（不限前台是谁——消费时校验进程一致）。
        // 打字活跃期（最近 3s 内有 kick）暂停周期读：动态内容应用（终端
        // 状态行时间戳等）的周期刷新令快照文本抖动 → prepare/score 双双
        // 失配 + 推理锁竞争（真机实测 150-215ms 长尾，2026-09-11）；
        // kick 读已覆盖上屏后的更新需求，打字期间快照保持稳定。
        if (uia_ok && (woken_by_kick ||
                       GetTickCount64() - g_last_kick_ms.load() > 3000)) {
            std::string txt;
            DWORD pid = 0;
            if (uia_read(uia, txt, pid)) {
                wchar_t ut[256] = {0};
                HWND ufh = GetForegroundWindow();
                if (ufh) GetWindowTextW(ufh, ut, 256);
                std::lock_guard<std::mutex> lk(g_mu);
                g_uia_text = txt;
                g_uia_stamp = GetTickCount64();
                g_uia_pid = pid;
                g_uia_title = ut;  // 消费端指纹：同进程多窗/标签切换即变
            } else {
                std::lock_guard<std::mutex> lk(g_mu);
                g_uia_text.clear();
                g_uia_stamp = 0;
                g_uia_pid = 0;
                g_uia_title.clear();
            }
        }
    }
}

static void ensure_started() {
    bool expect = false;
    if (g_started.compare_exchange_strong(expect, true))
        std::thread(thread_proc).detach();
}
static void kick() {
    g_last_kick_ms.store(GetTickCount64());
    g_kick.store(true);
    g_cv.notify_all();
}

// 任何"光标前文本可能变了"的已知信号（编辑键/点击，2026-09-12 信号层
// 统一）：成对执行 (a) 快照失效 (b) kick 延迟重读。失效保证 50ms 读回
// 窗口内旧快照不冒充（pick 拒用 → lua 落历史）；kick 保证应用处理完编辑
// 后读到新真文。缺失效则读早竞态冒充，缺 kick 则退格后首词无上文。
// 点击候选窗选词场景：click 信号废掉 commit kick 刚刷新的正确快照，
// 但随后的 kick 读回同样正确的文本（点击已处理完、含新上屏词），自愈。
static void edit_reset() {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_com_text.clear();
        g_com_stamp = 0;
        g_com_title.clear();
        g_uia_text.clear();
        g_uia_stamp = 0;
        g_uia_pid = 0;
        g_uia_title.clear();
    }
    kick();
}

// lua 调用线程消费：三层前两层判定（历史兜底由 lua 侧完成）。
// 返回 true 时 text/src 有效（src = "com" | "uia"）。
static bool pick(std::string & text, const char *& src) {
    unsigned long long now = GetTickCount64();
    if (foreground_office() != OfficeKind::NONE) {
        // 标题指纹：发布后前台标题变了（WPS 切标签/切窗即换标题）→ 旧快照
        // 属于旧标签，拒用——挡住 ≤2s 轮询间隙的串档（commit kick 会立刻
        // 换上新标签文本）。任一侧标题取不到 → 保守放行
        wchar_t t[256] = {0};
        HWND fh = GetForegroundWindow();
        bool title_ok =
            (!fh || GetWindowTextW(fh, t, 256) <= 0);
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_com_stamp && now - g_com_stamp <= 2500 && !g_com_text.empty() &&
            (title_ok || g_com_title.empty() || g_com_title == t)) {
            text = g_com_text;
            src = "com";
            return true;
        }
    }
    {
        // UIA 快照：新鲜 && 前台进程 == 快照来源进程（防切窗残留文本误用）
        // + 标题指纹（2026-09-12）：pid 校验挡不住同进程多窗/多标签（Chrome
        // 双窗、Win11 记事本标签同 pid）——发布后标题变 = 已切窗/标签 →
        // 拒用，与 COM 同款；任一侧取不到标题 → 保守放行
        DWORD fg = foreground_pid();
        wchar_t t[256] = {0};
        HWND fh = GetForegroundWindow();
        bool title_ok = (!fh || GetWindowTextW(fh, t, 256) <= 0);
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_uia_stamp && now - g_uia_stamp <= 2500 && !g_uia_text.empty() &&
            g_uia_pid != 0 && g_uia_pid == fg &&
            (title_ok || g_uia_title.empty() || g_uia_title == t)) {
            text = g_uia_text;
            src = "uia";
            return true;
        }
    }
    return false;
}

}  // namespace ctxlink

#else
// Linux/fcitx5：无 COM/UIA 光标采集旁路。所有入口保持原签名，
// pick 恒失败 → lua 侧回落 commit_history 上文（AI·历史 徽章）；
// fcitx5 每个窗口独立 rime 会话，commit_history 天然按应用隔离。
namespace ctxlink {

inline void ensure_started() {}
inline void kick() {}
inline void edit_reset() {}
inline bool pick(std::string &, const char * &) { return false; }
inline unsigned long foreground_pid() { return 0; }

}  // namespace ctxlink

#endif  // _WIN32


// ============================================================
// 异步模型加载
// ============================================================
// Vulkan 显存策略（2026-10-02，本机 7800 XT 实测）：强制权重进 device-local
// 显存，避开 host-visible（pinned host 内存，RADV+ReBAR 下的默认优选）的
// PCIe 带宽瓶颈——中载 Step2 29ms → 16ms（门槛测试报告 §4.2）。
// 必须在设备初始化（llama_backend_init / 首次模型加载）前设置；
// 外部环境已显式设置时不覆盖（留出用户覆盖口）。
static void apply_vulkan_defaults() {
#ifndef _WIN32
    if (!getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"))
        setenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "1", 1);
#endif
}

static void load_model_async() {
    if (g_loaded.load() || g_loading.load()) return;
    g_loading.store(true);

    std::thread([]() {
#ifdef _WIN32
        // 加载期让渡 CPU（2026-09-04，与源码版同步）：模型读取+反量化+预热
        // 吃满 CPU/IO，会拖挂并发启动的进程（源码版实测 QQ 音乐等 CEF 应用
        // 秒级挂起）；warmup 前恢复 NORMAL——llama 的 decode worker 靠条件
        // 变量协同，低优先级下会被 NORMAL 线程饿死，卡在 warmup 不返回
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
        log_msg("loading model: %s", g_model_path.c_str());

        apply_vulkan_defaults();  // 必须在 backend/设备初始化之前
        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = g_n_gpu_layers;
#ifdef _WIN32
        mparams.use_mmap = 1;
#else
        // 2026-10 llama.cpp 已以 load_mode 取代 use_mmap 布尔
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
#endif

        g_model = llama_model_load_from_file(g_model_path.c_str(), mparams);
        if (!g_model) {
            log_msg("ERROR: failed to load model");
            g_loading.store(false);
            return;
        }
        g_vocab = llama_model_get_vocab(g_model);

        int n_thr = g_n_threads;

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = g_n_ctx;
        cparams.n_threads       = n_thr;
        cparams.n_threads_batch = n_thr;
        cparams.n_seq_max       = g_n_seq_max;

#ifdef _WIN32
        g_ctx = llama_new_context_with_model(g_model, cparams);
#else
        // 2026-10 llama.cpp 更名 llama_init_from_model
        g_ctx = llama_init_from_model(g_model, cparams);
#endif
        if (!g_ctx) {
            log_msg("ERROR: failed to create context");
            llama_model_free(g_model);
            g_model = nullptr;
            g_loading.store(false);
            return;
        }

#ifdef _WIN32
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
#endif

        // warmup
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            llama_memory_clear(llama_get_memory(g_ctx), false);
            const char * warmup = "\n";
            llama_token tokens[4];
            int n_tokens = llama_tokenize(g_vocab, warmup, (int)strlen(warmup),
                                           tokens, 4, true, true);
            if (n_tokens > 0) {
                llama_batch batch = llama_batch_get_one(tokens, n_tokens);
                llama_decode(g_ctx, batch);
            }
        }

        g_loaded.store(true);
        g_loading.store(false);
        log_msg("model ready (n_ctx=%d threads=%d n_gpu_layers=%d)", g_n_ctx,
                n_thr, g_n_gpu_layers);
    }).detach();
}

// 卸载模型（model_path 变更时由 lua_newindex 调用，2026-09-01）：旧模型
// 驻留内存会让新路径被无视——卸载后下一次 prepare/score 懒加载新路径。
// 与源码版 llm_filter.cc 的 unload_model 同构（等待在途加载 + 锁内释放）。
static void unload_model() {
    while (g_loading.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_loaded.load() || g_model || g_ctx) {
        if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
        if (g_model) { llama_model_free(g_model); g_model = nullptr; }
        g_vocab = nullptr;
        g_loaded.store(false);
        g_prep_ready = false;
        g_prep_ctx.clear();
        g_prep_logits.clear();
        log_msg("model unloaded (model_path changed)");
    }
}

// ============================================================
// 分词
// ============================================================
static std::vector<llama_token> tokenize(const char * text) {
    std::vector<llama_token> toks(128);
    int n = llama_tokenize(g_vocab, text, (int)strlen(text),
                            toks.data(), (int)toks.size(), true, true);
    // llama_tokenize API 约定：buffer 不足时返回负值，-n 为所需 token 数。
    // 首次调用带 128 上限，超长文本（如整段上文）会返回负 → resize 后重试。
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(g_vocab, text, (int)strlen(text),
                           toks.data(), (int)toks.size(), true, true);
    }
    toks.resize(std::max(0, n));
    return toks;
}

// 上文 token 化 + min_tokens/max_ctx 截尾（prepare/score 入口共用）。
// token 数不足 min_tokens（方案 llm_rerank/min_tokens，lua 传入）→ 返回空
//（空 ctx 无推理是配置语义, 非 bug）。
static std::vector<llama_token> ctx_token_ids(const char * context) {
    auto ids = tokenize(context);
    if ((int)ids.size() < g_min_tokens) return {};
    if ((int)ids.size() > g_max_ctx_tokens)
        ids.erase(ids.begin(), ids.end() - g_max_ctx_tokens);
    return ids;
}

// ============================================================
// Softmax CE 辅助：-log(softmax(x)[target])
// ============================================================
static double cross_entropy(float * logits, int vs, int target_id) {
    float m = -1e30f;
    for (int k = 0; k < vs; k++) if (logits[k] > m) m = logits[k];
    double se = 0;
    for (int k = 0; k < vs; k++) se += exp((double)(logits[k] - m));
    return -((double)(logits[target_id] - m) - log(se));
}

// 归一化 (max, logsumexp) 一次计算, 多目标 CE 共享复用:
// Step 1 所有候选对同一 ctx logits 打分, 原实现对每个候选重扫
// O(vocab) (5 候选取 5 次 -> 实测 CE1 段 ~6ms), 共享后一次扫描 + 查表
// (-> ~2ms)
static void logits_normalizer(float * logits, int vs, float & m, double & lse) {
    m = -1e30f;
    for (int k = 0; k < vs; k++) if (logits[k] > m) m = logits[k];
    double se = 0;
    for (int k = 0; k < vs; k++) se += exp((double)(logits[k] - m));
    lse = log(se);
}
static double ce_target(float * logits, int target_id, float m, double lse) {
    return -((double)(logits[target_id] - m) - lse);
}

// ============================================================
// 核心评分：ctx 仅 1 次 decode + KV copy + 多序列分层并行候选 decode
//
// Step 1: decode ctx → 保存 logits → 所有候选的第 1 个 token CE 从此计算
// Step 2: KV copy ctx → M 个 seq，并行 decode 候选首 token → 第 2 token CE
// Step 3: 同一批 seq 继续 decode → 第 3 token CE
//
// 原理：P(tok_i0|ctx) 对所有 i 共享同一个 ctx_last hidden state
//       多 token 候选每个 seq 仅 1 个新 token，SSM 跨序列干扰可忽略
//       ith = batch 数组索引（不是 logits 标记序号）
// ============================================================
// score 持锁标记：清零后 prep worker 才能重新开跑。带显式 clear() 是因为日志要
// 挪到放锁之后（见函数尾），不能只靠析构。
struct ScoreActiveGuard {
    ScoreActiveGuard()  { g_score_active.fetch_add(1); }
    void clear() { if (active_) { g_score_active.fetch_sub(1); active_ = false; } }
    ~ScoreActiveGuard() { clear(); }
    bool active_ = true;
};

static void score_batch(const std::vector<llama_token> & ctx_ids,
                         const std::vector<std::vector<llama_token>> & cands,
                         std::vector<double> & scores_out) {
    scores_out.assign(cands.size(), -1e10);
    int n_cands = (int)cands.size();
    if (n_cands == 0) return;

    // 这几个量放锁之后还要打日志，所以在锁外声明（锁内只赋值）
    int ctx_len = (int)ctx_ids.size();
    double wait_ms = 0, ms1 = 0, ms_ce1 = 0, ms2a = 0, ms2b = 0, ms3 = 0;
    double ms_score = 0, total_ms = 0;
    const char * prep_miss = "none";   // prep 未命中原因，命中时打印 "-"

    auto t0 = std::chrono::high_resolution_clock::now();
    // 让 prep worker 严格让路：先置 waiting → 拿锁 → 置 active；两者都清零后
    // worker 才会开启新的长 decode。顺序不能反，否则 worker 能插到中间抢锁。
    g_score_waiting.fetch_add(1);
    std::unique_lock<std::mutex> lock(g_mutex);
    g_score_waiting.fetch_sub(1);
    ScoreActiveGuard score_active;
    auto t1 = std::chrono::high_resolution_clock::now();
    wait_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int vs = llama_n_vocab(g_vocab);


    // 按 token 数分组
    std::vector<int> idx2, idx3;
    for (int i = 0; i < n_cands; i++) {
        if (cands[i].size() >= 2) idx2.push_back(i);
        if (cands[i].size() >= 3) idx3.push_back(i);
    }
    int M = (int)idx2.size();
    int K = (int)idx3.size();
    std::vector<int> cand_to_seq(n_cands, -1);  // candidate index → seq id

    // 检测预解码状态：ctx 完全匹配 且 seq0 未被其他 decode 覆盖 → 跳过 Step 1
    // 代次匹配防止: 完整流程 score 覆盖 seq0 后, 旧 prep 状态被误用 (KV 与 logits 不一致)
    bool use_prep = g_prep_ready && ctx_ids == g_prep_ctx && g_seq0_gen == g_prep_gen;
    if (!use_prep) {
        // prep=0 原先完全不可解释，这里给出原因：none=压根没有预解码结果
        //（请求被丢弃/还没来得及跑）；ctx=有结果但 ctx 不同；gen=ctx 相同但
        // seq0 已被别的 decode 覆盖（代次失配）
        if (!g_prep_ready)              prep_miss = "none";
        else if (ctx_ids != g_prep_ctx) prep_miss = "ctx";
        else                            prep_miss = "gen";
    }
    std::vector<float> ctx_logits;
    ms1 = 0;

    if (!use_prep) {
        // ---- 完整流程：Step 1 ctx decode ----
        auto ts1_0 = std::chrono::high_resolution_clock::now();
        // memory_clear 第 2 参：true=零化整个 KV cache（40 万次 decode 的
        // benchmark 会拖慢吞吐）；false=仅重置元数据。新序列的因果注意力
        // 掩码天然隔离旧 KV，无需显式归零。
        llama_memory_clear(llama_get_memory(g_ctx), false);
        llama_batch ctx_batch = llama_batch_init(ctx_len, 0, 1);
        for (int j = 0; j < ctx_len; j++) {
            ctx_batch.token[j] = ctx_ids[j]; ctx_batch.pos[j] = j;
            ctx_batch.n_seq_id[j] = 1; ctx_batch.seq_id[j][0] = 0;
        }
        ctx_batch.logits[ctx_len - 1] = 1;
        ctx_batch.n_tokens = ctx_len;
        if (llama_decode(g_ctx, ctx_batch) == 0) {
            g_seq0_gen++;  // seq0 KV 已更新
            float* cl = llama_get_logits_ith(g_ctx, ctx_len - 1);
            if (cl) ctx_logits.assign(cl, cl + vs);
        }
        llama_batch_free(ctx_batch);

        if (ctx_logits.empty()) return;  // decode failed
        // 完整流程自我刷新 prep: 刚 decode 完 seq0 (代次已递增), KV/logits 一致,
        // 保存后同 ctx 的后续 score (翻页/候选窗重建) 全部命中——
        // 否则一次未命中导致代次失配, 后续 score 连锁全部 prep=0 直到下次 commit
        g_prep_gen = g_seq0_gen;
        g_prep_ctx = ctx_ids;
        g_prep_logits = ctx_logits;
        g_prep_ready = true;
        auto ts1_1 = std::chrono::high_resolution_clock::now();
        ms1 = std::chrono::duration<double, std::milli>(ts1_1 - ts1_0).count();
    } else {
        // 预解码命中：直接使用已保存的 logits
        ctx_logits = g_prep_logits;
    }

    // 所有候选的第 1 个 token CE (共享同一 ctx logits -> 归一化只算一次)
    auto ts_ce1_0 = std::chrono::high_resolution_clock::now();
    std::vector<double> ce_sum(n_cands, 0);
    float m0;
    double lse0;
    logits_normalizer(ctx_logits.data(), vs, m0, lse0);
    for (int i = 0; i < n_cands; i++) {
        ce_sum[i] = kCeW1 * ce_target(ctx_logits.data(), cands[i][0], m0, lse0);
    }
    auto ts_ce1_1 = std::chrono::high_resolution_clock::now();
    ms_ce1 =
        std::chrono::duration<double, std::milli>(ts_ce1_1 - ts_ce1_0).count();

    // ---- Step 2: 多 token 候选的首 token 并行 decode（每 seq 1 token）----
    ms2a = 0; ms2b = 0;
    if (M > 0) {
        auto ts2_0 = std::chrono::high_resolution_clock::now();
        // KV copy ctx (seq 0) → worker seqs 1..M
        for (int s = 0; s < M; s++) {
            llama_memory_seq_cp(llama_get_memory(g_ctx), 0, s + 1, 0, -1);
            cand_to_seq[idx2[s]] = s + 1;
        }
        auto ts2_kv = std::chrono::high_resolution_clock::now();

        llama_batch b2 = llama_batch_init(M, 0, M);
        for (int s = 0; s < M; s++) {
            int ci = idx2[s];
            b2.token[s] = cands[ci][0];
            b2.pos[s] = ctx_len;
            b2.n_seq_id[s] = 1;
            b2.seq_id[s][0] = s + 1;
            b2.logits[s] = 1;
        }
        b2.n_tokens = M;
        if (llama_decode(g_ctx, b2) == 0) {
            for (int s = 0; s < M; s++) {
                int ci = idx2[s];
                float* l = llama_get_logits_ith(g_ctx, s);
                if (l) ce_sum[ci] += kCeW2 * cross_entropy(l, vs, cands[ci][1]);
                else   ce_sum[ci] = -1e10;
            }
        } else {
            for (int ci : idx2) ce_sum[ci] = -1e10;
            log_msg("WARN: step2 decode failed");
            // 失败自愈：prep 命中路径无 memory_clear，decode 失败多为 KV 耗尽
            // （见下方 worker 清理说明）——置无效强制下次 score 走全流程重建
            g_prep_ready = false;
        }
        llama_batch_free(b2);
        auto ts2_1 = std::chrono::high_resolution_clock::now();
        ms2a = std::chrono::duration<double, std::milli>(ts2_kv - ts2_0).count();
        ms2b = std::chrono::duration<double, std::milli>(ts2_1 - ts2_kv).count();
    }

    // ---- Step 3: 3-token 候选的次 token 并行 decode ----
    ms3 = 0;
    if (K > 0) {
        auto ts3_0 = std::chrono::high_resolution_clock::now();
        llama_batch b3 = llama_batch_init(K, 0, K);
        for (int s = 0; s < K; s++) {
            int ci = idx3[s];
            int seq_id = cand_to_seq[ci];  // same seq as step 2
            b3.token[s] = cands[ci][1];
            b3.pos[s] = ctx_len + 1;
            b3.n_seq_id[s] = 1;
            b3.seq_id[s][0] = seq_id;
            b3.logits[s] = 1;
        }
        b3.n_tokens = K;
        if (llama_decode(g_ctx, b3) == 0) {
            for (int s = 0; s < K; s++) {
                int ci = idx3[s];
                float* l = llama_get_logits_ith(g_ctx, s);
                if (l) ce_sum[ci] += kCeW3 * cross_entropy(l, vs, cands[ci][2]);
                else   ce_sum[ci] = -1e10;
            }
        } else {
            for (int ci : idx3) ce_sum[ci] = -1e10;
            log_msg("WARN: step3 decode failed");
            g_prep_ready = false;  // 同 Step2：失败自愈
        }
        llama_batch_free(b3);
        auto ts3_1 = std::chrono::high_resolution_clock::now();
        ms3 = std::chrono::duration<double, std::milli>(ts3_1 - ts3_0).count();
    }

    // ---- worker 序列 KV 清理（2026-09-04 修"推理自停"根因，与源码版同步）----
    // seq_cp 是共享标记非复制：Step2/3 的 decode cell 在评分结束后仍归属
    // seq 1..M。prep 命中路径无 memory_clear，同 ctx 连续评分（退格重打/
    // 改码不换 ctx/失败重试）每轮净耗 M+K cell —— n_ctx=64 约 6~7 轮耗尽
    // → decode 静默失败 → 全哨兵分且 prep_ready 仍真 → 卡死失败循环，
    // 仅 commit 触发 prepare 或重部署可解（非本机实测）。评分尾部立即释放
    // worker 归属：共享 cell 只去掉一个归属方，seq0 与 prep 状态不受影响，
    // prep 命中照常，每次评分净耗归零。
    if (M > 0) {
        auto *mem = llama_get_memory(g_ctx);
        for (int s = 1; s <= M; s++)
            llama_memory_seq_rm(mem, s, 0, -1);
    }

    // ---- 输出分数 ----
    auto ts_sc_0 = std::chrono::high_resolution_clock::now();
    // 加权 CE 和取负（kCeW1/2/3 见配置区注释；4+ token 候选只算前三项，
    // 尾部外推 2026-09-16 证伪删除：λ=0 全曲线最优，取消零损失）
    for (int i = 0; i < n_cands; i++) {
        scores_out[i] = ce_sum[i] > -1e9 ? -ce_sum[i] : -1e10;
    }

    // 注意: prep 状态不消耗——同一 ctx 期间可重复命中
    // (代次机制保证 seq0 被覆盖后自动失效, 无需主动清除)

    auto t_end = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    ms_score =
        std::chrono::duration<double, std::milli>(t_end - ts_sc_0).count();

    // 放锁之后再打日志：log_msg 每次都 fopen/fprintf/fclose，留在锁里会把这段
    // 耗时算进别人看到的 wait —— 而 wait 正是本项目要测的量。
    lock.unlock();
    score_active.clear();

    // 计时: wait=锁等待 S1=ctx decode(0 on prep) CE1=P(cand0|ctx)
    //       KV=KV copy S2=decode cand0 S3=decode cand1 score=加权求和
    // 不再限频（原为"每 10 次记一次 + 总耗时 >100ms 必记"）：那样样本天然偏向
    // 慢请求，占比无法当全量口径。体量可忽略——真人实测约 0.8 次评分/秒。
    log_msg("score: wait=%.0fms S1=%.0fms CE1=%.0fms KV=%.0fms S2=%.0fms "
            "S3=%.0fms score=%.0fms total=%.0fms prep=%d ctx_tok=%d cand=%d "
            "prep_miss=%s",
            wait_ms, ms1, ms_ce1, ms2a, ms2b, ms3, ms_score, total_ms,
            use_prep ? 1 : 0, ctx_len, n_cands,
            use_prep ? "-" : prep_miss);
}

// ============================================================
// 预解码：在 commit 后异步执行 Step 1（ctx decode）
// 调度方式见 g_pending_* 那组全局的注释：单常驻 worker + 最新请求槽 + 代次校验
// ============================================================

// 持锁执行：ctx decode + 发布 prep 状态。调用方必须已持有 g_mutex，且校验过
// g_ctx/g_vocab 非空与代次有效。
static void prep_decode_locked(const std::vector<llama_token> & ctx_ids, int seq) {
    int ctx_len = (int)ctx_ids.size();
    auto tp0 = std::chrono::high_resolution_clock::now();
    log_msg("prepare: start ctx_tok=%d seq=%d", ctx_len, seq);
    int vs = llama_n_vocab(g_vocab);
    auto* mem = llama_get_memory(g_ctx);

    // 清除上次的预解码
    g_prep_ready = false;
    g_prep_ctx.clear();
    g_prep_logits.clear();

    // Step 1: decode ctx → seq 0
    // 同 score_batch：memory_clear 第 2 参 false = 不零化 KV，仅重置元数据
    llama_memory_clear(mem, false);
    llama_batch ctx_batch = llama_batch_init(ctx_len, 0, 1);
    for (int j = 0; j < ctx_len; j++) {
        ctx_batch.token[j] = ctx_ids[j]; ctx_batch.pos[j] = j;
        ctx_batch.n_seq_id[j] = 1; ctx_batch.seq_id[j][0] = 0;
    }
    ctx_batch.logits[ctx_len - 1] = 1;
    ctx_batch.n_tokens = ctx_len;

    if (llama_decode(g_ctx, ctx_batch) != 0) {
        llama_batch_free(ctx_batch);
        return;  // decode 失败，score 时走完整流程
    }
    float* cl = llama_get_logits_ith(g_ctx, ctx_len - 1);
    if (!cl) { llama_batch_free(ctx_batch); return; }
    g_prep_logits.assign(cl, cl + vs);
    llama_batch_free(ctx_batch);

    // 不预复制 KV——score 的 KV copy 本身很快（n_ctx=128 下微秒级），
    // score() 检测 ctx 一致时跳过 Step 1，但仍执行 KV copy + Step 2

    g_seq0_gen++;   // seq0 KV 已更新 (本代)
    g_prep_gen = g_seq0_gen;
    g_prep_ctx = ctx_ids;
    g_prep_ready = true;
    double took_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - tp0).count();
    log_msg("prepare: done ctx_tok=%d seq=%d took=%.0fms", ctx_len, seq, took_ms);
}

// 取走最新请求（覆盖式槽位；只等请求，永不返回失败）。
static void prep_take(std::vector<llama_token> & out, int & seq) {
    std::unique_lock<std::mutex> lk(g_prep_mu);
    g_prep_cv.wait(lk, [] { return g_pending_valid; });
    out = g_pending_ctx;
    seq = g_pending_seq.load();
    g_pending_valid = false;
}

// 常驻 worker：取最新请求 → 抢锁（抢不到就退避重试，**不丢弃**）→ 解码发布。
//
// 它是**常驻**的：不设停止标志，随进程结束而消失（与上游 Windows 侧 ctxlink
// 后台线程同一形态）。为什么不加"优雅停止"：librime 会在一个进程里走多轮
// initialize/finalize（fcitx5 启动期的部署/重载就会），而停止标记是单向的——
// 任何一次 finalize 置下它，之后的预习就**静默失效**。真机实测过这个回归：
// 一轮 finalize 之后 31 次预习请求全部"未开始即被更新取代"、完成 0 次，只剩
// score 自己同步解码。退出期也不需要它停：进程里没有任何析构路径会释放 llama
// 上下文（只有 unload_model 会释放，而它要拿 g_mutex，与 worker 天然互斥）。
//
// "常驻"有一个前提，见 g_prep_mu/g_prep_cv 的注释：worker 停驻的那把锁和 cv 必须
// **永不析构**，否则退出时主线程会在 pthread_cond_destroy 里等一个永远不走的等待者。
static void prep_worker() {
    for (;;) {
        std::vector<llama_token> ctx;
        int seq = 0;
        prep_take(ctx, seq);
        for (;;) {
            // 已被更新的请求取代：这是唯一合法的"作废"（旧 ctx 已无人需要）
            if (seq != g_prep_seq.load()) {
                g_prep_superseded.fetch_add(1);
                log_msg("prepare: superseded seq=%d ctx_tok=%d (latest=%d)",
                        seq, (int)ctx.size(), g_prep_seq.load());
                break;
            }
            // 给 score 让路：有人等锁或持锁时不开启新的长 decode。
            // 这是硬约束——改造后不得给 score 增加任何等待。
            if (g_score_waiting.load() || g_score_active.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
            if (!lock.owns_lock()) {
                // 原实现在这里直接丢弃请求（log "SKIP score in progress"），
                // 被丢的那个 ctx 紧接着就被 score 需要 → 只能同步解码 70~90ms。
                // 改成退避重试：请求只会因"被更新取代"而作废，不会因抢不到锁而丢失。
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (seq != g_prep_seq.load()) {
                g_prep_superseded.fetch_add(1);
                log_msg("prepare: superseded seq=%d ctx_tok=%d (latest=%d)",
                        seq, (int)ctx.size(), g_prep_seq.load());
                break;
            }
            if (g_score_waiting.load()) continue;  // 让出锁，交给 score 先走
            if (!g_ctx || !g_vocab) break;         // 模型已被卸载
            prep_decode_locked(ctx, seq);
            break;
        }
    }
}

static void ensure_prep_worker() {
    bool expected = false;
    if (g_prep_worker_started.compare_exchange_strong(expected, true)) {
        // detach 而不是留一个可 join 的 std::thread 全局：那个全局在静态析构期
        // 若仍 joinable，标准库会直接 std::terminate（实测退出时报
        // "terminate called without an active exception" / abort）。
        std::thread(prep_worker).detach();
    }
}

// ============================================================
// Lua API
// ============================================================
static int lua_prepare(lua_State * L) {
    // 懒加载触发点（2026-08-31 修 model_path 不生效）：加载不在 luaopen
    // 急切启动——那时 lua 侧还没把 schema 的 model_path 写进来，加载线程
    // 会拿默认路径开跑，配置形同虚设。首个触发方是 processor 的 prepare
    //（其 lua 在 require 后、prepare 前已设好 model_path），score 是兜底。
    if (!g_loaded.load()) {
        if (!g_loading.load()) load_model_async();
        lua_pushboolean(L, 0);
        return 1;
    }
    const char * context = luaL_checkstring(L, 1);
    // 上文 token 数 < min_tokens → 不预解码（空 ctx 无推理是配置语义）
    auto ctx_ids = ctx_token_ids(context);
    if (ctx_ids.empty()) {
        lua_pushboolean(L, 0);
        return 1;
    }

    // 只登记"最新请求"（覆盖式）：旧请求被新请求取代即作废，worker 取走的
    // 一定是此刻最新的那一个。这里不起线程——原实现每次 new 一个 detached
    // 线程，既贵（每键一次线程创建）又会堆在 g_mutex 上互相争用。
    int seq = ++g_prep_seq;
    int over_seq = 0, over_tok = 0;   // 被这次覆盖掉的旧请求（用于日志，见下）
    {
        std::lock_guard<std::mutex> lk(g_prep_mu);
        if (g_pending_valid) {
            over_seq = g_pending_seq.load();
            over_tok = (int)g_pending_ctx.size();
            g_prep_superseded.fetch_add(1);
        }
        g_pending_ctx = ctx_ids;
        g_pending_seq.store(seq);
        g_pending_valid = true;
    }
    // 日志放在锁外（log_msg 要开关文件）。为什么必须记：worker 还没取走就被
    // 覆盖的请求不会有 start/done 行，不记的话统计上会少算请求数，改前改后
    // 的"丢弃率"就不可比了（原实现是逐条 SKIP，条数完整）。
    if (over_seq)
        log_msg("prepare: superseded seq=%d ctx_tok=%d (未开始即被更新取代)",
                over_seq, over_tok);
    ensure_prep_worker();
    g_prep_cv.notify_one();

    lua_pushboolean(L, 1);
    return 1;
}

static int lua_score(lua_State * L) {
    if (!g_loaded.load()) {
        if (!g_loading.load()) load_model_async();
        lua_pushnil(L);
        return 1;
    }

    const char * context = luaL_checkstring(L, 1);
    luaL_checktype(L, 2, LUA_TTABLE);

    std::vector<std::string> cand_texts;
    int n = (int)luaL_len(L, 2);
    if (n < 2) { lua_pushnil(L); return 1; }
    if (n > g_n_seq_max) n = g_n_seq_max;
    for (int i = 1; i <= n; i++) {
        lua_rawgeti(L, 2, i);
        const char * s = lua_tostring(L, -1);
        if (s) cand_texts.push_back(s);
        lua_pop(L, 1);
    }
    if (cand_texts.size() < 2) { lua_pushnil(L); return 1; }

    std::vector<llama_token> ctx_ids = ctx_token_ids(context);
    if (ctx_ids.empty()) { lua_pushnil(L); return 1; }

    std::vector<std::vector<llama_token>> cand_ids;
    for (auto & s : cand_texts) {
        auto ids = tokenize(s.c_str());
        if (ids.empty()) ids.push_back(0);
        cand_ids.push_back(ids);
    }

    std::vector<double> scores;
    try {
        score_batch(ctx_ids, cand_ids, scores);
    } catch (...) {
        log_msg("ERROR: exception in score_batch");
        lua_pushnil(L);
        return 1;
    }

    // 按分数降序排列
    std::vector<int> order(scores.size());
    for (int i = 0; i < (int)order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    g_last_scores = scores;
    g_last_cands  = cand_texts;

    lua_newtable(L);
    for (int i = 0; i < (int)order.size(); i++) {
        lua_pushstring(L, cand_texts[order[i]].c_str());
        lua_rawseti(L, -2, i + 1);
    }
    return 1;
}

static int lua_get_scores(lua_State * L) {
    std::lock_guard<std::mutex> lock(g_mutex);
    lua_newtable(L);
    for (size_t i = 0; i < g_last_cands.size(); i++) {
        lua_pushnumber(L, g_last_scores[i]);
        lua_setfield(L, -2, g_last_cands[i].c_str());
    }
    return 1;
}

static int lua_is_ready(lua_State * L) {
    lua_pushboolean(L, g_loaded.load() ? 1 : 0);
    return 1;
}

// 光标上文（三层前两层，2026-09-10）：成功返回 (text, "com"|"uia")，
// 无可用快照返回 nil —— lua 侧回落 commit_history。
// COM/UIA 调用全部在后台线程（本函数只读 mutex 快照 + 前台窗口判定，
// 按键路径零阻塞）
static int lua_llm_context(lua_State * L) {
    ctxlink::ensure_started();
    std::string text;
    const char * src = nullptr;
    if (ctxlink::pick(text, src)) {
        lua_pushstring(L, text.c_str());
        lua_pushstring(L, src);
        return 2;
    }
    lua_pushnil(L);
    return 1;
}

// commit 后触发：立即刷新 COM/UIA 快照（新词已入文档/控件，读链 ~6-13ms
// 异步完成，下一键即有新鲜上文）
static int lua_kick_context(lua_State * L) {
    ctxlink::kick();
    return 0;
}

// 编辑键/点击信号（2026-09-12 信号层统一）：失效 COM/UIA 快照 + kick
// 延迟重读（语义见 ctxlink::edit_reset）。processor 在编辑键与点击检测
// 分支调用——快照制的失效信号不再只有 commit。
static int lua_edit_reset_context(lua_State * L) {
    ctxlink::ensure_started();
    ctxlink::edit_reset();
    return 0;
}

// 前台窗口切换检测（跨应用，2026-09-11）：插件版 lua 无焦点事件，经此
// 让 processor 在切换后重置历史上文（对齐源码版 focus:switch reset 语义
// ——真机踩坑：切新文档首词用旧窗口历史冒充上文标 AI·历史）。
// pid+HWND 双比对（2026-09-12）：仅比 pid 挡不住同应用多窗/多标签切换
// （Chrome 双窗、Win11 记事本标签，pid 相同），旧窗历史照样冒充；HWND
// 级对齐源码版 TSF prevFocus!=focus。候选窗不抢前台，打字中不误触。
static int lua_fg_changed(lua_State * L) {
#ifdef _WIN32
    static DWORD last_pid = 0;
    static HWND last_hwnd = nullptr;
    DWORD cur_pid = ctxlink::foreground_pid();
    HWND cur_hwnd = GetForegroundWindow();
    bool changed = (last_pid != 0 && cur_pid != 0 &&
                    (cur_pid != last_pid || cur_hwnd != last_hwnd));
    last_pid = cur_pid;
    last_hwnd = cur_hwnd;
    lua_pushboolean(L, changed ? 1 : 0);
#else
    // Linux：无前台窗口检测；fcitx5 每窗口独立 rime 会话，历史按应用隔离
    lua_pushboolean(L, 0);
#endif
    return 1;
}

// 鼠标点击检测（2026-09-11 深夜用户定案：鼠标移动主要靠点击检测——
// 任何点击=光标可能移动→清历史上文兜底，不做例外排除——候选窗/工具
// 栏点击也清，上下文宁可变短不可错）。2026-09-12 起 processor 在点击
// 分支同时调 edit_reset_context 失效快照+重读（真文快照也是旧时刻的，
// 点击移光标后同样冒充）。
// GetAsyncKeyState LSB=自本进程上次查询以来按下过，专为轮询设计；
// 首次调用建立基线防进程历史点击误报。
static int lua_click_happened(lua_State * L) {
#ifdef _WIN32
    static bool inited = false;
    short b = GetAsyncKeyState(VK_LBUTTON) | GetAsyncKeyState(VK_RBUTTON) |
              GetAsyncKeyState(VK_MBUTTON);
    bool clicked = inited && (b & 1);
    inited = true;
    lua_pushboolean(L, clicked ? 1 : 0);
#else
    // Linux：无全局鼠标轮询；点击移光标后历史上文过期属已知限制
    lua_pushboolean(L, 0);
#endif
    return 1;
}

// ============================================================
// 编码模式匹配（2026-09-30 取代 min_code_len/max_code_len 区间）
// ============================================================
// match_code(pattern, code) -> bool
//   全串匹配（与 Rime 自身 speller/auto_select_pattern 的语义一致，
//   其实现为 boost::regex_match）。这里用 std::regex（MSVC 标准库，
//   零新依赖）承载同一套 ECMAScript 语法——用户写的 .{4} / .{4,} /
//   .{3,4} / [abcde]{4} 在 boost 与 std::regex 下等价。
//   语义约定：
//     - 模式为空或 nil → 视为「总是匹配」（回退默认 .{4} 由调用方决定）
//     - 模式非法（编译异常）→ 回退默认模式并记日志，不抛给 lua
//     - 匹配对象是**输入编码串**（非候选词）
//   性能：仅当模式字符串变化时重新编译（用户改配置才触发）。
static const char *kDefaultCodePattern = ".{4}";  // 恰 4 码 = 原 min_code_len=4

static int lua_match_code(lua_State * L) {
    const char * pat = luaL_optstring(L, 1, NULL);
    const char * code = luaL_optstring(L, 2, "");
    if (!pat || !*pat) {          // 空模式 = 总是匹配
        lua_pushboolean(L, 1);
        return 1;
    }
    static std::string  last_pat;
    static std::regex   re;
    static bool         re_valid = false;
    if (last_pat != pat || (!re_valid && last_pat.empty())) {
        last_pat = pat;
        try {
            re = std::regex(pat, std::regex::ECMAScript);
            re_valid = true;
        } catch (const std::regex_error & e) {
            log_msg("[match_code] invalid pattern '%s' (%s) -> fallback '%s'",
                    pat, e.what(), kDefaultCodePattern);
            try {
                last_pat = kDefaultCodePattern;
                re = std::regex(kDefaultCodePattern, std::regex::ECMAScript);
                re_valid = true;
            } catch (...) {
                re_valid = false;
            }
        }
    }
    bool ok = false;
    if (re_valid) {
        try {
            ok = std::regex_match(std::string(code), re);
        } catch (...) {
            ok = false;
        }
    }
    lua_pushboolean(L, ok ? 1 : 0);
    return 1;
}

// ============================================================
// __index / __newindex
// ============================================================
static int lua_index(lua_State * L) {
    const char * key = luaL_checkstring(L, 2);
    if (strcmp(key, "is_ready") == 0)       lua_pushcfunction(L, lua_is_ready);
    else if (strcmp(key, "get_scores") == 0) lua_pushcfunction(L, lua_get_scores);
    else if (strcmp(key, "score") == 0)     lua_pushcfunction(L, lua_score);
    else if (strcmp(key, "prepare") == 0)   lua_pushcfunction(L, lua_prepare);
    else if (strcmp(key, "llm_context") == 0)    lua_pushcfunction(L, lua_llm_context);
    else if (strcmp(key, "kick_context") == 0)   lua_pushcfunction(L, lua_kick_context);
    else if (strcmp(key, "edit_reset_context") == 0) lua_pushcfunction(L, lua_edit_reset_context);
    else if (strcmp(key, "fg_changed") == 0)     lua_pushcfunction(L, lua_fg_changed);
    else if (strcmp(key, "click_happened") == 0) lua_pushcfunction(L, lua_click_happened);
    else if (strcmp(key, "model_path") == 0) lua_pushstring(L, g_model_path.c_str());
    else if (strcmp(key, "max_ctx") == 0)   lua_pushinteger(L, g_max_ctx_tokens);
    else if (strcmp(key, "min_tokens") == 0) lua_pushinteger(L, g_min_tokens);
    else if (strcmp(key, "n_threads") == 0) lua_pushinteger(L, g_n_threads);
    else if (strcmp(key, "n_ctx") == 0)     lua_pushinteger(L, g_n_ctx);
    else if (strcmp(key, "n_seq_max") == 0) lua_pushinteger(L, g_n_seq_max);
    else if (strcmp(key, "n_gpu_layers") == 0) lua_pushinteger(L, g_n_gpu_layers);
    else lua_pushnil(L);
    return 1;
}

static int lua_newindex(lua_State * L) {
    const char * key = luaL_checkstring(L, 2);
    if (strcmp(key, "model_path") == 0) {
        std::string np = luaL_checkstring(L, 3);
        if (np != g_model_path) {
            g_model_path = np;
            // 路径变更：卸载旧模型（等待在途加载有界；free 毫秒级），
            // 下一次 prepare/score 以新路径懒加载
            if (g_loaded.load()) unload_model();
        }
    }
    else if (strcmp(key, "max_ctx") == 0)    g_max_ctx_tokens = (int)luaL_checkinteger(L, 3);
    else if (strcmp(key, "min_tokens") == 0) g_min_tokens = (int)luaL_checkinteger(L, 3);
    // 线程数/上下文长度只在 llama 上下文**创建时**生效（见 load_model_async 的
    // cparams）。原实现只赋值不卸载，于是"改 cpu_cores 不重启 fcitx5 不生效"；
    // 更隐蔽的是首次加载存在竞态——加载由 processor 的 prepare 触发，而
    // `n_threads` 要等 filter 第一次评分才写进来，谁先到看运气（实测同一份配置
    // 出现过 threads=8 也出现过 threads=4）。变更即卸载，下次 prepare/score 以
    // 新参数重载；值相同时不动作，所以每键同步也不会反复卸载。
    else if (strcmp(key, "n_threads") == 0) {
        int v = (int)luaL_checkinteger(L, 3);
        if (v != g_n_threads) {
            g_n_threads = v;
            if (g_loaded.load()) unload_model();
        }
    }
    else if (strcmp(key, "n_ctx") == 0) {
        int v = (int)luaL_checkinteger(L, 3);
        if (v != g_n_ctx) {
            g_n_ctx = v;
            if (g_loaded.load()) unload_model();
        }
    }
    else if (strcmp(key, "n_seq_max") == 0)  g_n_seq_max = (int)luaL_checkinteger(L, 3);
    // GPU offload 层数（schema: llm_rerank/n_gpu_layers）：-1 全 offload / 0 纯 CPU。
    // 只在模型加载时生效（同 n_threads），变更即卸载重载。
    else if (strcmp(key, "n_gpu_layers") == 0) {
        int v = (int)luaL_checkinteger(L, 3);
        if (v != g_n_gpu_layers) {
            g_n_gpu_layers = v;
            if (g_loaded.load()) unload_model();
        }
    }
    else if (strcmp(key, "log_dir") == 0)    g_log_dir = luaL_checkstring(L, 3);  // RIME 用户目录
    // 日志总开关（schema: llm_rerank/log_enabled，lua 侧初始化配置时传入）
    else if (strcmp(key, "log_enabled") == 0) g_log_enabled.store(lua_toboolean(L, 3) != 0);
    return 0;
}

// ============================================================
// 模块入口
// ============================================================
#ifdef _WIN32
extern "C" __declspec(dllexport) int luaopen_rime_llm(lua_State * L) {
#else
extern "C" __attribute__((visibility("default")))
int luaopen_rime_llm(lua_State * L) {
#endif
    lua_newtable(L);

    lua_pushcfunction(L, lua_score);
    lua_setfield(L, -2, "score");

    lua_pushcfunction(L, lua_prepare);
    lua_setfield(L, -2, "prepare");

    lua_pushcfunction(L, lua_is_ready);
    lua_setfield(L, -2, "is_ready");

    // 光标上文旁路（2026-09-10 三层上文前两层）：函数注册与 score/prepare
    // 同款（setfield 直挂，不走 __newindex 属性语义）
    lua_pushcfunction(L, lua_llm_context);
    lua_setfield(L, -2, "llm_context");

    lua_pushcfunction(L, lua_kick_context);
    lua_setfield(L, -2, "kick_context");

    lua_pushcfunction(L, lua_edit_reset_context);
    lua_setfield(L, -2, "edit_reset_context");

    lua_pushcfunction(L, lua_fg_changed);
    lua_setfield(L, -2, "fg_changed");

    lua_pushcfunction(L, lua_click_happened);
    lua_setfield(L, -2, "click_happened");

    // 编码模式匹配（2026-09-30）：取代 min_code_len/max_code_len 区间，
    // 与 Rime 自身 auto_select_pattern 同语义（全串匹配）
    lua_pushcfunction(L, lua_match_code);
    lua_setfield(L, -2, "match_code");
    lua_pushstring(L, kDefaultCodePattern);
    lua_setfield(L, -2, "default_code_pattern");

    // 可写属性（model_path/max_ctx/n_threads/n_ctx/n_seq_max/min_tokens）
    // 严禁在此预填充为原始字段——Lua 的 __newindex 只对表中不存在的键
    // 触发，预填充过的键被赋值时只是 rawset 覆盖，永远到不了 C++ setter
    //（2026-08-31 终审定位：model_path 配置自插件版诞生即被此吞掉，一直
    // 靠默认路径恰好正确掩盖）。读由 __index 动态服务，写走 __newindex。
    lua_newtable(L);
    lua_pushcfunction(L, lua_index);
    lua_setfield(L, -2, "__index");
    lua_pushcfunction(L, lua_newindex);
    lua_setfield(L, -2, "__newindex");
    lua_setmetatable(L, -2);

    // 不在此启动模型加载（2026-08-31）：require 时 model_path 尚未由 lua
    // 写入，急切加载会锁定默认路径——加载改由 prepare/score 懒触发。
    return 1;
}
