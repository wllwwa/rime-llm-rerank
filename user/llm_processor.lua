-- llm_processor.lua — 上屏文字收集 + 预解码触发
-- 输出格式（一个场景一行）：
--   词1\t码1|词2\t码2|←|词3\t码3
--   | 分隔条目，\t 分隔词和码，← 退格，无码时码为空串

-- 状态归属（2026-10-01 改）：**每个组件实例一份，存在 env 上**。
-- 为什么不能再用脚本顶层 local：librime-lua 全进程只建一个 Lua 实例
-- （plugins/lua/src/modules.cc:93-106），顶层 local 是**所有窗口共享**的。多窗口
-- 下会互相覆盖：history/commit_base 串台；更严重的是 commit_hook_done 是模块级，
-- 第二个窗口的 processor 根本不连自己的 commit_notifier —— 上屏即预习在第二窗口
-- 完全失效。而 env 是 librime-lua 为**每个组件实例**建的表（raw_init 里
-- lua_newtable；每个 session 有自己的 Engine → 自己的组件实例 → 自己的 env），
-- 官方示例也写明可以往 env 上挂自定义字段。
-- 轻量诊断日志：写 C++ 侧那份 rime_llm_log.txt。
-- 格式刻意**不带毫秒**（分析脚本只认 "HH:MM:SS.mmm" 开头的行），所以这些行
-- 不会被算进 prepare/score 的统计，只是给人看的路标。多窗口验证就靠这些行：
-- 两个窗口各应出现一组"实例状态已建立 / commit 钩子已连接"。
-- 日志总开关（llm_rerank/log_enabled，processor 每次按键刷新；缺省 true）。
-- 关掉后 plog 与本文件内的其它日志写入一律跳过（C++ 侧的 rime_llm_log.txt
-- 明细由 llm_filter 经 cpp.log_enabled 传递同一个值后一并关闭）。
local log_enabled = true

local function plog(msg)
    if not log_enabled then return end
    local ok, ud = pcall(function() return rime_api.get_user_data_dir() end)
    if not ok or not ud or ud == "" then return end
    local f = io.open(ud .. "/rime_llm_log.txt", "a")
    if f then
        f:write(string.format("%s component: %s\n", os.date("%H:%M:%S"), msg))
        f:close()
    end
end

local function new_state()
    return {
        history = {},        -- 当前上屏词序列
        prev_hist = {},      -- 上次 history 快照
        commit_base = 0,     -- 上文基座: 只认 commit_history 中 base 之后的新词
                             -- (编辑键后旧词永久忽略——librime 无法清 commit_history)
        prev_input = "",     -- 上一轮的输入码
        pending_code = "",   -- 手动选词上屏码（输入变空瞬间捕获）
        last_full = "",      -- 最后满码（4码），顶屏时回退用
        last_prep_ctx = "",  -- 上次 prepare 的 context，避免重复调用
        commit_conn = nil,   -- 本实例的 commit_notifier 连接（fini 时断开）
        ctx_obj = nil,       -- 本会话的 Rime Context（读 llm_surrounding property）
        ctx_raw = nil,       -- 上次读到的 property 原始值（解码缓存键）
        ctx_info = nil,      -- 解码后的结构化上下文（前文/后文/选区）
    }
end

local fallback_state = nil   -- env 不可写时的兜底（退化为旧行为：单份共享状态）
local last_state = nil       -- 最近一次 processor 调用的实例状态（仅供 _G.llm_context_get 兜底）

-- 取本实例状态。写回后校验一次：若 env 不可写会**静默退化成"每次都是空
-- history"**——那会让每次评分都付一次同步解码（~70ms），是隐性性能回归，所以
-- 校验失败要告警并退回共享状态（等价于改造前的行为，不影响正确性）。
local function state_of(env)
    local st = env.__llm_state
    if st then return st end
    local cand = new_state()
    env.__llm_state = cand
    if env.__llm_state == cand then
        plog("实例状态已建立（每会话一份；多窗口会各出现一次）")
        return cand
    end
    if not fallback_state then
        fallback_state = cand
        local f = log_enabled
            and io.open(rime_api.get_user_data_dir() .. "/rime_llm_log.txt", "a")
            or nil
        if f then
            f:write(string.format("%s prepare: WARN env 不可写，组件状态退回全局共享\n",
                                  os.date("%H:%M:%S")))
            f:close()
        end
    end
    return fallback_state
end

local env_ctx_enabled = true  -- 外部光标上文旁路总开关（2026-09-29 定案：不做配置项、
                              -- 无关闭态；排障直接改此值）。Linux 下 fcitx5-rime
                              -- 桥接的 surrounding text 与 Windows COM/UIA 共用此开关。
local SPLIT = "|"
local TAB = "\t"
local BSP = "←"
local MAX_CODE = 4       -- 满码长度
local llm_prep = nil     -- 缓存的 llm 模块（全进程一份：lua 的 package.loaded）

local NAV_KEYS = { Left=true, Right=true, Up=true, Down=true,
                   Home=true, End=true, Page_Up=true, Page_Down=true }

-- 导航/换行类键（含组合导航 Control+Left / Shift+Home 等：以导航键名结尾的
-- repr 均算）。与删除类键分开：编辑键重置两者共用，训练数据标记只区分这两类
--（导航/回车 → 换行 "\n"；删除 → 退格 "←"）。
local function is_nav_ish_key(k)
    if NAV_KEYS[k] or k == "Return" or k == "KP_Enter" then return true end
    for name in pairs(NAV_KEYS) do
        if #k > #name and k:sub(-#name) == name then return true end
    end
    return false
end

-- 编辑位置变化键: 退格/删除 (删词) + 导航键 (光标移动/滚动) + 回车 (换行)
-- + Ctrl+z/y/v/x (撤销/重做/粘贴/剪切——改变光标前文本, 2026-09-12 信号层
-- 统一补入; 字符类含大写形态: Shift 按下时 keysym 为大写)
-- 这些键使会话上屏词序列不再代表光标前上文 → 上屏历史上文重置为空
local function is_edit_key(k)
    if k == "BackSpace" or k == "Delete"
       or k == "Control+BackSpace" or k == "Control+Delete"
       or k:match("^Control%+[zyvxZYVX]$")
       or is_nav_ish_key(k) then
        return true
    end
    return false
end

local get_context   -- 前置声明（trigger_prepare 要用，定义在文件后半）

-- === fcitx5 surrounding text 桥接解码（2026-10-02，Linux）===
-- fcitx5-rime 在 process_key() 前把当前 InputContext 的 surrounding text 编码
-- 写入当前 session 的 llm_surrounding property（单属性、版本化、十六进制），
-- 编码侧见 fcitx5-rime/src/rimestate.cpp。属性值格式：
--   "0"                                   桥接可用但上下文无效（客户端不支持/
--                                         敏感字段/非法偏移）
--   "1:<provided>:<cursor>:<anchor>:<hex>"  有效；cursor/anchor 为窗口内
--                                         Unicode 字符偏移（非字节偏移），
--                                         hex 为窗口内 UTF-8 文本十六进制
-- 其它形态一律视为不可用 → 回退上屏历史（向前兼容：将来新增版本号不影响旧码）。
local CTX_PROP = "llm_surrounding"

local function hex_decode(hex)
    if #hex % 2 ~= 0 or hex:find("[^0-9a-fA-F]") then return nil end
    return (hex:gsub("%x%x", function(cc) return string.char(tonumber(cc, 16)) end))
end

-- 按 Unicode 字符数把 text 切成"前 nchars 个字符"和"其余部分"（字节安全）
local function split_at_char(text, nchars)
    if nchars <= 0 then return "", text end
    local pos = utf8.offset(text, nchars + 1)  -- 第 nchars+1 个字符的字节起点
    if not pos then return text, "" end        -- nchars == #text 时可能是 nil
    return text:sub(1, pos - 1), text:sub(pos)
end

-- 解码 property 原始值；返回 nil 表示本键无可用桥接数据
local function decode_surrounding(raw)
    if type(raw) ~= "string" or raw == "" then return nil end
    if raw == "0" then
        return { valid = false, provided = true }
    end
    local provided, cursor_s, anchor_s, hex =
        raw:match("^1:(%d):(%d+):(%d+):(.*)$")
    if provided == nil then return nil end
    local text = hex_decode(hex)
    if not text then return nil end
    local n = utf8.len(text)                    -- UTF-8 校验；失败返回 nil
    local cursor, anchor = tonumber(cursor_s), tonumber(anchor_s)
    if not n or not cursor or not anchor or cursor > n or anchor > n then
        return nil
    end
    local before, after = split_at_char(text, cursor)
    local selected = ""
    if anchor ~= cursor then
        local lo = math.min(anchor, cursor)
        local hi = math.max(anchor, cursor)
        local lo_prefix = split_at_char(text, lo)
        local hi_prefix = split_at_char(text, hi)
        selected = hi_prefix:sub(#lo_prefix + 1)
    end
    return {
        valid = true, provided = provided == "1",
        text = text, cursor = cursor, anchor = anchor,
        before = before, after = after, selected = selected,
    }
end

-- 读取并缓存桥接上下文：桥接侧只在快照变化时改写 property，因此以原始值为
-- 缓存键，解码结果可跨按键复用（避免每键 hex 解码）。
local function read_bridge_context(st)
    local ctx = st and st.ctx_obj
    if not ctx or not ctx.get_property then return nil end
    local ok, raw = pcall(function() return ctx:get_property(CTX_PROP) end)
    if not ok then return nil end
    if raw == st.ctx_raw then return st.ctx_info end
    st.ctx_raw = raw
    st.ctx_info = decode_surrounding(raw)
    return st.ctx_info
end

local function reset_history(st)
    st.history = {}
    st.prev_hist = {}
end

-- 仅从 commit_history 快照重建 history（纯读 commit_base、写 history，无副作用）。
-- 按键路径的同步段与 commit_notifier 回调共用，保证两处算出的上文字符串
-- 完全一致（不一致会让 C++ 预解码永不命中 → 每次 score 走完整解码）。
local function rebuild_history(st, all)
    if not all then return end
    -- 引擎清空 commit_history（退格等）→ 基座重置重新计数
    if #all < st.commit_base then st.commit_base = 0 end
    if #all > st.commit_base then
        st.history = {}
        for i = st.commit_base + 1, #all do
            local entry = all[i]
            if entry and entry.text and #entry.text >= 1 then
                table.insert(st.history, entry.text)
            end
        end
    end
end

-- 异步预解码触发（归一化 + 去重，按键路径与编辑键重置路径共用）。
-- ctx 归一化与 llm_filter 一致 (去空白): C++ prep 命中 = token 序列比较,
-- 不一致会导致 prep 永远不命中 → 每次 score 完整解码 (~50ms)。
-- 中文无空白两者相同; 含英文/空格 (如 "Hello world 你好") 时保证一致。
local function trigger_prepare(st)
    if not (llm_prep and llm_prep.prepare) then return end
    local cur_ctx = (get_context(st) or ""):gsub('%s+', '')
    if cur_ctx ~= st.last_prep_ctx then
        st.last_prep_ctx = cur_ctx
        llm_prep.prepare(cur_ctx)
    end
end

-- 上屏即时预习（2026-10-01 修 wait 卡顿）：commit_notifier 在"上屏已写入
-- commit_history 之后"同步触发（librime 序：Context::Commit → ConcreteEngine::
-- OnCommit Push 历史 → 本回调），回调里能读到含新词的历史。回调只做"重建
-- 上文字符串 + 启动异步预习"——把 ctx decode 藏进用户想下一个词的间隙；
-- 此前预习要等下一次按键才启动，与同键 score 抢锁，实测 wait 100~167ms。
-- 注意：绝不能在此触发写语料/词频/prev_hist 等一次性副作用（processor 照旧）。
-- 2026-10-01 改：**每个实例连自己的 context**（原来用模块级 commit_hook_done
-- 只让第一个实例连上，第二个窗口因此完全不预习），并把回调写成捕获本实例状态
-- 的闭包；连接保存在 state 里，fini 时断开。
local function connect_commit_hook(env, st)
    if st.commit_conn then return end
    local ok, conn = pcall(function()
        return env.engine.context.commit_notifier:connect(function(ctx)
            local ch = ctx and ctx.commit_history
            if ch then
                local ok2, all = pcall(function() return ch:to_table() end)
                if ok2 then rebuild_history(st, all) end
            end
            trigger_prepare(st)
        end)
    end)
    if ok and conn then
        st.commit_conn = conn
        plog("commit 钩子已连接（本实例的 context；多窗口会各出现一次）")
    end
end

local function append_raw(text)
    local f = io.open(rime_api.get_user_data_dir() .. "/llm_training.txt", "a")
    if f then
        f:write(text)
        f:close()
    end
end

-- === 用户词频计数 (llm_filter 排序融合用, 2026-08-19; 2026-08-21 改 Rime 时间衰减) ===
-- _G.user_word_freq: 词 → {c=累计次数, d=衰减计数 dee, t=最后提交 tick}
-- _G.user_freq_tick: 全局 tick (每词提交 +1, 同 rime userdb UpdateTickCount)
-- 衰减算法 = librime algo::formula_d (user_dictionary.cc 调频同源):
--   提交: dee = 1 + dee·exp((t_old - t_now)/200)   时间常数 τ=200 tick
--   查询: eff = dee·exp((t_word - t_now)/200)      (未提交期间持续衰减)
-- 每 20 个新词重写落盘 user_freq.tsv (崩溃最多丢 19 次计数)。仅计含中文的词
-- (判定为非 ASCII 即计, 与 Rime userdb "任何词条提交推 tick" 同源口径——
-- 全角标点也计, 有意为之; 2026-09-03 tick 含一字词定案同源原则)。
-- 格式: 首行 "#tick=N", 数据行 "词\t累计\tdee\ttick"; 兼容旧版 "词\t次数"
-- (迁移为 dee=次数, tick=当前 — 视为刚提交过)。
-- 落盘修剪 (2026-09-04): eff < 1e-3 的冷词条不再写 (对融合分影响
-- < 0.002 nats, 远低于任何实际分差) — 文件有界, 旧冷词条自然淘汰。
local FREQ_TAU = 200
local FREQ_FLUSH = 20
local freq_dirty = 0
local function freq_file()
    return rime_api.get_user_data_dir() .. "/user_freq.tsv"
end
local function freq_load()
    _G.user_word_freq = _G.user_word_freq or {}
    local tick, maxt, legacy = nil, 0, {}
    local f = io.open(freq_file(), "r")
    if f then
        for line in f:lines() do
            local t = line:match("^#tick=(%d+)$")
            if t then
                tick = tonumber(t)
            else
                local w, c, d, tk = line:match("^(.-)\t(%d+)\t([%d%.]+)\t(%d+)$")
                if w and c and d and tk then
                    _G.user_word_freq[w] = { c = tonumber(c), d = tonumber(d), t = tonumber(tk) }
                    if tonumber(tk) > maxt then maxt = tonumber(tk) end
                else
                    local w2, n = line:match("^(.-)\t(%d+)$")
                    if w2 and n then legacy[#legacy + 1] = { w2, tonumber(n) } end
                end
            end
        end
        f:close()
    end
    _G.user_freq_tick = tick or maxt
    for _, e in ipairs(legacy) do
        _G.user_word_freq[e[1]] = { c = e[2], d = e[2], t = _G.user_freq_tick }
    end
end
local function freq_save()
    local f = io.open(freq_file(), "w")
    if f then
        f:write("#tick=" .. (_G.user_freq_tick or 0) .. "\n")
        local now = _G.user_freq_tick or 0
        for w, e in pairs(_G.user_word_freq) do
            local eff = e.d * math.exp((e.t - now) / FREQ_TAU)
            if eff >= 1e-3 then  -- 修剪: 衰减到噪声级的冷词条不落盘
                f:write(string.format("%s\t%d\t%.3f\t%d\n", w, e.c, e.d, e.t))
            end
        end
        f:close()
    end
end
local function freq_bump(word)
    if not _G.user_word_freq then freq_load() end
    _G.user_freq_tick = (_G.user_freq_tick or 0) + 1
    local e = _G.user_word_freq[word]
    if not e then
        e = { c = 0, d = 0, t = 0 }
        _G.user_word_freq[word] = e
    end
    e.d = 1 + e.d * math.exp((e.t - _G.user_freq_tick) / FREQ_TAU)
    e.c = e.c + 1
    e.t = _G.user_freq_tick
    freq_dirty = freq_dirty + 1
    if freq_dirty >= FREQ_FLUSH then
        freq_save()
        freq_dirty = 0
    end
end

-- 衰减有效计数 (llm_filter 融合用): rime formula_d 查询式
_G.user_freq_eff = function(w)
    local e = _G.user_word_freq and _G.user_word_freq[w]
    if not e or e.d <= 0 then return 0 end
    return e.d * math.exp((e.t - (_G.user_freq_tick or e.t)) / FREQ_TAU)
end

local function find_overlap(prev, curr)
    local np, nc = #prev, #curr
    for len = math.min(np, nc), 0, -1 do
        local match = true
        for j = 1, len do
            if prev[np - len + j] ~= curr[j] then
                match = false
                break
            end
        end
        if match then return len end
    end
    return 0
end

local function processor(key, env)
    if key:release() then return 2 end
    local sc = env.engine.schema.config
    -- 日志总开关（每次按键刷新，重新部署后即生效；缺省 true = 与加开关前一致）。
    -- 必须在 state_of 之前读：组件实例首次建立时的 plog 用的就是本次的值。
    local logv = sc:get_bool("llm_rerank/log_enabled")
    log_enabled = (logv == nil) and true or logv
    -- 本实例状态（每个 session 一份）；同时记为"最近一次调用的状态"，
    -- 供 llm_filter 经 _G.llm_context_get 兜底读取（见文件末说明）
    local st = state_of(env)
    last_state = st
    -- 缓存本会话 Context（读 llm_surrounding property 用；每实例一份）
    if not st.ctx_obj then st.ctx_obj = env.engine.context end

    -- 上文检查 + 预解码 (每次按键): commit_history 变化 → 立即异步预解码
    local enabled = sc:get_bool("llm_rerank/enabled") or false
    if not enabled then
        llm_prep = nil  -- 释放已加载的 DLL 引用
    else
        if not llm_prep then
            local ok, result = pcall(require, "rime_llm")
            if not ok then
                -- Linux shim（2026-10-01 移植）：require 失败时绝对路径
                -- loadlib；filter 侧若已写过 package.loaded 则上面 require
                -- 直接命中同一实例，不会走到这里
                local fn = package.loadlib("/usr/lib/rime-plugins/librime-llm.so",
                                           "luaopen_rime_llm")
                if fn then ok, result = pcall(fn) end
                if ok and result then package.loaded.rime_llm = result end
            end
            if ok then
                -- 日志目录: RIME 用户目录 (与 filter 共用同一模块实例)
                local okd, ud = pcall(function() return rime_api.get_user_data_dir() end)
                if okd and ud and ud ~= "" then result.log_dir = ud end
                llm_prep = result
            end
        end
        -- 每个实例都要连**自己的** context（connect_commit_hook 内部靠 state 去重）。
        -- 绝不能放进上面那个"首次 require"块：那里有模块级 llm_prep 守卫，第二个
        -- 窗口根本进不去，它的 commit 钩子就永远连不上——而这正是本次要修的 bug，
        -- 真机日志留下过证据（只出现一组"commit 钩子已连接"，第二个窗口的上屏预习
        -- 全程失效）。
        if llm_prep then connect_commit_hook(env, st) end
        -- model_path 每键轻量同步（仅变化时写，2026-08-31 修配置不生效的
        -- 根因）：路径只在"一次性 require 块"里读的话，首个触发 require 的
        -- 会话（部署期编译会话/配置未就绪的瞬间）会把默认路径永久固化到
        -- 模块级 llm_prep，之后的按键永不重读。prepare 是懒加载触发点，
        -- 同步必须发生在它之前——本键同步、本键 prepare，路径必然就位。
        local mp = sc:get_string("llm_rerank/model_path")
        if mp and mp ~= "" and llm_prep and llm_prep.model_path ~= mp then
            llm_prep.model_path = mp
            -- 模型将卸载重载, 旧模型的评分缓存作废 (2026-09-04; 源码版
            -- unload_model 清 s_cache 同理)——不清则同码重打期间继续用旧分
            _G.llm_filter_cache = nil
        end
        -- cpu_cores / max_tokens 同理必须赶在 prepare 之前同步（2026-10-01）：
        -- llama 上下文的线程数与 ctx 长度只在**创建时**读一次，而加载是
        -- prepare 触发的、filter 要等第一次评分才写这两个值——谁先到看运气。
        -- 实测同一份 cpu_cores: 8 的配置，日志里出现过 threads=8 也出现过
        -- threads=4（后者连带 ctx 上限也没生效）。C++ 侧改成"值变化即卸载"，
        -- 这里保证首次加载前就把值写好。
        local thr = sc:get_int("llm_rerank/cpu_cores")
        if thr and thr > 0 and llm_prep and llm_prep.n_threads ~= thr then
            llm_prep.n_threads = thr
            _G.llm_filter_cache = nil
        end
        local mtk = sc:get_int("llm_rerank/max_tokens")
        if mtk and mtk > 0 and llm_prep and llm_prep.max_ctx ~= mtk then
            llm_prep.max_ctx = mtk
            _G.llm_filter_cache = nil
        end
    end
    -- 鼠标点击检测（2026-09-11 深夜用户定案：任何点击=光标可能移动→
    -- 清历史上文兜底，不做例外排除——候选窗/工具栏点击也清，上下文
    -- 宁可变短不可错；2026-09-12 起同时失效 COM/UIA 快照+重读——真文
    -- 快照也是旧时刻的，点击移光标后同样冒充）
    if llm_prep and llm_prep.click_happened and llm_prep.click_happened() then
        local ch0 = env.engine.context.commit_history
        if ch0 then st.commit_base = #ch0:to_table() end
        reset_history(st)
        -- 点击候选窗选词场景：commit kick 刚刷新的正确快照会被废，但
        -- kick 50ms 后读回同样正确的文本（点击已处理完、含新上屏词），自愈
        if llm_prep.edit_reset_context then
            pcall(llm_prep.edit_reset_context)
        end
    end

    -- 上文检查 + 预解码 (每次按键): commit_history 变化 → 立即异步预解码
    trigger_prepare(st)

    local ctx = env.engine.context
    local ch = ctx.commit_history
    if not ch then return 2 end

    -- 跨应用切换检测（2026-09-11，DLL fg_changed 内部比前台 pid）：插件版
    -- lua 无焦点事件，切换后旧窗口的 history 残留会冒充新窗口上文（真机
    -- 踩坑：切新文档首词标 AI·历史而非无徽章）。切到新进程 → 重置历史 +
    -- commit_history 旧词记为基座（只认切换后的新词，与编辑键同款语义）。
    -- 比 pid 不比 HWND：同应用内对话框/多文档窗口不误清。
    if llm_prep and llm_prep.fg_changed and llm_prep.fg_changed() then
        local all0 = ch:to_table()
        st.commit_base = #all0
        reset_history(st)
    end

    -- 追踪满码（顶屏时回退用）
    if ctx.input ~= "" and #ctx.input >= MAX_CODE then
        st.last_full = ctx.input
    end

    -- 输入变空 → 捕获本次上屏码（手动选词、Tab、数字键）
    if st.prev_input ~= "" and ctx.input == "" then
        st.pending_code = st.prev_input
        st.last_full = ""  -- 已消费
    end
    st.prev_input = ctx.input

    -- 退格 / Delete / 导航键 (composition 为空时): 编辑位置变化
    --   rime 来源上文重置为空 (会话上屏词序列不再代表光标前上文), 立即重新预解码
    --   return 0 放行按键 (导航键/退格由后续 processor 正常处理, 光标移动),
    --   本分支已提前返回, 后续 commit_history 同步不会执行 (不覆盖重置;
    --   退格后剩余词不会作为新词重新记录训练数据)
    if ctx.input == "" and is_edit_key(key:repr()) then
        local k = key:repr()
        if is_nav_ish_key(k) then
            if #st.history > 0 then
                append_raw("\n")
            end
        elseif #st.history > 0 then
            append_raw(SPLIT .. BSP)
        end
        reset_history(st)
        -- 编辑键: 当前 commit_history 位置记为基座——旧词永久忽略
        -- (librime lua 无法清 commit_history, 旧词留在引擎里;
        --  退格会被引擎清空 commit_history → 同步段检测缩短自动重置基座)
        st.commit_base = #ch:to_table()
        -- 编辑后重打相同词 (ctx+input 相同) 必须重新推理: 清空 filter 结果缓存
        _G.llm_filter_cache = nil
        -- 2026-09-12 信号层统一: COM/UIA 快照失效 + kick 延迟重读(~50ms)——
        -- 退格/导航/撤销/粘贴后旧快照不再 2.5s 新鲜窗内冒充真文; 打首词
        -- (人手 >300ms) 时新快照已就绪, 首词即有正确 AI·UIA/COM 上文
        if llm_prep and llm_prep.edit_reset_context then
            pcall(llm_prep.edit_reset_context)
        end
        if sc:get_bool("llm_rerank/debug_fusion") then
            pcall(function()
                local f = io.open(rime_api.get_user_data_dir()
                    .. "/rime_llm_debug.txt", "a")
                if f then
                    f:write(string.format(
                        "%s|reset|编辑键(%s)→上文重置+缓存清空\n",
                        os.date("%H:%M:%S"), key:repr()))
                    f:close()
                end
            end)
        end
        -- 上文已重置 → 立即异步预解码 (空上文)
        trigger_prepare(st)
        return 0
    end

    -- 同步 commit_history（history 重建与 commit 回调共用 rebuild_history，
    -- 含"引擎清空 → 基座重置"；写语料/词频/prev_hist 等副作用仍只在此执行）
    local all = ch:to_table()
    rebuild_history(st, all)
    if all and #all > st.commit_base then
        local overlap = find_overlap(st.prev_hist, st.history)
        local new_words = {}
        for i = overlap + 1, #st.history do
            table.insert(new_words, st.history[i])
        end

        if #new_words > 0 then
            if overlap == 0 then
                append_raw("\n")
            end

            local parts = {}
            for _, w in ipairs(new_words) do
                -- 含中文才分配码，跳过纯英文/数字/标点
                local has_chinese = w:match("[^\1-\127]")
                local code = ""
                if has_chinese then
                    -- 优先手动捕获的码，其次满码（顶屏回退），用完即清
                    code = st.pending_code
                    if code == "" then
                        code = st.last_full
                    end
                end
                st.pending_code = ""
                st.last_full = ""
                -- 词频计数: 仅含中文的词 (含英文/数字/标点不参与融合)
                if has_chinese then
                    if not _G.user_word_freq then freq_load() end  -- 诊断读数前先就位
                    local before = _G.user_freq_eff and _G.user_freq_eff(w) or 0
                    local t_before = _G.user_freq_tick or 0
                    freq_bump(w)
                    local dbg_flag = sc:get_bool("llm_rerank/debug_fusion")
                    if dbg_flag then
                        pcall(function()
                            local f = io.open(rime_api.get_user_data_dir()
                                .. "/rime_llm_debug.txt", "a")
                            if f then
                                f:write(string.format(
                                    "%s|commit|词=%s|码=%s|eff %.3f→%.3f|tick %d→%d|flush脏%d/20\n",
                                    os.date("%H:%M:%S"), w, code, before,
                                    _G.user_freq_eff(w), t_before,
                                    _G.user_freq_tick or 0, freq_dirty))
                                f:close()
                            end
                        end)
                    end
                end
                -- 真实候选窗快照 (llm_filter 记录): 用截断前的完整码匹配
                -- (filter 记录的是完整 input; 训练样本带真实候选窗 词\t码\t候选1,候选2,...,
                -- LLM 重排目标 = 在窗内把正确词排第一; 无快照/不匹配回退旧格式 仅词+码)
                local window_cands = ""
                if code ~= "" and _G.llm_last_window
                        and _G.llm_last_window.input == code then
                    window_cands = TAB .. table.concat(_G.llm_last_window.cands, ",")
                    _G.llm_last_window = nil  -- 已消费
                end
                -- 单字 3 码只需前 2 码（第 3 码是形码，由字本身决定）
                if #w == 1 and #code >= 3 then
                    code = code:sub(1, 2)
                end
                table.insert(parts, w .. TAB .. code .. window_cands)
            end
            local sep = (overlap > 0 and SPLIT or "")
            append_raw(sep .. table.concat(parts, SPLIT))

            -- 上屏后立即刷新 COM/UIA 快照（2026-09-10）：新词此刻已入
            -- 文档/控件，读链 6-13ms 在后台线程完成，下一键即有新鲜上文
            if llm_prep and llm_prep.kick_context then
                pcall(llm_prep.kick_context)
            end
        end

        if #new_words == 0 and #st.history < #st.prev_hist and #st.history < 3 then
            st.pending_code = ""
            st.last_full = ""
            append_raw("\n")
        end

        st.prev_hist = {}
        for _, v in ipairs(st.history) do table.insert(st.prev_hist, v) end
    end

    return 2
end

get_context = function(st)
    -- 三层上文（2026-09-10 全功能；2026-10-02 补 Linux 桥接层）：COM（Word/WPS
    -- 文字文档模型）→ UIA TextPattern（通用现代应用）→ fcitx5 surrounding text
    -- （Linux，fcitx5-rime 桥接的 llm_surrounding property）→ 上屏历史（兜底）。
    -- 前两层由 DLL llm_context() 完成（前台门控 + 快照新鲜度 + 来源进程校验都在
    -- C++，按键路径零阻塞）；桥接层只读本会话 property（同步、无 IPC）；
    -- 都拿不到才回退历史。旁路无常关配置（2026-09-29 定案删 com_context
    -- 配置项；排障改文件头 env_ctx_enabled 重部署）
    if env_ctx_enabled and llm_prep and llm_prep.llm_context then
        local ok, t, src = pcall(llm_prep.llm_context)
        if ok and type(t) == "string" and #t > 0 then
            return t, src
        end
    end
    if env_ctx_enabled then
        local info = read_bridge_context(st)
        -- 必须"有效且含非空白字符"才算应用上文：文本编辑器空行等场景下，应用
        -- 上报的 surrounding text 可能只有换行/空格（before="\n"），此时实际
        -- 上文为空；若判为 fcitx5 会出现"没有上文却显示 AI·应用"的徽章。
        -- 与其它无效情况一致，回退上屏历史（徽章 AI·历史）。
        if info and info.valid and info.before:find("%S") then
            return info.before, "fcitx5"
        end
    end
    return table.concat(st.history, ""), "rime"
end

-- llm_filter 取上文的入口。**它是全进程单值**：多窗口下可能读到另一个窗口最近
-- 一次调用写的上文。本可以改成"每会话 Context 属性"通道，但实测不可行——
-- Context::set_property 会触发 ConcreteEngine::OnPropertyUpdate →
-- message_sink_("property", ...) + LOG(INFO)（src/rime/engine.cc:154-162），
-- 每键一次通知客户端，代价不可接受；而 librime-lua 也不提供稳定的每会话标识
-- （LuaType<T*>::pushdata 每次访问都新建 userdata，身份不稳定，不能当表键）。
-- 单窗口下这就是本会话的上文，与改造前一致；多窗口的残余影响见 notes/fixes。
_G.llm_context_get = function()
    if last_state then return get_context(last_state) end
    return "", "rime"
end

-- 结构化上下文出口（前文/后文/选区/光标/是否由客户端提供）：当前 filter 只用
-- 前文参与评分，完整字段留给后续编辑类能力（如选区替换）。注意只有 fcitx5
-- 桥接层提供结构化字段；Windows COM/UIA 与历史回退只有整段文本。
_G.llm_context_info = function()
    if last_state then return read_bridge_context(last_state) end
    return nil
end

-- 组件析构时调用（librime-lua 在 LuaProcessor::~LuaProcessor 里调 fini(env)）：
-- 断开本实例的 commit_notifier 连接。boost::signals2 的槽随 Context 销毁而释放，
-- 所以这里主要是"重连前先断"的卫生；返回表形式 {'func','fini'} 是 librime-lua
-- 支持的模块形状（plugins/lua/src/lua_gears.cc:121-166）。
local function fini(env)
    local st = env.__llm_state
    if st and st.commit_conn then
        pcall(function() st.commit_conn:disconnect() end)
        st.commit_conn = nil
    end
end

return { func = processor, fini = fini }
