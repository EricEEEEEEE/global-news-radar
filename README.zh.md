# 全球实时信号雷达

常驻的全球金融突发雷达。它只推送过去约一小时内、不能等到日报的事件；不做价格面板、
技术位、持仓分析、长期展望或综合晨报。

English: [README.md](README.md)

## 生产边界

- 只读公开 RSS/API，不连接券商，不交易。
- P0 由确定性规则判定；官方源可单源确认，媒体源必须在 20 分钟内由两个独立主流来源确认。
- P1 在两小时窗口内至少积累两个不同事件才合并发送。
- 21:00–21:30 强制静默，静默结束后重新校验时效。
- LLM 只翻译、压缩已确认事实。它不能决定级别、event key、数字或因果。
- Telegram 采用纯 HTML 文字；不生成图片。每次尝试先写 HTML/JSON outbox。
- 发送成功后才写 delivery/topic 去重状态。失败进入 SQLite 重试队列。
- SQLite 是雷达真相源；`state/radar_events.jsonl` 是给下游日报的只读导出。

## 数据源

- Primary：Federal Reserve、ECB、SEC、CFTC、Bank of Japan RSS。
- Structured：FMP economic/earnings calendar。只有带可验证发布时间的记录才会通过
  75 分钟 freshness gate。默认关闭。
- Discovery：一组商业媒体 RSS（CNBC/MarketWatch/FT/BBC/Al Jazeera/SCMP/Investing/
  CoinDesk）+ 最多三个 Google News RSS 查询；只有当这一批总条数低于
  `discovery.gdelt_min_items` 时才回落 GDELT。

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python main.py once --readonly
```

`.env` 里留空的行等于「没设」，会回落到 `config/radar.yaml`，所以只填你需要的那几行。
不要提交 `.env`。生产文件权限必须为 `600`。

可选采集器默认关闭，新克隆不会误烧别人的 API 额度。要开就在 `.env` 里跟凭证放一起：

```dotenv
RADAR_FMP_ENABLED=true
RADAR_LLM_ENABLED=true
```

开关属于运维配置，不放进 tracked 文件——否则每个部署都要想办法让它不进自己的 commit。
`LLM_*` 取代了早期的 `FE_LLM_*`，旧名仍然认，升级不会让摘要静默失效。

聊天群不使用 Telegram forum topic 时，thread id 填 `0`。

## 命令

```bash
# 一轮完整扫描，不发 Telegram，但会写状态：标记 seen、推进 topic 冷却、写心跳。
# 对着生产 state 目录跑会让守护进程漏掉这一批新闻。
.venv/bin/python main.py once

# 只看不动：不标 seen、不写 topic/心跳、不消耗 ETag 缓存，可安全对生产 state 执行
.venv/bin/python main.py once --readonly

# 生产 Supervisor 入口
.venv/bin/python main.py run --send

# heartbeat 检查
.venv/bin/python main.py health --max-age-seconds 600

# 存活看门狗，异常推 monitor topic（不带 --send 只检查不发）
.venv/bin/python main.py watchdog --max-age-seconds 900 --send

# 导出下游日报兼容 ledger
.venv/bin/python main.py export
```

首次启动只建立当前 75 分钟候选的 baseline，绝不补推历史。

## 存活看门狗

守护进程能报自己的源故障，但**已经死掉的进程报不了自己死了**，卡在某一轮的同样报不了。
`watchdog` 只补这一个洞，因此刻意不依赖被监控的东西：只读 `state/heartbeat.json`，
自己写一个独立的小 JSON 状态，**不开守护进程的 SQLite，也不写它的 outbox**——数据库
被锁或损坏本身就是它必须能报出来的故障之一，靠那个文件发告警等于没有。

必须挂 cron，不要挂在管着守护进程的那个 Supervisor 下面，否则 Supervisor 一倒，
警报跟着一起没了：

```cron
*/10 * * * * cd /path/to/global-news-radar && ./.venv/bin/python main.py --root . watchdog --send >> logs/watchdog.log 2>&1
```

告警只在**状态跳变**时发，不是每轮都发：第一次失败发一条到 monitor topic，
`watchdog_alert_cooldown_hours`（默认 6h）盖住中断的其余时间，恢复后补一条
`✅ 新闻雷达恢复` 带中断时长把事件关掉。从没告警过的抖动不会单独冒出一条「恢复」。
退出码 0 健康 / 1 不健康，cron 邮件或外部 ping 服务也能直接用。

**边界**：整台机器宕了它报不出来——跑在这台机器上的任何东西都报不出来。它看的是
守护进程，不是主机。

## 源健康告警

告警阈值按源分级，因为同一个「连续失败 N 次」在不同轮询节奏下代表的时间完全不同：

- 官方源 + FMP 用 `source_failure_alert_after`（3）。官方源每轮跑，约 6 分钟。
- Discovery 源用 `source_failure_alert_after_discovery`（6）。它们每
  `discovery.interval_cycles` 才跑一轮，6 次 ≈ 1 小时；十几选一的商业 CDN 抖动
  半小时不值得打断人。

源恢复后会补发一条 `✅ 数据源恢复`，带中断时长，告警不会一直挂着无人关闭。恢复即
关闭该次事件：下一次故障是新事件，可以重新告警，但必须先过
`source_realert_minutes`（60），避免反复抖动的源刷屏。

## 状态和证据

- `state/radar.sqlite3`：item/topic/delivery/P1/source-health 状态。
- `state/heartbeat.json`：最近一轮心跳和源错误。
- `state/radar_events.jsonl`：只包含真实发送成功的事件。
- `outbox/latest.html`：最近一次 Telegram HTML。
- `outbox/latest.json`：来源、时间、event key、content hash、message ID。
- `logs/radar.log`：业务日志，8MB × 5 份自轮转；Supervisor 另有 stdout/stderr 轮转。

`source-error:*` 与 `source-recovery:*` 基础设施告警不入账 `radar_events.jsonl`。
`item_retention_days`（默认 7）和 `delivery_retention_days`（默认 90）控制 SQLite 与
outbox 的自动清理。所有日志、告警文案和心跳都经过 secret 脱敏，API key 只会显示为 `***`。

## VisualSpec

`config/visual_spec.json` 按 TG Watch Visual Compiler 选择 Telegram HTML 文本：

1. P0 第一行直接写事件，不以系统标签开场；
2. 证据区只放核心事实与即时影响；
3. `发生`与`雷达`时间分开，原始来源可点击；
4. P1 每条动态独立绑定事实、发生时间和来源。

每次 outbox manifest 都保存本次消息的 source-bound `visual_spec`。HTML 被 Telegram
拒绝时自动降级 plain text；P0/P1 可见字符均限制在 400 以内。

## 测试

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q main.py radar tests
```

## 已知边界

- Telegram Bot API 无端到端 idempotency key。极罕见的「Telegram 已接收但客户端在收到
  response 前断线」可能造成一次重试重复；系统选择不静默丢失 P0。
- FMP calendar 若只有日期而没有发布时间，会被 freshness gate 拒绝，不会伪装成实时。
- ETF 日流量和链上鲸鱼不在此服务内；它们需要独立结构化 monitor。
- 没有可靠行情证据时只写「影响方向」，不声称市场已经因该事件涨跌。

## 回滚

```bash
supervisorctl stop global_news_radar
```

停止服务不会删除 SQLite、日志或 outbox。
