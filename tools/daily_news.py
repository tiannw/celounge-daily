"""
策论阁 · 每日时政自动提取（方案C：GitHub Actions 定时 + 静态托管）

数据源（均为官方，国内可直连）：
  1) 央视网《新闻联播》每日列表  https://tv.cctv.com/lm/xwlb/day/YYYYMMDD.shtml
     解析当日全部新闻标题（含完整版[视频]前缀，已清洗）
  2) 当日完整版视频页（取第一个 VIDE 链接）到「节目主要内容」编号提要

流程：抓取 -> 清洗 -> DeepSeek 提取时政卡 -> App 格式 JSON -> web/data/daily/YYYY-MM.json
App 端定时从 GitHub raw CDN 拉取该文件增量导入。

用法：
  python tools/daily_news.py                 # 默认处理今天（若无广播则回退昨天）
  python tools/daily_news.py --date 2026-08-03
  python tools/daily_news.py --no-llm        # 只抓取整理源文本，不调 LLM（调试用）
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import requests
from openai import OpenAI

# ============================================================
# 配置（GitHub Actions 中通过 secrets 注入，本地需设置同名环境变量）
# ============================================================
DEEPSEEK_API_KEY = os.environ["CELOUNGE_DEEPSEEK_API_KEY"]  # 无环境变量时直接报错，不再内置 key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.environ.get("CELOUNGE_DEEPSEEK_MODEL", "deepseek-v4-flash")

BASE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.normpath(os.path.join(BASE, "..", "web", "data"))
DAILY_DIR = os.path.join(WEB_DIR, "daily")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
}

STRUCTURE_PROMPT = """你是公考时政专家。任务：从《新闻联播》当日节目内容中提取「时政知识点」复习卡。

输入是一天的新闻联播标题列表 + 编号提要。每条新闻可能讲同一政策/事件的多个角度。

要求：
1. 只提取值得考的时政知识点，忽略纯娱乐、纯体育花絮、无关国际八卦。
2. 每条知识点一张卡：
   - title: 知识点标题（简洁，30字内，含关键主体与事件，如「国务院成立彭水"7·17"山体崩塌灾害调查评估组」）
   - content: 核心要点（2-5条，务必保留时间、数字、关键人物、专有名词、政策关键词，每条要点一行）
   - importance: 1-5（越高越重要，如国务院成立调查组/重大政策/领导人活动=5）
   - tags: 分类（学习/会议/文件/科技/外交/经济/法治/生态/民生/党建/综合）
   - mnemonic: 挖坑提示或记忆口诀（如「贴息≠免息」「提高≠降低」），没有则留空字符串
   - time_window: 时效窗口（如「近半年重点关注」），没有则留空
3. 多条新闻讲同一事件时合并为一张卡，在 content 中补充不同角度。
4. 输出纯 JSON 数组，不要 markdown 标记。"""



# ============================================================
# 抓取与解析
# ============================================================
def fetch(url: str, timeout: int = 30) -> str:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            # 央视页面 content-type 无 charset，requests 会误判为 ISO-8859-1，
            # 统一按 UTF-8 解码字节。
            return r.content.decode("utf-8", errors="replace")
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return ""


def parse_day_titles(html: str) -> list:
    """从当日列表页提取新闻标题，清洗前缀。"""
    titles = []
    seen = set()
    for m in re.finditer(r'<a[^>]*href="([^"]*VIDE[^"]*)"[^>]*>(.*?)</a>', html, re.S):
        text = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        # 跳过整场联播入口（不含 [视频] 前缀的「完整版《新闻联播》」）
        if "《新闻联播》" in text and "[视频]" not in text:
            continue
        text = text.replace("完整版[视频]", "").replace("完整版", "").strip()
        text = re.sub(r"^\[视频\]", "", text).strip()
        if text and text not in seen:
            seen.add(text)
            titles.append(text)
    return titles


def parse_tiyao(html: str) -> list:
    """从视频页提取「节目主要内容」编号提要列表。"""
    body = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    body = re.sub(r"<style.*?</style>", "", body, flags=re.S)
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"&\w+;", " ", body)
    body = re.sub(r"\s+", "", body)

    idx = body.find("节目主要内容")
    if idx < 0:
        return []
    seg = body[idx:idx + 4000]
    # 有些页面在文末会重复一遍「本期节目主要内容」，截断
    second = seg.find("节目主要内容", 10)
    if second > 0:
        seg = seg[:second]
    # 去掉页脚垃圾（《新闻联播》2026080419:00 / 视频简介 / 来源等）
    for cut in ("《新闻联播》", "视频简介", "来源：央视网"):
        k = seg.find(cut)
        if k > 0:
            seg = seg[:k]
    seg = re.sub(r"^.*?[:：]", "", seg, count=1)  # 去掉「节目主要内容：」头

    items = []
    for part in seg.split("；"):
        part = part.strip()
        m = re.match(r"^(\d{1,2})[.、．](.*)$", part)
        if m:
            items.append(m.group(2).strip())
        elif items and part:
            items[-1] += "；" + part  # 子条目（（1）（2））续接到上一条
    return items


def get_daily_source(date: datetime):
    """返回 (标题列表, 提要列表)。标题为空时抛异常（当日无广播）。"""
    day = date.strftime("%Y%m%d")
    page_url = "https://tv.cctv.com/lm/xwlb/day/%s.shtml" % day
    html = fetch(page_url)
    titles = parse_day_titles(html)

    if not titles:
        raise RuntimeError("%s 无新闻联播列表（可能未播出）" % date.strftime("%Y-%m-%d"))

    # 取完整版视频页提取提要
    tiyao = []
    m = re.search(r'href="(https?://[^"]*VIDE[^"]*\.shtml)"', html)
    if m:
        try:
            video_html = fetch(m.group(1))
            tiyao = parse_tiyao(video_html)
        except Exception:
            tiyao = []
    return titles, tiyao


# ============================================================
# DeepSeek 提取
# ============================================================
def llm_extract(source_text: str) -> list:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    last_err = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": STRUCTURE_PROMPT},
                    {"role": "user", "content": source_text},
                ],
                temperature=0.3,
                max_tokens=16384,
            )
            content = (r.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            start, end = content.find("["), content.rfind("]")
            if start >= 0 and end > start:
                content = content[start:end + 1]
            data = json.loads(content)
            if isinstance(data, list):
                return [c for c in data if isinstance(c, dict) and c.get("title")]
            last_err = "not a list"
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
    print("  WARN: chunk failed after 3 tries: %s" % last_err, flush=True)
    return []


def to_app_card(card: dict, month: str) -> dict:
    raw = card.get("content") or ""
    if isinstance(raw, list):
        content_txt = "\n".join(str(x) for x in raw if str(x).strip()).strip()
    else:
        content_txt = str(raw).strip()
    mnemonic = (card.get("mnemonic") or "").strip()
    back = content_txt
    if mnemonic:
        back += "\n\n【挖坑/口诀】" + mnemonic
    raw_tags = card.get("tags", "")
    if isinstance(raw_tags, list):
        tags = ",".join(str(x) for x in raw_tags if str(x).strip())
    else:
        tags = str(raw_tags).strip()
    return {
        "front": (card.get("title") or "").strip(),
        "back": back,
        "card_type": "common_knowledge",
        "deck_id": "deck_current_%s" % month,
        "tags": tags,
        "fields_json": json.dumps({
            "category": "时政",
            "explain": content_txt,
            "mnemonic": mnemonic,
            "importance": card.get("importance", 3),
            "time_window": card.get("time_window", ""),
        }, ensure_ascii=False),
        "reverse_hint": content_txt[:50],
    }


# ============================================================
# 落盘合并
# ============================================================
def merge_and_save(month: str, cards: list):
    os.makedirs(DAILY_DIR, exist_ok=True)
    path = os.path.join(DAILY_DIR, "%s.json" % month)
    merged = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            merged = json.load(f)
    seen = {c.get("front") for c in merged}
    added = 0
    for c in cards:
        if c.get("front") in seen:
            continue
        seen.add(c["front"])
        merged.append(c)
        added += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    return merged, added


def main():
    args = sys.argv[1:]
    no_llm = "--no-llm" in args

    # 目标日期：--date 指定 / 今天；今天无广播则回退昨天
    target = None
    if "--date" in args:
        target = datetime.strptime(args[args.index("--date") + 1], "%Y-%m-%d")
    else:
        target = datetime.now()

    titles, tiyao = None, []
    for d in [target, target - timedelta(days=1)]:
        try:
            titles, tiyao = get_daily_source(d)
            if titles:
                print("[源] %s 抓取成功：%d 条标题, 提要 %d 条" % (
                    d.strftime("%Y-%m-%d"), len(titles), len(tiyao)), flush=True)
                target = d
                break
        except Exception as e:
            print("[源] %s 失败：%s" % (d.strftime("%Y-%m-%d"), e), flush=True)

    if not titles:
        print("ERROR: 无可用新闻联播数据", flush=True)
        sys.exit(1)

    month = target.strftime("%Y-%m")
    src_lines = ["日期：%s 新闻联播" % target.strftime("%Y年%m月%d日")]
    src_lines.append("")
    src_lines.append("【节目标题】")
    for i, t in enumerate(titles, 1):
        src_lines.append("%d. %s" % (i, t))
    if tiyao:
        src_lines.append("")
        src_lines.append("【节目主要内容】")
        for i, t in enumerate(tiyao, 1):
            src_lines.append("%d. %s" % (i, t))
    source_text = "\n".join(src_lines)

    # 调试输出源文本
    os.makedirs(DAILY_DIR, exist_ok=True)
    with open(os.path.join(DAILY_DIR, "_last_source.txt"), "w", encoding="utf-8") as f:
        f.write(source_text)
    print("  [源] 文本 %d 字符" % len(source_text), flush=True)

    if no_llm:
        print("  [OK] --no-llm 模式，仅抓取。_last_source.txt 已保存", flush=True)
        return

    print("  [LLM] DeepSeek 提取时政卡...", flush=True)
    cards = llm_extract(source_text)
    app_cards = [to_app_card(c, month) for c in cards if c.get("title")]
    if not app_cards:
        print("  WARN: 本次无新增卡，不更新文件", flush=True)
        return

    merged, added = merge_and_save(month, app_cards)
    path = os.path.join(DAILY_DIR, "%s.json" % month)
    print("  [OK] %s -> 新增 %d 张，当月累计 %d 张 -> %s"
          % (target.strftime("%Y-%m-%d"), added, len(merged), path), flush=True)
    for c in app_cards:
        print("   + [%s] %s" % (c["tags"], c["front"]), flush=True)


if __name__ == "__main__":
    main()
