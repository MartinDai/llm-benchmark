import argparse
import asyncio
import os
import shutil
import time
from statistics import mean

from openai import AsyncOpenAI


async def query(
        base_url,
        model,
        semaphore,
        api_key,
        max_tokens=500,
        progress_callback=None,
        content_callback=None,
        task_id=None,
):
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async with semaphore:
        try:
            start_time = time.time()
            first_token_time = None
            token_count = 0
            full_content = []

            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "写一篇500字的AI介绍"}],
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    full_content.append(token)
                    token_count += 1

                    if first_token_time is None:
                        first_token_time = time.time()

                    if progress_callback:
                        await progress_callback(task_id, token_count)

                    if content_callback:
                        await content_callback(task_id, token)

            total_time = time.time() - start_time
            token_rate = token_count / total_time if total_time > 0 else 0
            ttft = (first_token_time - start_time) if first_token_time else total_time
            tgl = (total_time - ttft) / (token_count - 1) if token_count > 1 else 0

            return {
                "token_count": token_count,
                "token_rate": token_rate,
                "ttft": ttft,
                "tgl": tgl,
                "total_time": total_time,
                "content": "".join(full_content),
                "success": True,
            }
        except Exception as e:
            error_msg = f"请求失败: {str(e)}"
            return {
                "token_count": 0,
                "content": error_msg,
                "success": False,
            }


class ProgressManager:
    def __init__(self, total_requests, max_tokens, model, concurrency, output_dir):
        self.total_requests = total_requests
        self.max_tokens = max_tokens
        self.model = model
        self.concurrency = concurrency
        self.output_dir = output_dir

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        self.progress = [0] * total_requests
        self.actual_tokens = [max_tokens] * total_requests
        self.completed = [False] * total_requests
        self.contents = [""] * total_requests
        self.start_time = time.time()
        self.completed_count = 0

        self.bar_width = 40
        self.max_display_lines = concurrency + 8
        self.last_displayed_count = 0
        self.last_saved_msg = ""  # 最近一次保存提示

    async def update_progress(self, task_id: int, token_count: int):
        self.progress[task_id] = token_count
        await self.render()

    async def append_content(self, task_id: int, token: str):
        self.contents[task_id] += token

    def mark_completed(self, task_id: int, result: dict):
        actual_tokens = result.get("token_count", 0)
        self.actual_tokens[task_id] = actual_tokens
        self.completed[task_id] = True
        self.completed_count += 1

        final_content = self.contents[task_id] or result.get("content", "")

        filename = os.path.join(self.output_dir, f"request_{task_id+1:03d}.txt")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                if result.get("success", True):
                    f.write(final_content)
                    self.last_saved_msg = f"✓ 已保存: request_{task_id+1:03d}.txt"
                else:
                    f.write(f"[ERROR] {final_content}")
                    self.last_saved_msg = f"✗ 保存失败: request_{task_id+1:03d}.txt"
        except Exception as e:
            self.last_saved_msg = f"✗ 保存错误: request_{task_id+1:03d}.txt ({e})"

        asyncio.create_task(self.render())

    async def render(self):
        elapsed = time.time() - self.start_time
        total_generated = sum(self.progress)

        active = [i for i in range(self.total_requests) if not self.completed[i]]
        recently_done = [i for i in range(self.total_requests) if self.completed[i]]
        recently_done.sort(reverse=True)

        display_ids = active + recently_done
        display_ids = display_ids[:self.max_display_lines]

        lines = []

        header1 = f"模型基准测试: {self.model}  |  并发: {self.concurrency}  |  已完成: {self.completed_count}/{self.total_requests}"
        header2 = f"耗时: {elapsed:6.1f}s  |  生成Tokens: {total_generated:6}  |  实时速率: {self._current_rate():5.1f} t/s"
        lines.append(header1.center(100))
        lines.append(header2.center(100))
        lines.append("=" * 100)

        for tid in display_ids:
            curr = self.progress[tid]
            total = self.actual_tokens[tid]
            percent = 100.0 if self.completed[tid] else min(100.0, curr / total * 100 if total > 0 else 0)

            filled = int(percent / 100 * self.bar_width)
            bar = "█" * filled + "░" * (self.bar_width - filled)

            status_str = "✓ 完成" if self.completed[tid] else "▶ 运行中"

            line = f"请求 {tid+1:3d} | {status_str} | [{bar}] {percent:6.1f}%  ({curr:4}/{total} tokens)"
            lines.append(line)

        if len(display_ids) < self.total_requests:
            remaining = self.total_requests - len(display_ids)
            lines.append(f"... 还有 {remaining} 个请求（等待或已完成，不显示）")

        # 显示最近保存信息
        if self.last_saved_msg:
            lines.append(self.last_saved_msg.center(100))

        lines.append(f"输出目录: {os.path.abspath(self.output_dir)}".center(100))

        new_line_count = len(lines)

        # 擦除多余行
        if new_line_count < self.last_displayed_count:
            erase = self.last_displayed_count - new_line_count
            print(f"\033[{erase}A\033[K" * erase, end="")

        # 回到区域顶部
        if self.last_displayed_count > 0:
            print(f"\033[{self.last_displayed_count}A", end="")

        # 重绘所有行
        for line in lines:
            print(f"{line}\r")

        self.last_displayed_count = new_line_count

    def _current_rate(self):
        elapsed = time.time() - self.start_time
        return sum(self.progress) / elapsed if elapsed > 0 else 0




async def load_test(base_url, model, concurrency=1, requests=1, api_key="xxx", output_dir="outputs"):
    semaphore = asyncio.Semaphore(concurrency)
    max_tokens = 500

    progress_manager = ProgressManager(requests, max_tokens, model, concurrency, output_dir)

    async def progress_callback(task_id, token_count):
        await progress_manager.update_progress(task_id, token_count)

    async def content_callback(task_id, token):
        await progress_manager.append_content(task_id, token)

    results = [None] * requests

    async def run_task(task_id):
        result = await query(
            base_url,
            model,
            semaphore,
            api_key,
            max_tokens,
            progress_callback,
            content_callback,
            task_id,
        )
        progress_manager.mark_completed(task_id, result)
        results[task_id] = result

    await progress_manager.render()  # 初始空界面

    await asyncio.gather(*[run_task(i) for i in range(requests)])

    # 最终统计
    successful = [r for r in results if r is not None and r.get("success", False)]
    if not successful:
        print(f"\n{model}: 所有请求均失败")
        return

    rates = [r["token_rate"] for r in successful]
    ttfts = [r["ttft"] for r in successful]
    tgls = [r["tgl"] for r in successful]
    token_counts = [r["token_count"] for r in successful]

    def percentile(data, p):
        s = sorted(data)
        idx = int(p * len(s))
        return s[idx] if idx < len(s) else s[-1]

    print("\n" + "="*80)
    print(f"基准测试完成: {model}")
    print(f"成功请求: {len(successful)}/{requests}")
    print(f"平均输出长度: {mean(token_counts):.1f} tokens")
    print(f"平均速度: {mean(rates):.2f} tokens/s "
          f"(P50: {percentile(rates,0.5):.2f}, P99: {percentile(rates,0.99):.2f})")
    print(f"平均TTFT: {mean(ttfts):.3f}s")
    print(f"平均每token延迟: {mean(tgls):.4f}s")
    print(f"所有响应已保存至: {os.path.abspath(output_dir)}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 并发基准测试")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--model", type=str, default="qwen2.5:7b")
    parser.add_argument("--api-key", type=str, default="xxx")
    parser.add_argument("--concurrency", type=int, default=2, help="并发数")
    parser.add_argument("--requests", type=int, default=4, help="总请求数")
    parser.add_argument("--output-dir", type=str, default="outputs", help="响应保存目录")

    args = parser.parse_args()
    asyncio.run(load_test(
        args.base_url,
        args.model,
        args.concurrency,
        args.requests,
        args.api_key,
        args.output_dir
    ))