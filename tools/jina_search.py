"""
Jina Search API Wrapper
Support for batch parallel search
"""
import asyncio
import aiohttp
from typing import List, Dict
import time


class JinaSearch:
    """Jina Reader Search API Wrapper"""

    def __init__(self, api_key: str, max_results_per_query: int = 20):
        self.api_key = api_key
        self.max_results = max_results_per_query
        self.base_url = "https://s.jina.ai/?q="

    def _optimize_query(self, query: str) -> str:
        """
        Optimize query to improve search success rate

        Optimization strategies:
        1. Limit length (max 100 characters)
        2. Remove question marks and complex sentence structures
        3. Keep keywords
        """
        # Remove meaningless prefixes like "年", "在" at the beginning
        query = query.lstrip('年在')

        # Remove question marks and exclamation marks
        query = query.replace('？', '').replace('?', '')
        query = query.replace('！', '').replace('!', '')

        # Remove question words like "是否", "有没有"
        query = query.replace('是否', '').replace('有没有', '')
        query = query.replace('是不是', '').replace('到底', '')

        # If too long, try to extract key parts (keep first 80 characters)
        if len(query) > 100:
            # Try to truncate at punctuation marks
            for sep in ['，', ',', '。', '.', '中', '的']:
                if sep in query[:100]:
                    parts = query[:100].split(sep)
                    if len(parts) > 1 and len(parts[0]) > 20:
                        query = parts[0]
                        break
            else:
                query = query[:80]

        return query.strip()

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Synchronous search method (compatibility interface)

        Parameters:
        - query: Search query
        - top_k: Number of results to return

        Returns: [{"title": ..., "content": ..., "url": ...}, ...]
        """
        # Optimize query
        optimized_query = self._optimize_query(query)
        if optimized_query != query:
            print(f"[Jina] Query optimized: {query[:50]}... -> {optimized_query[:50]}...")

        self.max_results = top_k
        return asyncio.run(self.search_single(optimized_query))

    async def search_single(self, query: str, task_context: str = None) -> List[Dict]:
        """
        Single search

        Parameters:
        - query: Search query (can be a question or task description)
        - task_context: Task context, e.g. "As a fact-checking expert, please help me..."

        Returns: [{"title": ..., "content": ..., "url": ...}, ...]
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Retain-Images": "no-content"
        }

        # If task context is provided, combine query
        if task_context:
            full_query = f"{task_context} {query}"
        else:
            full_query = query

        url = f"{self.base_url}{full_query}"

        print(f"[Jina] Query URL: {url[:100]}...")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    print(f"[Jina] Status code: {response.status}")

                    if response.status == 200:
                        text = await response.text()
                        print(f"[Jina] Response length: {len(text)} characters")

                        # Parse Jina's Markdown format response
                        results = self._parse_jina_response(text)
                        print(f"[Jina] Parsed {len(results)} results")

                        return results[:self.max_results]
                    elif response.status == 401:
                        print(f"[Jina] API Key invalid or not set!")
                        print(f"[Jina] Current Key: {self.api_key[:10]}...")
                        return []
                    elif response.status == 429:
                        print(f"[Jina] API call limit exceeded!")
                        return []
                    else:
                        error_text = await response.text()
                        print(f"[Jina] Search failed: {response.status}")
                        print(f"[Jina] Error message: {error_text[:200]}")
                        return []
        except asyncio.TimeoutError:
            print(f"[Jina] Request timeout (30 seconds)")
            return []
        except Exception as e:
            print(f"[Jina] Search exception: {query[:50]} - {e}")
            return []

    def _parse_jina_response(self, markdown_text: str) -> List[Dict]:
        """
        Parse Jina response format
        New format:
        [1] Title: xxx
        [1] URL Source: xxx
        [1] Description: xxx
        [1] Published Time: xxx (optional)
        Content...
        """
        results = []

        # Check format type
        if "[1] Title:" in markdown_text:
            # New format: [n] Title: ...
            print(f"[Jina Parse] Detected new format [n] Title:")

            # 按[数字]分割
            import re
            from datetime import datetime
            blocks = re.split(r'\[\d+\] Title:', markdown_text)

            for block in blocks[1:]:  # Skip the first empty block
                lines = block.strip().split('\n')

                title = lines[0].strip() if lines else ""
                url = ""
                description = ""
                content_lines = []
                published_time = None
                metadata_end_index = 0

                # Extract URL, description and published time
                for i, line in enumerate(lines):
                    if line.startswith('[') and '] URL Source:' in line:
                        url = line.split('] URL Source:')[1].strip()
                        metadata_end_index = max(metadata_end_index, i)
                    elif line.startswith('[') and '] Description:' in line:
                        description = line.split('] Description:')[1].strip()
                        metadata_end_index = max(metadata_end_index, i)
                    # Support multiple published time field names
                    elif line.startswith('[') and (
                        '] Published Time:' in line or
                        '] Publish Time:' in line or
                        '] Publication Date:' in line or
                        '] Date:' in line or
                        '] Time:' in line
                    ):
                        # Extract time string
                        for sep in ['] Published Time:', '] Publish Time:', '] Publication Date:', '] Date:', '] Time:']:
                            if sep in line:
                                time_str = line.split(sep)[1].strip()
                                if time_str:
                                    published_time = time_str
                                metadata_end_index = max(metadata_end_index, i)
                                break

                # Extract actual content after metadata (as fallback)
                if metadata_end_index > 0 and len(lines) > metadata_end_index + 1:
                    # Skip empty lines, extract actual content
                    for i in range(metadata_end_index + 1, len(lines)):
                        line = lines[i].strip()
                        if line:  # Skip empty lines
                            content_lines.append(line)

                # Combine content: prefer description, if empty use full content
                if description:
                    content = description
                elif content_lines:
                    content = ' '.join(content_lines)[:1000]
                else:
                    content = ""

                # If no published time, try to extract date from multiple sources (by priority)
                if not published_time:
                    # 1. Extract from title first
                    published_time = self._extract_date_from_text(title)

                    # 2. 再从description提取
                    if not published_time:
                        published_time = self._extract_date_from_text(description)

                    # 3. 从URL提取
                    if not published_time:
                        published_time = self._extract_date_from_text(url)

                    # 4. 从内容前500字符提取
                    if not published_time:
                        published_time = self._extract_date_from_text(content[:500])

                if title and url:
                    results.append({
                        "title": title,
                        "url": url,
                        "content": content,
                        "published_time": published_time,
                        "retrieved_time": datetime.now().isoformat()
                    })

        elif "---" in markdown_text:
            # Old format: --- Title: ... ---
            print(f"[Jina Parse] Detected old format ---")
            from datetime import datetime
            blocks = markdown_text.split("---\n")

            for i in range(1, len(blocks), 2):
                if i + 1 >= len(blocks):
                    break

                metadata_block = blocks[i]
                content_block = blocks[i + 1] if i + 1 < len(blocks) else ""

                title = ""
                url = ""
                published_time = None
                for line in metadata_block.split("\n"):
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.startswith("URL:"):
                        url = line.replace("URL:", "").strip()
                    # 支持多种发布时间字段名称
                    elif any(line.startswith(prefix) for prefix in [
                        "Published Time:", "Publish Time:", "Publication Date:", "Date:", "Time:"
                    ]):
                        for prefix in ["Published Time:", "Publish Time:", "Publication Date:", "Date:", "Time:"]:
                            if line.startswith(prefix):
                                time_str = line.replace(prefix, "").strip()
                                if time_str:
                                    published_time = time_str
                                break

                content = content_block.strip()[:1000]

                # 如果没有发布时间，尝试从多个来源提取日期（按优先级）
                if not published_time:
                    # 1. 先从标题提取
                    published_time = self._extract_date_from_text(title)

                    # 2. 从URL提取
                    if not published_time:
                        published_time = self._extract_date_from_text(url)

                    # 3. 从内容前500字符提取
                    if not published_time:
                        published_time = self._extract_date_from_text(content[:500])

                if title and url and content:
                    results.append({
                        "title": title,
                        "url": url,
                        "content": content,
                        "published_time": published_time,
                        "retrieved_time": datetime.now().isoformat()
                    })
        else:
            print(f"[Jina Parse] Unrecognized format")
            print(f"[Jina Parse] First 300 characters: {markdown_text[:300]}")
            return []

        print(f"[Jina Parse] Successfully parsed {len(results)} results")
        print(results)
        return results

    async def search_batch(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """
        Batch parallel search
        Returns: {query: [results...], ...}
        """
        tasks = [self.search_single(q) for q in queries]
        results = await asyncio.gather(*tasks)

        return {
            query: result
            for query, result in zip(queries, results)
        }

    def search_batch_sync(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """Synchronous version of batch search"""
        return asyncio.run(self.search_batch(queries))

    def _extract_date_from_text(self, text: str) -> str:
        """
        Extract date information from text
        Support multiple date formats and relative time
        """
        if not text or not isinstance(text, str):
            return None

        import re
        from datetime import datetime, timedelta

        # 1. Try to extract relative time (e.g. "2 days ago", "3 hours ago")
        relative_time = self._parse_relative_time(text)
        if relative_time:
            return relative_time

        # 2. Regular expressions for common date formats (sorted by priority)
        date_patterns = [
            # ISO 8601 格式
            (r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', 'iso8601'),
            # 完整日期时间：2024-01-07 12:30:45
            (r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', 'datetime'),
            # YYYY-MM-DD 或 YYYY/MM/DD 或 YYYY.MM.DD
            (r'(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})', 'ymd'),
            # DD-MM-YYYY 或 MM-DD-YYYY 或 DD.MM.YYYY
            (r'(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})', 'dmy'),
            # 中文日期格式：2024年1月7日
            (r'(\d{4}年\d{1,2}月\d{1,2}日)', 'chinese'),
            # 带文字前缀的中文日期：发布于2024年1月7日、更新时间：2024年1月7日
            (r'(?:发布于|更新时间[：:]|时间[：:])\s*(\d{4}年\d{1,2}月\d{1,2}日)', 'chinese_prefix'),
            # 英文月份格式：Jan 7, 2024 或 January 7, 2024
            (r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})', 'month_name'),
            # 年月日（紧凑）：20240107
            (r'(20\d{6})', 'compact'),
            # 仅年月：2024-01 或 2024/01
            (r'(\d{4}[-/]\d{1,2})', 'ym'),
            # 仅年份 (2000-2099)
            (r'(20\d{2})', 'year'),
        ]

        for pattern, format_type in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                # 标准化日期格式
                try:
                    if format_type in ['chinese', 'chinese_prefix']:
                        # 转换中文日期格式：2024年1月7日 -> 2024-01-07
                        date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
                    elif format_type == 'compact':
                        # 转换紧凑格式：20240107 -> 2024-01-07
                        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    elif format_type == 'month_name':
                        # 英文月份格式保持原样
                        pass
                    return date_str
                except:
                    return date_str

        return None

    def _parse_relative_time(self, text: str) -> str:
        """
        Parse relative time (e.g. "2 days ago", "3 hours ago") to specific date
        """
        if not text or not isinstance(text, str):
            return None

        import re
        from datetime import datetime, timedelta

        # 中文相对时间
        patterns = [
            (r'(\d+)\s*秒[钟前]', 'seconds'),
            (r'(\d+)\s*分钟前', 'minutes'),
            (r'(\d+)\s*小时前', 'hours'),
            (r'(\d+)\s*天前', 'days'),
            (r'(\d+)\s*周前', 'weeks'),
            (r'(\d+)\s*月前', 'months'),
            (r'(\d+)\s*年前', 'years'),
            (r'昨天', 'yesterday'),
            (r'前天', 'day_before_yesterday'),
        ]

        # 英文相对时间
        patterns_en = [
            (r'(\d+)\s*seconds?\s+ago', 'seconds'),
            (r'(\d+)\s*minutes?\s+ago', 'minutes'),
            (r'(\d+)\s*hours?\s+ago', 'hours'),
            (r'(\d+)\s*days?\s+ago', 'days'),
            (r'(\d+)\s*weeks?\s+ago', 'weeks'),
            (r'(\d+)\s*months?\s+ago', 'months'),
            (r'(\d+)\s*years?\s+ago', 'years'),
            (r'yesterday', 'yesterday'),
        ]

        all_patterns = patterns + patterns_en

        for pattern, unit in all_patterns:
            match = re.search(pattern, text.lower())
            if match:
                now = datetime.now()

                if unit == 'yesterday':
                    target_date = now - timedelta(days=1)
                elif unit == 'day_before_yesterday':
                    target_date = now - timedelta(days=2)
                else:
                    amount = int(match.group(1))
                    if unit == 'seconds':
                        target_date = now - timedelta(seconds=amount)
                    elif unit == 'minutes':
                        target_date = now - timedelta(minutes=amount)
                    elif unit == 'hours':
                        target_date = now - timedelta(hours=amount)
                    elif unit == 'days':
                        target_date = now - timedelta(days=amount)
                    elif unit == 'weeks':
                        target_date = now - timedelta(weeks=amount)
                    elif unit == 'months':
                        # 近似：1个月 = 30天
                        target_date = now - timedelta(days=amount * 30)
                    elif unit == 'years':
                        # 近似：1年 = 365天
                        target_date = now - timedelta(days=amount * 365)
                    else:
                        continue

                return target_date.strftime('%Y-%m-%d')

        return None

