#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文日常对话数据集生成器
批量调用本地 LLM API 生成结构化对话数据
"""

import requests
import json
import time
import random
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


def load_env(env_path: Path = None) -> Dict[str, str]:
    """从 .env 文件加载环境变量"""
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"
    
    env_vars = {}
    
    if not env_path.exists():
        # 创建示例 .env 文件
        example_content = """# LLM API 配置
API_URL=http://localhost:8000/v1/chat/completions
API_KEY=EMPTY
MODEL=your-model

# 生成配置
TARGET_BATCHES=100
BATCH_SIZE=100
"""
        example_path = env_path.parent / ".env.example"
        example_path.write_text(example_content, encoding='utf-8')
        print(f"警告: 未找到 .env 文件，已创建示例 {example_path}")
        print("请复制 .env.example 为 .env 并填写实际配置")
        return env_vars
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    
    return env_vars


@dataclass
class Config:
    """配置类"""
    api_url: str = "http://localhost:8000/v1/chat/completions"
    api_key: str = "EMPTY"
    model: str = "your-model"
    output_file: str = "dataset.jsonl"
    target_batches: int = 100
    batch_size: int = 100
    temperature: float = 0.9
    max_tokens: int = 90000
    timeout: int = 300
    retry_max: int = 3
    retry_delay: float = 10.0
    sleep_min: float = 2.0
    sleep_max: float = 5.0
    
    # 对话主题池
    topics: tuple = (
        "日常闲聊", "美食推荐", "旅行计划", "购物建议", 
        "健康养生", "家庭琐事", "宠物饲养", "运动健身",
        "电影音乐", "读书分享", "职场吐槽", "情感咨询",
        "兴趣爱好", "节日庆祝", "天气交通"
    )
    
    @classmethod
    def from_env(cls, env_vars: Dict[str, str]) -> "Config":
        """从环境变量创建配置"""
        return cls(
            api_url=env_vars.get("API_URL", cls.api_url),
            api_key=env_vars.get("API_KEY", cls.api_key),
            model=env_vars.get("MODEL", cls.model),
            target_batches=int(env_vars.get("TARGET_BATCHES", cls.target_batches)),
            batch_size=int(env_vars.get("BATCH_SIZE", cls.batch_size)),
        )


class DatasetGenerator:
    """数据集生成器"""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.session = requests.Session()
        self.success_count = 0
        self.fail_count = 0
        self.total_conversations = 0
        
        # 输出路径改为 data/ 目录
        output_dir = Path(__file__).parent
        self.output_path = output_dir / self.cfg.output_file
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 已存在的对话哈希
        self.seen_hashes = set()
        if self.output_path.exists():
            self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """加载已存在的数据"""
        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.seen_hashes.add(hash(line.strip()))
            print(f"已加载现有数据: {len(self.seen_hashes)} 条")
        except Exception as e:
            print(f"加载现有数据警告: {e}")
    
    def _generate_system_prompt(self) -> str:
        """动态生成系统提示"""
        topic = random.choice(self.cfg.topics)
        
        return f"""你是一个数据生成器，用于生成中文日常对话数据。

当前主题倾向：{topic}

生成规则：
1. 只生成日常生活聊天场景，禁止编程、科技、专业知识
2. 每条对话 2-10 轮，角色严格交替 user/assistant
3. 内容自然口语化，符合中国人日常表达习惯
4. 包含适当的语气词、口语化表达
5. 场景多样化：微信聊天、面对面交流、电话沟通等
6. 每条对话独立完整，有自然的开头和结尾
7. 禁止重复或相似的对话内容

输出格式要求：
{{
  "conversations": [
    {{
      "conversation": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}}
      ]
    }}
  ]
}}

严格输出 JSON，不要包含 markdown 代码块或其他说明文字。一次生成 {self.cfg.batch_size} 条对话。"""

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求，带重试机制"""
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(self.cfg.retry_max):
            try:
                response = self.session.post(
                    self.cfg.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                print(f"  超时 (尝试 {attempt + 1}/{self.cfg.retry_max})")
            except requests.exceptions.HTTPError as e:
                print(f"  HTTP错误 {e.response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"  连接错误 (尝试 {attempt + 1}/{self.cfg.retry_max})")
            except Exception as e:
                print(f"  请求异常: {e}")
            
            if attempt < self.cfg.retry_max - 1:
                sleep_time = self.cfg.retry_delay * (2 ** attempt)
                print(f"  {sleep_time:.1f}秒后重试...")
                time.sleep(sleep_time)
        
        raise Exception(f"{self.cfg.retry_max} 次重试后仍失败")
    
    def _parse_response(self, content: str) -> List[Dict[str, Any]]:
        """解析并验证响应内容"""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        
        if "conversations" not in data:
            raise ValueError("响应缺少 'conversations' 字段")
        
        conversations = data["conversations"]
        valid_convs = []
        
        for conv in conversations:
            if "conversation" not in conv:
                continue
            
            messages = conv["conversation"]
            valid = True
            
            for i, msg in enumerate(messages):
                expected_role = "user" if i % 2 == 0 else "assistant"
                if msg.get("role") != expected_role:
                    valid = False
                    break
            
            if valid and len(messages) >= 2:
                conv_str = json.dumps(conv, ensure_ascii=False, sort_keys=True)
                conv_hash = hash(conv_str)
                
                if conv_hash not in self.seen_hashes:
                    self.seen_hashes.add(conv_hash)
                    valid_convs.append(conv)
        
        return valid_convs
    
    def generate_batch(self) -> int:
        """生成一批数据"""
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": self._generate_system_prompt()},
                {"role": "user", "content": f"生成{self.cfg.batch_size}条日常中文对话"}
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        data = self._make_request(payload)
        content = data["choices"][0]["message"]["content"]
        conversations = self._parse_response(content)
        
        with open(self.output_path, "a", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        
        return len(conversations)
    
    def run(self):
        """主运行循环"""
        print(f"{'='*50}")
        print(f"数据集生成开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API: {self.cfg.api_url}")
        print(f"模型: {self.cfg.model}")
        print(f"目标: {self.cfg.target_batches} 批 × ~{self.cfg.batch_size} 条")
        print(f"输出: {self.output_path}")
        print(f"{'='*50}\n")
        
        start_time = time.time()
        
        for i in range(self.cfg.target_batches):
            batch_start = time.time()
            
            try:
                count = self.generate_batch()
                self.success_count += 1
                self.total_conversations += count
                
                batch_time = time.time() - batch_start
                print(f"[{i+1:3d}/{self.cfg.target_batches}] ✓ "
                      f"生成 {count:3d} 条 | "
                      f"耗时 {batch_time:.1f}s | "
                      f"累计 {self.total_conversations} 条")
                
                if i < self.cfg.target_batches - 1:
                    sleep_time = random.uniform(self.cfg.sleep_min, self.cfg.sleep_max)
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.fail_count += 1
                print(f"[{i+1:3d}/{self.cfg.target_batches}] ✗ 错误: {e}")
                time.sleep(self.cfg.retry_delay)
        
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print("生成完成!")
        print(f"总耗时: {total_time/60:.1f} 分钟")
        print(f"成功批次: {self.success_count}/{self.cfg.target_batches}")
        print(f"失败批次: {self.fail_count}")
        print(f"总对话数: {self.total_conversations}")
        print(f"平均速度: {self.total_conversations/total_time:.1f} 条/秒")
        print(f"{'='*50}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="中文对话数据集生成器")
    parser.add_argument("--batches", "-b", type=int, help="覆盖批次数量")
    parser.add_argument("--output", "-o", type=str, help="覆盖输出文件名")
    args = parser.parse_args()
    
    # 加载 .env
    env_vars = load_env()
    
    # 创建配置
    config = Config.from_env(env_vars)
    
    # 命令行参数覆盖
    if args.batches:
        config.target_batches = args.batches
    if args.output:
        config.output_file = args.output
    
    generator = DatasetGenerator(config)
    
    try:
        generator.run()
    except KeyboardInterrupt:
        print("\n\n用户中断，已保存进度")
        sys.exit(0)


if __name__ == "__main__":
    main()