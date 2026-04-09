"""
过滤 accepted_points，只保留指定字段
用于在传递给 judge_chain 时减少数据量
"""

import sys
from pathlib import Path
from typing import List, Dict, Union, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import ClaimPoint


def filter_accepted_points(
    accepted_points: List[Union[ClaimPoint, Dict]],
    fields: Optional[List[str]] = None
) -> List[Dict]:
    """
    过滤 accepted_points，只保留指定字段
    
    Args:
        accepted_points: ClaimPoint 对象列表或字典列表
        fields: 要保留的字段列表，如果为 None 则使用默认字段：
            - point_text
            - supporting_evidence_snippets
            - authority_priority
            - timeliness_priority
            - evidence_published_times
    
    Returns:
        过滤后的字典列表
    """
    if fields is None:
        fields = [
            "point_text",
            "supporting_evidence_snippets",
            "authority_priority",
            "timeliness_priority",
            "evidence_published_times"
        ]
    
    filtered_points = []
    
    for point in accepted_points:
        # 如果是 ClaimPoint 对象，转换为字典
        if isinstance(point, ClaimPoint):
            point_dict = point.model_dump()
        elif isinstance(point, dict):
            point_dict = point
        else:
            # 如果既不是对象也不是字典，尝试获取属性
            point_dict = {}
            for field in fields:
                if hasattr(point, field):
                    value = getattr(point, field)
                    # Processing datetime 等特殊类型
                    if hasattr(value, 'isoformat'):
                        point_dict[field] = value.isoformat()
                    elif isinstance(value, list):
                        point_dict[field] = list(value)
                    else:
                        point_dict[field] = value
        
        # 只保留指定字段
        filtered_point = {}
        for field in fields:
            if field in point_dict:
                filtered_point[field] = point_dict[field]
            else:
                # 如果字段不存在，设置为默认值
                if field == "point_text":
                    filtered_point[field] = ""
                elif field == "supporting_evidence_snippets":
                    filtered_point[field] = []
                elif field == "authority_priority":
                    filtered_point[field] = "Low"
                elif field == "timeliness_priority":
                    filtered_point[field] = 0
                elif field == "evidence_published_times":
                    filtered_point[field] = []
                else:
                    filtered_point[field] = None
        
        filtered_points.append(filtered_point)
    
    return filtered_points


def filter_accepted_points_for_judge(
    accepted_points: List[Union[ClaimPoint, Dict]]
) -> List[Dict]:
    """
    专门用于 judge_chain 的过滤函数
    只保留 judge_chain 需要的字段
    
    Args:
        accepted_points: ClaimPoint 对象列表或字典列表
    
    Returns:
        过滤后的字典列表，包含以下字段：
        - point_text
        - supporting_evidence_snippets
        - authority_priority
        - timeliness_priority
        - evidence_published_times
    """
    return filter_accepted_points(
        accepted_points,
        fields=[
            "point_text",
            "supporting_evidence_snippets",
            "authority_priority",
            "timeliness_priority",
            "evidence_published_times"
        ]
    )


# 使用示例
if __name__ == "__main__":
    # 示例：从 ClaimPoint 对象过滤
    from utils.models import ClaimPoint
    from datetime import datetime
    
    # 创建示例 ClaimPoint
    example_point = ClaimPoint(
        id="point_001",
        point_text="这是一个示例论点",
        sub_claim_id="sub_001",
        sub_claim_text="子claim文本",
        supporting_evidence_ids=["ev_001", "ev_002"],
        supporting_evidence_snippets=["证据片段1", "证据片段2"],
        source_urls=["https://example.com"],
        source_domains=["example.com"],
        credibility="High",
        retrieved_by="pro",
        round_num=1,
        authority_priority="High",
        timeliness_priority=2,
        evidence_published_times=["2024-01-01", "2024-01-02"]
    )
    
    # 过滤
    filtered = filter_accepted_points_for_judge([example_point])
    
    print("原始 ClaimPoint 字段数量:", len(example_point.model_dump()))
    print("过滤后字段数量:", len(filtered[0]))
    print("\n过滤后的数据:")
    import json
    print(json.dumps(filtered, ensure_ascii=False, indent=2))

