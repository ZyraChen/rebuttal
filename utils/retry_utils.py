"""
重试工具 - 用于Processing API 调用failed

支持：
- 指数退避 (Exponential Backoff)
- 可配置的重试次数
- 指定需要重试的异常类型
"""

import time
import asyncio
from functools import wraps
from typing import Callable, Tuple, Type, Any
import logging

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] = None
):
    """
    重试装饰器，支持指数退避

    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        backoff_factor: 退避因子（每次重试延迟乘以此因子）
        exceptions: 需要重试的异常类型元组
        on_retry: 重试时的回调函数 (exception, retry_count) -> None

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def call_api():
            # API 调用
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # 最后一次尝试failed
                        logger.error(f"{func.__name__} failed，已达到最大重试次数 {max_retries}")
                        raise

                    # 调用重试回调
                    if on_retry:
                        on_retry(e, attempt + 1)
                    else:
                        logger.warning(
                            f"{func.__name__} failed (尝试 {attempt + 1}/{max_retries + 1}): {e}. "
                            f"将在 {delay:.1f} 秒后重试..."
                        )

                    # 等待后重试
                    time.sleep(delay)
                    delay *= backoff_factor

            # 不应该到达这里
            raise last_exception

        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] = None
):
    """
    异步版本的重试装饰器

    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        backoff_factor: 退避因子
        exceptions: 需要重试的异常类型元组
        on_retry: 重试时的回调函数

    Example:
        @async_retry_with_backoff(max_retries=3)
        async def call_api_async():
            # 异步 API 调用
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed，已达到最大重试次数 {max_retries}")
                        raise

                    if on_retry:
                        on_retry(e, attempt + 1)
                    else:
                        logger.warning(
                            f"{func.__name__} failed (尝试 {attempt + 1}/{max_retries + 1}): {e}. "
                            f"将在 {delay:.1f} 秒后重试..."
                        )

                    await asyncio.sleep(delay)
                    delay *= backoff_factor

            raise last_exception

        return wrapper
    return decorator


def call_with_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs
) -> Any:
    """
    直接调用函数并重试（不使用装饰器）

    Args:
        func: 要调用的函数
        *args: 函数参数
        max_retries: 最大重试次数
        initial_delay: 初始延迟
        backoff_factor: 退避因子
        exceptions: 需要重试的异常类型
        **kwargs: 函数关键字参数

    Returns:
        函数返回值

    Example:
        result = call_with_retry(api_call, arg1, arg2, max_retries=5)
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"{func.__name__} failed，已达到最大重试次数 {max_retries}")
                raise

            logger.warning(
                f"{func.__name__} failed (尝试 {attempt + 1}/{max_retries + 1}): {e}. "
                f"将在 {delay:.1f} 秒后重试..."
            )

            time.sleep(delay)
            delay *= backoff_factor

    raise last_exception


async def async_call_with_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs
) -> Any:
    """
    异步版本的 call_with_retry
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"{func.__name__} failed，已达到最大重试次数 {max_retries}")
                raise

            logger.warning(
                f"{func.__name__} failed (尝试 {attempt + 1}/{max_retries + 1}): {e}. "
                f"将在 {delay:.1f} 秒后重试..."
            )

            await asyncio.sleep(delay)
            delay *= backoff_factor

    raise last_exception


def call_with_retry_until_success(
    func: Callable,
    *args,
    validate_result = None,
    max_retries = None,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs
) -> Any:
    """
    持续重试直到successful（支持返回值验证）
    
    Args:
        func: 要调用的函数
        *args: 函数参数
        validate_result: 验证返回值的函数，返回 True 表示successful，False 表示需要重试
        max_retries: 最大重试次数（None 表示无限重试）
        initial_delay: 初始延迟
        backoff_factor: 退避因子
        max_delay: 最大延迟时间
        exceptions: 需要重试的异常类型
        **kwargs: 函数关键字参数
        
    Returns:
        函数返回值
        
    Example:
        # 重试直到返回非空列表
        result = call_with_retry_until_success(
            api_call, 
            arg1, 
            validate_result=lambda x: x and len(x) > 0,
            max_retries=None  # 无限重试
        )
    """
    delay = initial_delay
    last_exception = None
    attempt = 0
    
    while max_retries is None or attempt <= max_retries:
        try:
            result = func(*args, **kwargs)
            
            # 如果没有验证函数，直接返回
            if validate_result is None:
                return result
            
            # 验证返回值
            if validate_result(result):
                return result
            else:
                # 返回值不符合要求，视为failed
                logger.warning(
                    f"{func.__name__} 返回值验证failed (尝试 {attempt + 1})"
                    f"{'' if max_retries is None else f'/{max_retries + 1}'}. "
                    f"将在 {delay:.1f} 秒后重试..."
                )
                
        except exceptions as e:
            last_exception = e
            
            logger.warning(
                f"{func.__name__} failed (尝试 {attempt + 1}"
                f"{'' if max_retries is None else f'/{max_retries + 1}'}): {e}. "
                f"将在 {delay:.1f} 秒后重试..."
            )
        
        # 检查是否达到最大重试次数
        if max_retries is not None and attempt == max_retries:
            if last_exception:
                logger.error(f"{func.__name__} failed，已达到最大重试次数 {max_retries}")
                raise last_exception
            else:
                logger.error(f"{func.__name__} 返回值验证failed，已达到最大重试次数 {max_retries}")
                return result  # 返回最后一次的结果
        
        # 等待后重试
        time.sleep(delay)
        delay = min(delay * backoff_factor, max_delay)  # 限制最大延迟
        attempt += 1
    
    # 不应该到达这里
    if last_exception:
        raise last_exception
    return result