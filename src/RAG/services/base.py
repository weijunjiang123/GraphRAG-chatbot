"""基础服务类"""
import logging
import time
from functools import wraps

import requests

from ..core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_on_failure(max_retries=3, delay=1):
    """重试装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    logger.error(f"请求失败: {str(e)}")
                    if attempt < max_retries - 1:
                        logger.info(f"操作失败，{delay}秒后重试: {str(e)}")
                        time.sleep(delay)
                    else:
                        raise ValueError(f"Ollama 服务连接失败: {str(e)}")
            return None

        return wrapper

    return decorator


class BaseService:
    """服务基类"""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.timeout = settings.REQUEST_TIMEOUT

    def _check_ollama_status(self):
        """检查 Ollama 服务状态"""
        try:
            response = requests.get(
                f"{self.base_url}/api/health",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return True
            logger.error(f"Ollama 服务状态检查失败: {response.status_code}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"无法连接到 Ollama 服务: {str(e)}")
            return False

    @retry_on_failure(max_retries=settings.MAX_RETRIES, delay=settings.RETRY_DELAY)
    def _make_request(self, method, endpoint, **kwargs):
        """发送请求到 Ollama 服务"""
        if not self._check_ollama_status():
            raise ValueError("Ollama 服务未启动或无法访问")

        url = f"{self.base_url}{endpoint}"
        response = requests.request(
            method,
            url,
            timeout=self.timeout,
            **kwargs
        )

        if response.status_code != 200:
            raise requests.exceptions.RequestException(
                f"请求失败 (状态码: {response.status_code})"
            )

        return response.json()
