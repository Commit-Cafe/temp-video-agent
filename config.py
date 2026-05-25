import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model_name: str


@dataclass
class Config:
    minimax: LLMConfig
    zhipuai: LLMConfig
    deepseek: LLMConfig
    default_model: str
    whisper_model_size: str
    whisper_device: str
    whisper_compute_type: str
    whisper_api_key: str
    temp_dir: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            minimax=LLMConfig(
                api_key=os.getenv("MINIMAX_API_KEY", ""),
                base_url=os.getenv(
                    "MINIMAX_BASE_URL", "https://api.minimax.chat/v1"
                ),
                model_name="minimax-m2.7",
            ),
            zhipuai=LLMConfig(
                api_key=os.getenv("ZHIPUAI_API_KEY", ""),
                base_url=os.getenv(
                    "ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
                ),
                model_name="glm-5.1",
            ),
            deepseek=LLMConfig(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv(
                    "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
                ),
                model_name="deepseek-v4-pro",
            ),
            default_model=os.getenv("DEFAULT_MODEL", "deepseek-v4-pro"),
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            whisper_api_key=os.getenv("WHISPER_API_KEY", ""),
            temp_dir=os.getenv("TEMP_DIR", "./temp_audio"),
        )

    def get_llm_config(self, model: str) -> LLMConfig:
        if "minimax" in model.lower():
            return self.minimax
        if "glm" in model.lower() or "zhipu" in model.lower():
            return self.zhipuai
        if "deepseek" in model.lower():
            return self.deepseek
        return self.deepseek
