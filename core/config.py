"""
MedGemma Agent Framework - Configuration Management

This module handles all configuration settings for the MedGemma agent framework:
- Model configuration (HuggingFace model ID, device, quantization)
- API keys for external services (PubMed, OpenFDA, etc.)
- Logging settings
- Tool enable/disable flags
- Resource limits and timeouts

Usage:
    from core.config import get_config, MedGemmaConfig

    # Get default configuration
    config = get_config()

    # Access settings
    model_id = config.model.model_id
    device = config.model.device

    # Override with environment variables
    # MEDGEMMA_MODEL_ID=google/medgemma-4b-it
    # MEDGEMMA_DEVICE=cuda
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceType(str, Enum):
    """Supported device types for model inference."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class QuantizationType(str, Enum):
    """Supported quantization types."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelConfig(BaseModel):
    """MedGemma model configuration."""

    model_id: str = Field(
        default="google/medgemma-4b-it",
        description="HuggingFace model ID for MedGemma"
    )
    skip_model_loading: bool = Field(
        default=True,
        description="Skip loading the LLM (tools-only mode). Set to False on Linux+GPU with 16GB+ VRAM"
    )
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Device to run model on"
    )
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE,
        description="Quantization type for model (int4=~4GB, int8=~8GB, none=~16GB). Note: int4/int8 require Linux+NVIDIA"
    )
    max_length: int = Field(
        default=4096,
        ge=128,
        le=32768,
        description="Maximum sequence length"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )
    top_k: int = Field(
        default=50,
        ge=1,
        description="Top-k sampling parameter"
    )
    torch_dtype: str = Field(
        default="auto",
        description="PyTorch dtype for model weights"
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code from HuggingFace"
    )
    use_flash_attention: Optional[bool] = Field(
        default=None,
        description="Use Flash Attention 2. None=auto-detect, True=force on, False=force off"
    )

    def get_device(self) -> str:
        """Resolve AUTO device to actual device."""
        if self.device == DeviceType.AUTO:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return self.device.value

    @staticmethod
    def is_flash_attention_available() -> bool:
        """Check if FlashAttention2 is available on this system."""
        import sys
        if sys.platform != "linux":
            return False
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            # Check for Ampere+ GPU (compute capability >= 8.0)
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:
                return False
            # Check if flash_attn is installed
            import importlib.util
            return importlib.util.find_spec("flash_attn") is not None
        except Exception:
            return False

    def should_use_flash_attention(self) -> bool:
        """Determine if flash attention should be used."""
        if self.use_flash_attention is None:
            # Auto-detect: use if available
            return self.is_flash_attention_available()
        # Explicit setting: respect user's choice
        return self.use_flash_attention


class APIKeysConfig(BaseModel):
    """External API keys configuration."""

    huggingface_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token for gated models"
    )
    pubmed_api_key: Optional[str] = Field(
        default=None,
        description="NCBI PubMed API key"
    )
    openfda_api_key: Optional[str] = Field(
        default=None,
        description="OpenFDA API key for drug data"
    )
    umls_api_key: Optional[str] = Field(
        default=None,
        description="UMLS API key for medical terminology"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files"
    )
    log_to_file: bool = Field(
        default=True,
        description="Write logs to file"
    )
    log_to_console: bool = Field(
        default=True,
        description="Write logs to console"
    )
    audit_log_enabled: bool = Field(
        default=True,
        description="Enable audit logging for compliance"
    )
    max_log_size_mb: int = Field(
        default=100,
        ge=1,
        description="Maximum log file size in MB"
    )
    log_retention_days: int = Field(
        default=30,
        ge=1,
        description="Days to retain log files"
    )


class ToolConfig(BaseModel):
    """Individual tool enable/disable configuration."""

    # Imaging tools
    imaging_enabled: bool = Field(default=True, description="Enable imaging analysis tools")
    dicom_handler_enabled: bool = Field(default=True)
    xray_analyzer_enabled: bool = Field(default=True)
    ct_analyzer_enabled: bool = Field(default=True)
    mri_analyzer_enabled: bool = Field(default=True)
    histopath_analyzer_enabled: bool = Field(default=True)
    fundus_analyzer_enabled: bool = Field(default=True)
    dermoscopy_analyzer_enabled: bool = Field(default=True)
    ultrasound_analyzer_enabled: bool = Field(default=True)

    # Document processing tools
    documents_enabled: bool = Field(default=True, description="Enable document processing tools")
    lab_report_parser_enabled: bool = Field(default=True)
    radiology_report_parser_enabled: bool = Field(default=True)
    pathology_report_parser_enabled: bool = Field(default=True)
    discharge_summary_parser_enabled: bool = Field(default=True)
    clinical_notes_parser_enabled: bool = Field(default=True)
    prescription_parser_enabled: bool = Field(default=True)
    insurance_claims_parser_enabled: bool = Field(default=True)

    # Knowledge tools
    knowledge_enabled: bool = Field(default=True, description="Enable medical knowledge tools")
    medical_calculator_enabled: bool = Field(default=True)
    icd_cpt_lookup_enabled: bool = Field(default=True)
    terminology_explainer_enabled: bool = Field(default=True)
    drug_interaction_enabled: bool = Field(default=True)
    guidelines_rag_enabled: bool = Field(default=True)
    pubmed_search_enabled: bool = Field(default=True)

    # Clinical decision support tools
    clinical_enabled: bool = Field(default=True, description="Enable clinical decision support tools")
    triage_classifier_enabled: bool = Field(default=True)
    risk_assessment_enabled: bool = Field(default=True)
    differential_diagnosis_enabled: bool = Field(default=True)
    treatment_recommender_enabled: bool = Field(default=True)

    # Integration tools
    integration_enabled: bool = Field(default=True, description="Enable healthcare integration tools")
    fhir_adapter_enabled: bool = Field(default=True)
    hl7_parser_enabled: bool = Field(default=True)
    patient_timeline_enabled: bool = Field(default=True)

    # Safety tools
    safety_enabled: bool = Field(default=True, description="Enable safety and compliance tools")
    audit_logger_enabled: bool = Field(default=True)
    safety_checker_enabled: bool = Field(default=True)
    hallucination_detector_enabled: bool = Field(default=True)
    uncertainty_quantifier_enabled: bool = Field(default=True)

    # Utility tools
    utilities_enabled: bool = Field(default=True, description="Enable utility tools")
    entity_extractor_enabled: bool = Field(default=True)
    medical_summarizer_enabled: bool = Field(default=True)
    patient_translator_enabled: bool = Field(default=True)
    report_generator_enabled: bool = Field(default=True)


class ResourceConfig(BaseModel):
    """Resource limits and timeouts configuration."""

    max_image_size_mb: int = Field(
        default=100,
        ge=1,
        description="Maximum image file size in MB"
    )
    max_document_size_mb: int = Field(
        default=50,
        ge=1,
        description="Maximum document file size in MB"
    )
    tool_timeout_seconds: int = Field(
        default=300,
        ge=10,
        description="Timeout for tool execution in seconds"
    )
    max_concurrent_tools: int = Field(
        default=4,
        ge=1,
        description="Maximum concurrent tool executions"
    )
    memory_limit_gb: float = Field(
        default=16.0,
        ge=1.0,
        description="Memory limit for processing in GB"
    )
    cache_dir: Path = Field(
        default=Path(".cache/medgemma"),
        description="Directory for caching"
    )
    cache_size_gb: float = Field(
        default=10.0,
        ge=0.0,
        description="Maximum cache size in GB"
    )


class SafetyConfig(BaseModel):
    """Safety and compliance configuration."""

    require_disclaimer: bool = Field(
        default=True,
        description="Require medical disclaimer in outputs"
    )
    block_dangerous_outputs: bool = Field(
        default=True,
        description="Block potentially dangerous medical advice"
    )
    require_confidence_scores: bool = Field(
        default=True,
        description="Require confidence scores in outputs"
    )
    min_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for outputs"
    )
    enable_hallucination_check: bool = Field(
        default=True,
        description="Enable hallucination detection"
    )
    hipaa_compliant_logging: bool = Field(
        default=True,
        description="Enable HIPAA-compliant logging (no PHI in logs)"
    )
    redact_phi_in_logs: bool = Field(
        default=True,
        description="Automatically redact PHI in logs"
    )


class MedGemmaConfig(BaseSettings):
    """
    Main configuration class for MedGemma Agent Framework.

    Configuration is loaded from:
    1. Default values
    2. Environment variables (prefixed with MEDGEMMA_)
    3. .env file if present
    4. Explicit overrides
    """

    model_config = SettingsConfigDict(
        env_prefix="MEDGEMMA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    # Top-level settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled."""
        # Map tool names to config attributes
        attr_name = f"{tool_name.replace('-', '_')}_enabled"

        # Check category-level enable first
        category_map = {
            "dicom_handler": "imaging",
            "xray_analyzer": "imaging",
            "ct_analyzer": "imaging",
            "mri_analyzer": "imaging",
            "histopath_analyzer": "imaging",
            "fundus_analyzer": "imaging",
            "dermoscopy_analyzer": "imaging",
            "ultrasound_analyzer": "imaging",
            "lab_report_parser": "documents",
            "radiology_report_parser": "documents",
            "pathology_report_parser": "documents",
            "discharge_summary_parser": "documents",
            "clinical_notes_parser": "documents",
            "prescription_parser": "documents",
            "insurance_claims_parser": "documents",
            "medical_calculator": "knowledge",
            "icd_cpt_lookup": "knowledge",
            "terminology_explainer": "knowledge",
            "drug_interaction": "knowledge",
            "guidelines_rag": "knowledge",
            "pubmed_search": "knowledge",
            "triage_classifier": "clinical",
            "risk_assessment": "clinical",
            "differential_diagnosis": "clinical",
            "treatment_recommender": "clinical",
            "fhir_adapter": "integration",
            "hl7_parser": "integration",
            "patient_timeline": "integration",
            "audit_logger": "safety",
            "safety_checker": "safety",
            "hallucination_detector": "safety",
            "uncertainty_quantifier": "safety",
            "entity_extractor": "utilities",
            "medical_summarizer": "utilities",
            "patient_translator": "utilities",
            "report_generator": "utilities",
        }

        tool_key = tool_name.replace("-", "_")
        category = category_map.get(tool_key)

        if category:
            category_enabled = getattr(self.tools, f"{category}_enabled", True)
            if not category_enabled:
                return False

        return getattr(self.tools, attr_name, True)

    def ensure_directories(self) -> None:
        """Create necessary directories."""
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)
        self.resources.cache_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    @classmethod
    def from_file(cls, config_path: Path) -> "MedGemmaConfig":
        """Load configuration from a YAML or JSON file."""
        import json

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        suffix = config_path.suffix.lower()

        if suffix == ".json":
            with open(config_path) as f:
                data = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
                with open(config_path) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")

        return cls(**data)


# Global configuration instance
_config: Optional[MedGemmaConfig] = None


def get_config() -> MedGemmaConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = MedGemmaConfig()
    return _config


def set_config(config: MedGemmaConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _config
    _config = None


# Medical disclaimer text
MEDICAL_DISCLAIMER = """
IMPORTANT MEDICAL DISCLAIMER:
This tool provides AI-generated analysis for educational and informational purposes only.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always seek the advice of qualified healthcare providers with questions about medical conditions.
Never disregard professional medical advice or delay seeking it based on AI-generated information.
"""
