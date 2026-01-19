"""
MedGemma Agent Framework - DICOM Handler Tool

Handles loading, preprocessing, and metadata extraction from DICOM files.
Supports single files and multi-slice series.

Usage:
    handler = DicomHandlerTool()
    result = await handler.run({
        "dicom_path": "/path/to/image.dcm",
        "extract_pixels": True,
        "anonymize": True
    })
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

import numpy as np
from pydantic import BaseModel, Field

from tools.base import (
    BaseTool,
    ToolCategory,
    ToolInput,
    ToolOutput,
    ToolStatus,
)


class DicomHandlerInput(ToolInput):
    """Input schema for DICOM handler."""

    dicom_path: Optional[str] = Field(
        default=None,
        description="Path to DICOM file or directory"
    )
    dicom_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw DICOM bytes"
    )
    extract_pixels: bool = Field(
        default=True,
        description="Extract pixel data"
    )
    anonymize: bool = Field(
        default=False,
        description="Remove PHI from metadata"
    )
    normalize: bool = Field(
        default=True,
        description="Normalize pixel values"
    )
    window_center: Optional[float] = Field(
        default=None,
        description="Window center for windowing"
    )
    window_width: Optional[float] = Field(
        default=None,
        description="Window width for windowing"
    )

    def model_post_init(self, __context):
        if not self.dicom_path and not self.dicom_bytes:
            raise ValueError("Either dicom_path or dicom_bytes must be provided")


class DicomMetadata(BaseModel):
    """DICOM metadata extracted from file."""

    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    study_description: Optional[str] = None
    series_description: Optional[str] = None
    modality: Optional[str] = None
    manufacturer: Optional[str] = None
    institution_name: Optional[str] = None
    body_part_examined: Optional[str] = None
    slice_thickness: Optional[float] = None
    pixel_spacing: Optional[List[float]] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    bits_allocated: Optional[int] = None
    bits_stored: Optional[int] = None
    photometric_interpretation: Optional[str] = None
    rescale_slope: Optional[float] = None
    rescale_intercept: Optional[float] = None
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    sop_class_uid: Optional[str] = None
    sop_instance_uid: Optional[str] = None
    series_instance_uid: Optional[str] = None
    study_instance_uid: Optional[str] = None


class DicomHandlerOutput(ToolOutput):
    """Output schema for DICOM handler."""

    metadata: Optional[DicomMetadata] = Field(
        default=None,
        description="Extracted DICOM metadata"
    )
    pixel_array_shape: Optional[List[int]] = Field(
        default=None,
        description="Shape of pixel array"
    )
    pixel_array_dtype: Optional[str] = Field(
        default=None,
        description="Data type of pixel array"
    )
    pixel_array_min: Optional[float] = Field(
        default=None,
        description="Minimum pixel value"
    )
    pixel_array_max: Optional[float] = Field(
        default=None,
        description="Maximum pixel value"
    )
    num_slices: Optional[int] = Field(
        default=None,
        description="Number of slices in series"
    )
    is_compressed: Optional[bool] = Field(
        default=None,
        description="Whether pixel data is compressed"
    )


# PHI tags to anonymize
PHI_TAGS = [
    (0x0010, 0x0010),  # Patient Name
    (0x0010, 0x0020),  # Patient ID
    (0x0010, 0x0030),  # Patient Birth Date
    (0x0010, 0x1000),  # Other Patient IDs
    (0x0010, 0x1001),  # Other Patient Names
    (0x0010, 0x1010),  # Patient Age (optional)
    (0x0008, 0x0080),  # Institution Name
    (0x0008, 0x0081),  # Institution Address
    (0x0008, 0x0090),  # Referring Physician Name
    (0x0008, 0x1048),  # Physician of Record
    (0x0008, 0x1050),  # Performing Physician Name
]


class DicomHandlerTool(BaseTool[DicomHandlerInput, DicomHandlerOutput]):
    """
    Tool for loading and preprocessing DICOM medical images.

    Features:
    - Load single DICOM files or series
    - Extract comprehensive metadata
    - Apply windowing and normalization
    - Optionally anonymize PHI
    - Support for compressed transfer syntaxes
    """

    name: ClassVar[str] = "dicom_handler"
    description: ClassVar[str] = (
        "Load and preprocess DICOM medical images. Extracts metadata, "
        "pixel data, and supports windowing and anonymization."
    )
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.IMAGING

    input_class: ClassVar[Type[DicomHandlerInput]] = DicomHandlerInput
    output_class: ClassVar[Type[DicomHandlerOutput]] = DicomHandlerOutput

    def __init__(self):
        super().__init__()
        self._pydicom = None

    async def setup(self) -> None:
        """Load pydicom library."""
        try:
            import pydicom
            self._pydicom = pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required for DICOM handling. "
                "Install with: pip install pydicom"
            )
        await super().setup()

    async def execute(self, input: DicomHandlerInput) -> DicomHandlerOutput:
        """Execute DICOM loading and preprocessing."""
        try:
            # Load DICOM file(s)
            if input.dicom_path:
                path = Path(input.dicom_path)
                if path.is_dir():
                    # Load series
                    ds_list = self._load_series(path)
                    if not ds_list:
                        return DicomHandlerOutput.from_error(
                            "No valid DICOM files found in directory"
                        )
                    ds = ds_list[0]  # Use first for metadata
                    num_slices = len(ds_list)
                else:
                    ds = self._pydicom.dcmread(str(path))
                    num_slices = 1
            else:
                ds = self._pydicom.dcmread(io.BytesIO(input.dicom_bytes))
                num_slices = 1

            # Anonymize if requested
            if input.anonymize:
                ds = self._anonymize(ds)

            # Extract metadata
            metadata = self._extract_metadata(ds)

            # Extract pixel data
            pixel_info = {}
            if input.extract_pixels and hasattr(ds, "PixelData"):
                try:
                    pixel_array = ds.pixel_array

                    # Apply rescale
                    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                        pixel_array = (
                            pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                        )

                    # Apply windowing
                    if input.window_center and input.window_width:
                        pixel_array = self._apply_window(
                            pixel_array,
                            input.window_center,
                            input.window_width
                        )
                    elif metadata.window_center and metadata.window_width:
                        pixel_array = self._apply_window(
                            pixel_array,
                            metadata.window_center,
                            metadata.window_width
                        )

                    # Normalize
                    if input.normalize:
                        pixel_array = self._normalize(pixel_array)

                    pixel_info = {
                        "pixel_array_shape": list(pixel_array.shape),
                        "pixel_array_dtype": str(pixel_array.dtype),
                        "pixel_array_min": float(pixel_array.min()),
                        "pixel_array_max": float(pixel_array.max()),
                    }
                except Exception as e:
                    pixel_info = {"pixel_error": str(e)}

            # Check if compressed
            is_compressed = (
                hasattr(ds, "file_meta") and
                hasattr(ds.file_meta, "TransferSyntaxUID") and
                ds.file_meta.TransferSyntaxUID.is_compressed
            )

            return DicomHandlerOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={
                    "metadata": metadata.model_dump(),
                    "num_slices": num_slices,
                    "is_compressed": is_compressed,
                    **pixel_info
                },
                metadata=metadata,
                num_slices=num_slices,
                is_compressed=is_compressed,
                **pixel_info
            )

        except Exception as e:
            return DicomHandlerOutput.from_error(f"DICOM loading failed: {str(e)}")

    def _load_series(self, directory: Path) -> List:
        """Load all DICOM files in a directory as a series."""
        ds_list = []
        for file_path in sorted(directory.glob("*")):
            if file_path.is_file():
                try:
                    ds = self._pydicom.dcmread(str(file_path))
                    ds_list.append(ds)
                except Exception:
                    continue
        return ds_list

    def _extract_metadata(self, ds) -> DicomMetadata:
        """Extract metadata from DICOM dataset."""

        def get_value(ds, tag, default=None):
            """Safely get value from dataset."""
            try:
                if hasattr(ds, tag):
                    val = getattr(ds, tag)
                    if val is not None:
                        return str(val) if not isinstance(val, (int, float, list)) else val
            except Exception:
                pass
            return default

        # Extract pixel spacing as list
        pixel_spacing = None
        if hasattr(ds, "PixelSpacing"):
            try:
                pixel_spacing = [float(x) for x in ds.PixelSpacing]
            except Exception:
                pass

        return DicomMetadata(
            patient_id=get_value(ds, "PatientID"),
            patient_name=get_value(ds, "PatientName"),
            patient_age=get_value(ds, "PatientAge"),
            patient_sex=get_value(ds, "PatientSex"),
            study_date=get_value(ds, "StudyDate"),
            study_time=get_value(ds, "StudyTime"),
            study_description=get_value(ds, "StudyDescription"),
            series_description=get_value(ds, "SeriesDescription"),
            modality=get_value(ds, "Modality"),
            manufacturer=get_value(ds, "Manufacturer"),
            institution_name=get_value(ds, "InstitutionName"),
            body_part_examined=get_value(ds, "BodyPartExamined"),
            slice_thickness=get_value(ds, "SliceThickness"),
            pixel_spacing=pixel_spacing,
            rows=get_value(ds, "Rows"),
            columns=get_value(ds, "Columns"),
            bits_allocated=get_value(ds, "BitsAllocated"),
            bits_stored=get_value(ds, "BitsStored"),
            photometric_interpretation=get_value(ds, "PhotometricInterpretation"),
            rescale_slope=get_value(ds, "RescaleSlope"),
            rescale_intercept=get_value(ds, "RescaleIntercept"),
            window_center=self._get_window_value(ds, "WindowCenter"),
            window_width=self._get_window_value(ds, "WindowWidth"),
            sop_class_uid=get_value(ds, "SOPClassUID"),
            sop_instance_uid=get_value(ds, "SOPInstanceUID"),
            series_instance_uid=get_value(ds, "SeriesInstanceUID"),
            study_instance_uid=get_value(ds, "StudyInstanceUID"),
        )

    def _get_window_value(self, ds, attr: str) -> Optional[float]:
        """Get window value (may be a sequence)."""
        try:
            if hasattr(ds, attr):
                val = getattr(ds, attr)
                if isinstance(val, (list, tuple)):
                    return float(val[0])
                return float(val)
        except Exception:
            pass
        return None

    def _anonymize(self, ds):
        """Remove PHI from DICOM dataset."""
        for tag in PHI_TAGS:
            if tag in ds:
                del ds[tag]
        return ds

    def _apply_window(
        self,
        pixel_array: np.ndarray,
        center: float,
        width: float
    ) -> np.ndarray:
        """Apply window/level to pixel array."""
        lower = center - width / 2
        upper = center + width / 2
        pixel_array = np.clip(pixel_array, lower, upper)
        return pixel_array

    def _normalize(self, pixel_array: np.ndarray) -> np.ndarray:
        """Normalize pixel array to 0-1 range."""
        min_val = pixel_array.min()
        max_val = pixel_array.max()
        if max_val > min_val:
            return (pixel_array - min_val) / (max_val - min_val)
        return pixel_array
