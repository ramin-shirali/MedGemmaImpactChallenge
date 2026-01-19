"""
MedGemma Agent Framework - HL7 Parser Tool

Parses HL7 v2.x messages.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class HL7ParserInput(ToolInput):
    """Input for HL7 parser."""
    message: str = Field(description="HL7 v2.x message string")
    extract_patient: bool = Field(default=True, description="Extract patient info")
    extract_observations: bool = Field(default=True, description="Extract OBX segments")


class HL7ParserOutput(ToolOutput):
    """Output for HL7 parser."""
    message_type: Optional[str] = None
    message_control_id: Optional[str] = None
    patient_info: Optional[Dict[str, Any]] = None
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    segments: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class HL7ParserTool(BaseTool[HL7ParserInput, HL7ParserOutput]):
    """Parse HL7 v2.x messages."""

    name: ClassVar[str] = "hl7_parser"
    description: ClassVar[str] = "Parse HL7 v2.x messages to extract patient data, observations, and structured segments."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.INTEGRATION

    input_class: ClassVar[Type[HL7ParserInput]] = HL7ParserInput
    output_class: ClassVar[Type[HL7ParserOutput]] = HL7ParserOutput

    async def execute(self, input: HL7ParserInput) -> HL7ParserOutput:
        try:
            message = input.message.replace('\r\n', '\r').replace('\n', '\r')
            segments = self._parse_segments(message)

            # Extract MSH info
            # Note: In HL7, MSH-1 is the field separator, so our parsed indices are off by 1
            # MSH-9 (message type) is at key '8', MSH-10 (control id) is at key '9'
            msh = segments.get('MSH', [{}])[0]
            msg_type_field = msh.get('8', {})
            if isinstance(msg_type_field, dict):
                message_type = msg_type_field.get('1', '') + '^' + msg_type_field.get('2', '')
            else:
                message_type = str(msg_type_field) if msg_type_field else ''
            control_id = msh.get('9', '')

            # Extract patient
            patient_info = None
            if input.extract_patient and 'PID' in segments:
                patient_info = self._parse_pid(segments['PID'][0])

            # Extract observations
            observations = []
            if input.extract_observations and 'OBX' in segments:
                for obx in segments['OBX']:
                    observations.append(self._parse_obx(obx))

            return HL7ParserOutput(
                success=True,
                status=ToolStatus.SUCCESS,
                data={"segment_count": len(segments)},
                message_type=message_type,
                message_control_id=control_id,
                patient_info=patient_info,
                observations=observations,
                segments=segments,
                confidence=0.9
            )

        except Exception as e:
            return HL7ParserOutput.from_error(str(e))

    def _parse_segments(self, message: str) -> Dict[str, List[Dict]]:
        """Parse message into segments."""
        segments: Dict[str, List[Dict]] = {}
        lines = message.split('\r')

        for line in lines:
            if not line.strip():
                continue

            # Parse segment
            fields = line.split('|')
            segment_name = fields[0]

            if segment_name not in segments:
                segments[segment_name] = []

            # Parse fields into dict
            segment_data = {}
            for i, field in enumerate(fields[1:], 1):
                if field:
                    # Handle components
                    components = field.split('^')
                    if len(components) > 1:
                        segment_data[str(i)] = {str(j+1): c for j, c in enumerate(components) if c}
                    else:
                        segment_data[str(i)] = field

            segments[segment_name].append(segment_data)

        return segments

    def _parse_pid(self, pid: Dict) -> Dict[str, Any]:
        """Parse PID segment for patient info."""
        patient = {}

        # Patient ID (PID-3)
        pid3 = pid.get('3', {})
        if isinstance(pid3, dict):
            patient['patient_id'] = pid3.get('1', '')
        else:
            patient['patient_id'] = pid3

        # Patient Name (PID-5)
        pid5 = pid.get('5', {})
        if isinstance(pid5, dict):
            patient['family_name'] = pid5.get('1', '')
            patient['given_name'] = pid5.get('2', '')
            patient['name'] = f"{pid5.get('2', '')} {pid5.get('1', '')}"
        else:
            patient['name'] = pid5

        # DOB (PID-7)
        patient['date_of_birth'] = pid.get('7', '')

        # Sex (PID-8)
        patient['sex'] = pid.get('8', '')

        # Address (PID-11)
        pid11 = pid.get('11', {})
        if isinstance(pid11, dict):
            patient['address'] = {
                'street': pid11.get('1', ''),
                'city': pid11.get('3', ''),
                'state': pid11.get('4', ''),
                'zip': pid11.get('5', '')
            }

        # Phone (PID-13)
        patient['phone'] = pid.get('13', '')

        return patient

    def _parse_obx(self, obx: Dict) -> Dict[str, Any]:
        """Parse OBX segment for observation."""
        observation = {}

        # Set ID (OBX-1)
        observation['set_id'] = obx.get('1', '')

        # Value Type (OBX-2)
        observation['value_type'] = obx.get('2', '')

        # Observation ID (OBX-3)
        obx3 = obx.get('3', {})
        if isinstance(obx3, dict):
            observation['code'] = obx3.get('1', '')
            observation['code_text'] = obx3.get('2', '')
            observation['code_system'] = obx3.get('3', '')
        else:
            observation['code'] = obx3

        # Value (OBX-5)
        observation['value'] = obx.get('5', '')

        # Units (OBX-6)
        obx6 = obx.get('6', {})
        if isinstance(obx6, dict):
            observation['units'] = obx6.get('1', '')
        else:
            observation['units'] = obx6

        # Reference Range (OBX-7)
        observation['reference_range'] = obx.get('7', '')

        # Abnormal Flags (OBX-8)
        observation['abnormal_flag'] = obx.get('8', '')

        # Status (OBX-11)
        observation['status'] = obx.get('11', '')

        return observation
