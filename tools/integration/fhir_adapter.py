"""
MedGemma Agent Framework - FHIR Adapter Tool

Handles FHIR R4 resource parsing and generation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import Field

from tools.base import BaseTool, ToolCategory, ToolInput, ToolOutput, ToolStatus


class FHIRAdapterInput(ToolInput):
    """Input for FHIR adapter."""
    operation: str = Field(description="Operation: parse, create, validate, search")
    resource_type: Optional[str] = Field(default=None, description="FHIR resource type")
    resource_data: Optional[Dict[str, Any]] = Field(default=None, description="Resource data")
    fhir_json: Optional[str] = Field(default=None, description="FHIR JSON string")
    search_params: Optional[Dict[str, str]] = Field(default=None, description="Search parameters")


class FHIRAdapterOutput(ToolOutput):
    """Output for FHIR adapter."""
    resource: Optional[Dict[str, Any]] = None
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    resource_type: Optional[str] = None
    validation_errors: List[str] = Field(default_factory=list)
    is_valid: bool = True


class FHIRAdapterTool(BaseTool[FHIRAdapterInput, FHIRAdapterOutput]):
    """Parse and generate FHIR R4 resources."""

    name: ClassVar[str] = "fhir_adapter"
    description: ClassVar[str] = "Parse, create, and validate FHIR R4 healthcare interoperability resources."
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[ToolCategory] = ToolCategory.INTEGRATION

    input_class: ClassVar[Type[FHIRAdapterInput]] = FHIRAdapterInput
    output_class: ClassVar[Type[FHIRAdapterOutput]] = FHIRAdapterOutput

    SUPPORTED_RESOURCES = [
        "Patient", "Observation", "Condition", "MedicationRequest",
        "DiagnosticReport", "Procedure", "Encounter", "AllergyIntolerance"
    ]

    async def execute(self, input: FHIRAdapterInput) -> FHIRAdapterOutput:
        try:
            operation = input.operation.lower()

            if operation == "parse":
                return self._parse_resource(input.fhir_json or input.resource_data)
            elif operation == "create":
                return self._create_resource(input.resource_type, input.resource_data)
            elif operation == "validate":
                return self._validate_resource(input.resource_data)
            else:
                return FHIRAdapterOutput.from_error(f"Unknown operation: {operation}")

        except Exception as e:
            return FHIRAdapterOutput.from_error(str(e))

    def _parse_resource(self, data: Any) -> FHIRAdapterOutput:
        """Parse FHIR resource from JSON."""
        import json

        if isinstance(data, str):
            data = json.loads(data)

        resource_type = data.get("resourceType")
        if not resource_type:
            return FHIRAdapterOutput.from_error("Missing resourceType")

        # Extract key information based on resource type
        parsed = {"resourceType": resource_type, "id": data.get("id")}

        if resource_type == "Patient":
            parsed.update(self._parse_patient(data))
        elif resource_type == "Observation":
            parsed.update(self._parse_observation(data))
        elif resource_type == "Condition":
            parsed.update(self._parse_condition(data))
        elif resource_type == "MedicationRequest":
            parsed.update(self._parse_medication_request(data))

        return FHIRAdapterOutput(
            success=True,
            status=ToolStatus.SUCCESS,
            data=parsed,
            resource=parsed,
            resource_type=resource_type,
            confidence=0.9
        )

    def _parse_patient(self, data: Dict) -> Dict:
        """Parse Patient resource."""
        result = {}

        # Name
        if data.get("name"):
            name = data["name"][0]
            result["name"] = f"{' '.join(name.get('given', []))} {name.get('family', '')}"

        # Identifiers
        if data.get("identifier"):
            result["identifiers"] = [
                {"system": i.get("system"), "value": i.get("value")}
                for i in data["identifier"]
            ]

        # Demographics
        result["birthDate"] = data.get("birthDate")
        result["gender"] = data.get("gender")

        # Contact
        if data.get("telecom"):
            result["telecom"] = [
                {"system": t.get("system"), "value": t.get("value")}
                for t in data["telecom"]
            ]

        return result

    def _parse_observation(self, data: Dict) -> Dict:
        """Parse Observation resource."""
        result = {
            "status": data.get("status"),
            "category": self._get_codeable_concept(data.get("category", [{}])[0] if data.get("category") else {}),
            "code": self._get_codeable_concept(data.get("code", {})),
            "effectiveDateTime": data.get("effectiveDateTime"),
        }

        # Value
        if data.get("valueQuantity"):
            vq = data["valueQuantity"]
            result["value"] = f"{vq.get('value')} {vq.get('unit', '')}"
        elif data.get("valueString"):
            result["value"] = data["valueString"]
        elif data.get("valueCodeableConcept"):
            result["value"] = self._get_codeable_concept(data["valueCodeableConcept"])

        # Reference range
        if data.get("referenceRange"):
            rr = data["referenceRange"][0]
            low = rr.get("low", {}).get("value", "")
            high = rr.get("high", {}).get("value", "")
            result["referenceRange"] = f"{low}-{high}"

        return result

    def _parse_condition(self, data: Dict) -> Dict:
        """Parse Condition resource."""
        return {
            "clinicalStatus": self._get_codeable_concept(data.get("clinicalStatus", {})),
            "verificationStatus": self._get_codeable_concept(data.get("verificationStatus", {})),
            "code": self._get_codeable_concept(data.get("code", {})),
            "onsetDateTime": data.get("onsetDateTime"),
            "recordedDate": data.get("recordedDate"),
        }

    def _parse_medication_request(self, data: Dict) -> Dict:
        """Parse MedicationRequest resource."""
        result = {
            "status": data.get("status"),
            "intent": data.get("intent"),
            "authoredOn": data.get("authoredOn"),
        }

        if data.get("medicationCodeableConcept"):
            result["medication"] = self._get_codeable_concept(data["medicationCodeableConcept"])

        if data.get("dosageInstruction"):
            di = data["dosageInstruction"][0]
            result["dosage"] = di.get("text")

        return result

    def _get_codeable_concept(self, cc: Dict) -> str:
        """Extract display text from CodeableConcept."""
        if cc.get("text"):
            return cc["text"]
        if cc.get("coding"):
            return cc["coding"][0].get("display", cc["coding"][0].get("code", ""))
        return ""

    def _create_resource(self, resource_type: str, data: Dict) -> FHIRAdapterOutput:
        """Create a FHIR resource."""
        if resource_type not in self.SUPPORTED_RESOURCES:
            return FHIRAdapterOutput.from_error(f"Unsupported resource type: {resource_type}")

        resource = {
            "resourceType": resource_type,
            "id": data.get("id", f"example-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            "meta": {
                "lastUpdated": datetime.now().isoformat()
            }
        }

        if resource_type == "Patient":
            resource.update(self._create_patient(data))
        elif resource_type == "Observation":
            resource.update(self._create_observation(data))
        elif resource_type == "Condition":
            resource.update(self._create_condition(data))

        return FHIRAdapterOutput(
            success=True,
            status=ToolStatus.SUCCESS,
            data=resource,
            resource=resource,
            resource_type=resource_type,
            confidence=0.95
        )

    def _create_patient(self, data: Dict) -> Dict:
        """Create Patient resource structure."""
        resource = {}

        if data.get("name"):
            parts = data["name"].split()
            resource["name"] = [{
                "use": "official",
                "family": parts[-1] if parts else "",
                "given": parts[:-1] if len(parts) > 1 else []
            }]

        if data.get("birthDate"):
            resource["birthDate"] = data["birthDate"]
        if data.get("gender"):
            resource["gender"] = data["gender"]

        return resource

    def _create_observation(self, data: Dict) -> Dict:
        """Create Observation resource structure."""
        resource = {
            "status": data.get("status", "final"),
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": data.get("loinc_code", ""),
                    "display": data.get("name", "")
                }]
            }
        }

        if data.get("value") is not None:
            resource["valueQuantity"] = {
                "value": data["value"],
                "unit": data.get("unit", ""),
                "system": "http://unitsofmeasure.org"
            }

        return resource

    def _create_condition(self, data: Dict) -> Dict:
        """Create Condition resource structure."""
        return {
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": data.get("clinical_status", "active")
                }]
            },
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": data.get("snomed_code", ""),
                    "display": data.get("diagnosis", "")
                }],
                "text": data.get("diagnosis", "")
            },
            "recordedDate": datetime.now().strftime("%Y-%m-%d")
        }

    def _validate_resource(self, data: Dict) -> FHIRAdapterOutput:
        """Validate FHIR resource structure."""
        errors = []

        if not data.get("resourceType"):
            errors.append("Missing required field: resourceType")

        resource_type = data.get("resourceType", "")

        # Basic validation by type
        if resource_type == "Patient":
            pass  # Minimal required fields
        elif resource_type == "Observation":
            if not data.get("status"):
                errors.append("Observation: missing status")
            if not data.get("code"):
                errors.append("Observation: missing code")
        elif resource_type == "Condition":
            if not data.get("code"):
                errors.append("Condition: missing code")

        return FHIRAdapterOutput(
            success=len(errors) == 0,
            status=ToolStatus.SUCCESS if not errors else ToolStatus.PARTIAL,
            data={"valid": len(errors) == 0},
            resource=data,
            resource_type=resource_type,
            validation_errors=errors,
            is_valid=len(errors) == 0,
            confidence=0.9
        )
