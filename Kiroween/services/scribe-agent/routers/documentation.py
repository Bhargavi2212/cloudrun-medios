"""
Clinical documentation endpoints for SOAP note generation.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from ..schemas import (
    SOAPNoteRequest,
    SOAPNoteResponse,
    ClinicalDocumentationRequest,
    ClinicalDocumentationResponse
)
from ..services.documentation_service import DocumentationService

router = APIRouter()
documentation_service = DocumentationService()


@router.post("/soap-note", response_model=SOAPNoteResponse)
async def generate_soap_note(request: SOAPNoteRequest):
    """Generate SOAP note from clinical dialogue or structured data."""
    try:
        soap_note = await documentation_service.generate_soap_note(
            dialogue=request.dialogue,
            patient_context=request.patient_context,
            structured_data=request.structured_data
        )
        
        return SOAPNoteResponse(
            patient_id=request.patient_id,
            soap_note=soap_note,
            confidence_score=0.85,  # TODO: Implement actual confidence scoring
            generated_at="2024-01-01T00:00:00Z",  # TODO: Use actual timestamp
            model_used="gemini-1.5-pro"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate SOAP note: {str(e)}"
        )


@router.post("/clinical-summary", response_model=ClinicalDocumentationResponse)
async def generate_clinical_summary(request: ClinicalDocumentationRequest):
    """Generate clinical summary from raw medical text."""
    try:
        summary = await documentation_service.generate_clinical_summary(
            raw_text=request.raw_text,
            document_type=request.document_type,
            patient_context=request.patient_context
        )
        
        return ClinicalDocumentationResponse(
            patient_id=request.patient_id,
            document_type=request.document_type,
            generated_content=summary,
            confidence_score=0.82,
            generated_at="2024-01-01T00:00:00Z",
            model_used="gemini-1.5-pro"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate clinical summary: {str(e)}"
        )


@router.post("/structured-extraction")
async def extract_structured_data(request: ClinicalDocumentationRequest):
    """Extract structured medical data from unstructured text."""
    try:
        structured_data = await documentation_service.extract_structured_data(
            raw_text=request.raw_text,
            extraction_schema=request.extraction_schema
        )
        
        return {
            "patient_id": request.patient_id,
            "extracted_data": structured_data,
            "extraction_confidence": 0.88,
            "extracted_at": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract structured data: {str(e)}"
        )


@router.get("/templates/{template_type}")
async def get_documentation_template(template_type: str):
    """Get documentation templates for different clinical scenarios."""
    templates = {
        "soap": {
            "subjective": "Patient reports...",
            "objective": "Vital signs: BP {bp}, HR {hr}, Temp {temp}...",
            "assessment": "Clinical impression...",
            "plan": "Treatment plan..."
        },
        "discharge_summary": {
            "admission_diagnosis": "",
            "discharge_diagnosis": "",
            "hospital_course": "",
            "discharge_medications": "",
            "follow_up": ""
        },
        "progress_note": {
            "interval_history": "",
            "physical_exam": "",
            "assessment_plan": ""
        }
    }
    
    template = templates.get(template_type)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_type}' not found"
        )
    
    return {
        "template_type": template_type,
        "template": template,
        "description": f"Standard template for {template_type.replace('_', ' ')}"
    }