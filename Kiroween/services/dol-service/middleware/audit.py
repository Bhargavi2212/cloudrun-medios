"""
Audit middleware for DOL Service.

This middleware logs all API access for compliance and security monitoring
while ensuring no PHI is exposed in audit logs.
"""

import logging
import json
import time
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.audit_storage import AuditStorageService
from ..config import get_settings

logger = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """Audit logging middleware for compliance tracking."""
    
    def __init__(self, app):
        super().__init__(app)
        
        self.settings = get_settings()
        self.audit_storage = None  # Will be initialized on first request
        
        # Configure audit logger
        self.audit_logger = logging.getLogger("dol_audit")
        
        # Sensitive endpoints that require detailed auditing
        self.sensitive_endpoints = [
            "/api/federated/patient/import",
            "/api/federated/patient/export",
            "/api/timeline",
            "/api/model_update"
        ]
        
        # Fields to exclude from audit logs (potential PHI)
        self.excluded_fields = [
            "encrypted_profile_data",
            "clinical_summary",
            "structured_data",
            "demographics",
            "model_parameters"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process audit logging for requests."""
        
        # Initialize audit storage if not already done
        if self.audit_storage is None:
            self.audit_storage = AuditStorageService(self.settings.hospital_id)
        
        start_time = time.time()
        
        # Capture request information
        audit_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown'),
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "user_agent": request.headers.get("user-agent", "unknown"),
            "remote_addr": request.client.host if request.client else "unknown",
            "hospital_id": getattr(request.state, 'hospital_id', self.settings.hospital_id)
        }
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add response information
            audit_data.update({
                "status_code": response.status_code,
                "processing_time_ms": round(processing_time * 1000, 2),
                "success": 200 <= response.status_code < 400,
                "auth_method": getattr(request.state, 'auth_method', 'none')
            })
            
            # Log audit entry
            await self._log_audit_entry(audit_data, request, response)
            
            return response
            
        except Exception as e:
            # Log error in audit
            processing_time = time.time() - start_time
            audit_data.update({
                "status_code": 500,
                "processing_time_ms": round(processing_time * 1000, 2),
                "success": False,
                "error": str(e),
                "auth_method": getattr(request.state, 'auth_method', 'none')
            })
            
            await self._log_audit_entry(audit_data, request, None)
            raise
    
    async def _log_audit_entry(
        self,
        audit_data: dict,
        request: Request,
        response: Response = None
    ):
        """
        Log audit entry with privacy protection.
        
        Args:
            audit_data: Basic audit information
            request: FastAPI request object
            response: FastAPI response object (optional)
        """
        try:
            # Determine audit level based on endpoint sensitivity
            is_sensitive = any(
                endpoint in request.url.path 
                for endpoint in self.sensitive_endpoints
            )
            
            # Add endpoint-specific audit information
            if is_sensitive:
                audit_data["sensitive_operation"] = True
                audit_data["compliance_required"] = True
                
                # Add sanitized request details for sensitive operations
                if request.method in ["POST", "PUT", "PATCH"]:
                    audit_data["request_body_present"] = True
                    audit_data["content_type"] = request.headers.get("content-type", "unknown")
                    # Note: We don't log actual body content to protect PHI
            
            # Add response details (without sensitive content)
            if response:
                audit_data["response_headers"] = {
                    key: value for key, value in response.headers.items()
                    if key.lower() not in ["authorization", "x-api-key"]
                }
            
            # Store in audit database if available
            audit_storage = getattr(request.app.state, 'audit_storage', None)
            if audit_storage:
                await audit_storage.store_audit_log(audit_data)
            
            # Log based on operation type
            if audit_data["path"].startswith("/api/federated/patient"):
                await self._log_patient_operation(audit_data, request)
            elif audit_data["path"].startswith("/api/timeline"):
                await self._log_timeline_operation(audit_data, request)
            elif audit_data["path"].startswith("/api/model_update"):
                await self._log_federated_learning_operation(audit_data, request)
            elif audit_data["path"].startswith("/api/registry"):
                await self._log_registry_operation(audit_data, request)
            else:
                await self._log_general_operation(audit_data)
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            # Don't fail the request if audit logging fails
    
    async def _log_patient_operation(self, audit_data: dict, request: Request):
        """Log patient profile operations."""
        try:
            operation_type = "unknown"
            
            if "import" in audit_data["path"]:
                operation_type = "profile_import"
            elif "export" in audit_data["path"]:
                operation_type = "profile_export"
            elif "verify" in audit_data["path"]:
                operation_type = "profile_verification"
            elif "upload" in audit_data["path"]:
                operation_type = "profile_upload"
            
            audit_entry = {
                **audit_data,
                "operation_type": operation_type,
                "operation_category": "patient_profile",
                "privacy_impact": "high",
                "data_classification": "phi_protected"
            }
            
            # Log to audit system and storage
            self.audit_logger.info(
                f"PATIENT_OPERATION: {json.dumps(audit_entry, default=str)}"
            )
            
            # Store in audit database
            await self.audit_storage.store_audit_log(audit_entry)
            
        except Exception as e:
            logger.error(f"Patient operation audit failed: {e}")
    
    async def _log_timeline_operation(self, audit_data: dict, request: Request):
        """Log clinical timeline operations."""
        try:
            operation_type = "timeline_access"
            
            if request.method == "POST" and "append" in audit_data["path"]:
                operation_type = "timeline_append"
            elif request.method == "POST" and "search" in audit_data["path"]:
                operation_type = "timeline_search"
            elif "integrity" in audit_data["path"]:
                operation_type = "timeline_integrity_check"
            elif "summary" in audit_data["path"]:
                operation_type = "timeline_summary"
            
            audit_entry = {
                **audit_data,
                "operation_type": operation_type,
                "operation_category": "clinical_timeline",
                "privacy_impact": "high",
                "data_classification": "phi_protected"
            }
            
            self.audit_logger.info(
                f"TIMELINE_OPERATION: {json.dumps(audit_entry, default=str)}"
            )
            
            # Store in audit database
            await self.audit_storage.store_audit_log(audit_entry)
            
        except Exception as e:
            logger.error(f"Timeline operation audit failed: {e}")
    
    async def _log_federated_learning_operation(self, audit_data: dict, request: Request):
        """Log federated learning operations."""
        try:
            operation_type = "federated_learning"
            
            if "submit" in audit_data["path"]:
                operation_type = "model_parameter_submission"
            elif "receive" in audit_data["path"]:
                operation_type = "global_model_update"
            elif "train" in audit_data["path"]:
                operation_type = "local_training_trigger"
            elif "status" in audit_data["path"]:
                operation_type = "training_status_check"
            elif "privacy" in audit_data["path"]:
                operation_type = "privacy_validation"
            
            audit_entry = {
                **audit_data,
                "operation_type": operation_type,
                "operation_category": "federated_learning",
                "privacy_impact": "medium",
                "data_classification": "model_parameters_only"
            }
            
            self.audit_logger.info(
                f"FEDERATED_OPERATION: {json.dumps(audit_entry, default=str)}"
            )
            
            # Store in audit database
            await self.audit_storage.store_audit_log(audit_entry)
            
        except Exception as e:
            logger.error(f"Federated learning operation audit failed: {e}")
    
    async def _log_general_operation(self, audit_data: dict):
        """Log general API operations."""
        try:
            audit_entry = {
                **audit_data,
                "operation_type": "general_api",
                "operation_category": "system",
                "privacy_impact": "low",
                "data_classification": "system_metadata"
            }
            
            self.audit_logger.info(
                f"GENERAL_OPERATION: {json.dumps(audit_entry, default=str)}"
            )
            
            # Store in audit database
            await self.audit_storage.store_audit_log(audit_entry)
            
        except Exception as e:
            logger.error(f"General operation audit failed: {e}")
    
    def _sanitize_for_audit(self, data: dict) -> dict:
        """
        Sanitize data for audit logging to remove PHI.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data safe for audit logs
        """
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Skip fields that might contain PHI
            if any(excluded in key_lower for excluded in self.excluded_fields):
                sanitized[key] = "[REDACTED_FOR_PRIVACY]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_for_audit(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_for_audit(item) if isinstance(item, dict) else "[REDACTED]"
                    for item in value[:5]  # Limit list size in audit logs
                ]
            elif isinstance(value, str) and len(value) > 100:
                # Truncate long strings that might contain PHI
                sanitized[key] = value[:50] + "...[TRUNCATED]"
            else:
                sanitized[key] = value
        
        return sanitized 
   async def _log_registry_operation(self, audit_data: dict, request: Request):
        """Log peer registry operations."""
        try:
            operation_type = "registry_access"
            
            if "peers" in audit_data["path"]:
                if request.method == "POST":
                    operation_type = "peer_add"
                elif request.method == "DELETE":
                    operation_type = "peer_remove"
                elif request.method == "PUT":
                    operation_type = "peer_update"
                else:
                    operation_type = "peer_query"
            elif "audit" in audit_data["path"]:
                operation_type = "audit_access"
            
            audit_entry = {
                **audit_data,
                "operation_type": operation_type,
                "operation_category": "peer_registry",
                "privacy_impact": "low",
                "data_classification": "peer_metadata"
            }
            
            self.audit_logger.info(
                f"REGISTRY_OPERATION: {json.dumps(audit_entry, default=str)}"
            )
            
        except Exception as e:
            logger.error(f"Registry operation audit failed: {e}")