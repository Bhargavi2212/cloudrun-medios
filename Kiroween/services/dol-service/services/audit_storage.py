"""
Audit Storage Service for DOL.

This service provides secure, compliant audit logging storage
without exposing PHI while maintaining comprehensive audit trails.
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLevel(str, Enum):
    """Audit logging levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(str, Enum):
    """Categories of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    PATIENT_ACCESS = "patient_access"
    PROFILE_IMPORT = "profile_import"
    PROFILE_EXPORT = "profile_export"
    TIMELINE_ACCESS = "timeline_access"
    FEDERATED_LEARNING = "federated_learning"
    SYSTEM_ADMIN = "system_admin"
    PRIVACY_VIOLATION = "privacy_violation"
    SECURITY_EVENT = "security_event"


class AuditEvent:
    """Represents a single audit event."""
    
    def __init__(
        self,
        event_id: str,
        category: AuditCategory,
        level: AuditLevel,
        message: str,
        hospital_id: str,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        request_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        self.event_id = event_id
        self.category = category
        self.level = level
        self.message = message
        self.hospital_id = hospital_id
        self.user_id = user_id
        self.patient_id = patient_id  # Hashed for privacy
        self.request_id = request_id
        self.additional_data = additional_data or {}
        self.timestamp = datetime.utcnow()
        
        # Privacy protection: hash patient ID if provided
        if self.patient_id:
            self.patient_id_hash = self._hash_patient_id(self.patient_id)
            self.patient_id = None  # Remove original ID for privacy
        else:
            self.patient_id_hash = None
    
    def _hash_patient_id(self, patient_id: str) -> str:
        """Hash patient ID for privacy protection."""
        # Use SHA-256 with salt for patient ID hashing
        salt = f"audit_salt_{self.hospital_id}"
        return hashlib.sha256(f"{patient_id}{salt}".encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "level": self.level.value,
            "message": self.message,
            "hospital_id": self.hospital_id,
            "user_id": self.user_id,
            "patient_id_hash": self.patient_id_hash,
            "request_id": self.request_id,
            "additional_data": self.additional_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create audit event from dictionary."""
        event = cls(
            event_id=data["event_id"],
            category=AuditCategory(data["category"]),
            level=AuditLevel(data["level"]),
            message=data["message"],
            hospital_id=data["hospital_id"],
            user_id=data.get("user_id"),
            request_id=data.get("request_id"),
            additional_data=data.get("additional_data", {})
        )
        
        # Restore timestamp and patient ID hash
        event.timestamp = datetime.fromisoformat(data["timestamp"])
        event.patient_id_hash = data.get("patient_id_hash")
        
        return event


class AuditStorageService:
    """Service for secure audit logging storage."""
    
    def __init__(
        self,
        hospital_id: str,
        audit_directory: str = "./logs/audit",
        max_file_size_mb: int = 100,
        retention_days: int = 2555  # 7 years for healthcare compliance
    ):
        self.hospital_id = hospital_id
        self.audit_directory = Path(audit_directory)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.retention_days = retention_days
        
        # Ensure audit directory exists
        self.audit_directory.mkdir(parents=True, exist_ok=True)
        
        # Current audit file
        self.current_audit_file = None
        self._initialize_audit_file()
        
        # In-memory cache for recent events (for quick queries)
        self.recent_events: List[AuditEvent] = []
        self.max_cache_size = 1000
        
        logger.info(f"Initialized AuditStorageService for hospital {hospital_id}")
    
    async def log_event(
        self,
        category: AuditCategory,
        level: AuditLevel,
        message: str,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        request_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event.
        
        Args:
            category: Event category
            level: Audit level
            message: Audit message
            user_id: User identifier (optional)
            patient_id: Patient identifier (will be hashed for privacy)
            request_id: Request identifier (optional)
            additional_data: Additional event data (optional)
            
        Returns:
            Event ID of the logged event
        """
        try:
            # Generate unique event ID
            event_id = self._generate_event_id()
            
            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                category=category,
                level=level,
                message=message,
                hospital_id=self.hospital_id,
                user_id=user_id,
                patient_id=patient_id,
                request_id=request_id,
                additional_data=self._sanitize_additional_data(additional_data)
            )
            
            # Add to cache
            self.recent_events.append(event)
            if len(self.recent_events) > self.max_cache_size:
                self.recent_events.pop(0)
            
            # Write to audit file
            await self._write_event_to_file(event)
            
            # Check if file rotation is needed
            await self._check_file_rotation()
            
            logger.debug(f"Logged audit event {event_id}: {category.value}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise
    
    async def log_authentication_event(
        self,
        success: bool,
        user_id: Optional[str] = None,
        method: str = "unknown",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Log authentication event."""
        message = f"Authentication {'successful' if success else 'failed'} using {method}"
        level = AuditLevel.INFO if success else AuditLevel.WARNING
        
        additional_data = {
            "authentication_method": method,
            "success": success
        }
        
        if ip_address:
            additional_data["ip_address"] = ip_address
        if user_agent:
            additional_data["user_agent"] = user_agent[:100]  # Truncate for storage
        
        return await self.log_event(
            category=AuditCategory.AUTHENTICATION,
            level=level,
            message=message,
            user_id=user_id,
            additional_data=additional_data
        )
    
    async def log_patient_access_event(
        self,
        patient_id: str,
        operation: str,
        user_id: Optional[str] = None,
        success: bool = True,
        details: Optional[str] = None
    ) -> str:
        """Log patient data access event."""
        message = f"Patient data {operation}: {'successful' if success else 'failed'}"
        if details:
            message += f" - {details}"
        
        level = AuditLevel.INFO if success else AuditLevel.ERROR
        
        return await self.log_event(
            category=AuditCategory.PATIENT_ACCESS,
            level=level,
            message=message,
            user_id=user_id,
            patient_id=patient_id,
            additional_data={
                "operation": operation,
                "success": success
            }
        )
    
    async def log_privacy_violation(
        self,
        violation_type: str,
        description: str,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        severity: str = "medium"
    ) -> str:
        """Log privacy violation event."""
        message = f"Privacy violation detected: {violation_type} - {description}"
        
        level = AuditLevel.CRITICAL if severity == "high" else AuditLevel.WARNING
        
        return await self.log_event(
            category=AuditCategory.PRIVACY_VIOLATION,
            level=level,
            message=message,
            user_id=user_id,
            patient_id=patient_id,
            additional_data={
                "violation_type": violation_type,
                "severity": severity,
                "description": description
            }
        )
    
    async def log_federated_learning_event(
        self,
        operation: str,
        model_type: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log federated learning event."""
        message = f"Federated learning {operation} for {model_type}: {'successful' if success else 'failed'}"
        
        additional_data = {
            "operation": operation,
            "model_type": model_type,
            "success": success
        }
        
        if details:
            additional_data.update(details)
        
        return await self.log_event(
            category=AuditCategory.FEDERATED_LEARNING,
            level=AuditLevel.INFO,
            message=message,
            additional_data=additional_data
        )
    
    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        category: Optional[AuditCategory] = None,
        level: Optional[AuditLevel] = None,
        user_id: Optional[str] = None,
        patient_id_hash: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Query audit events with filters.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            category: Event category filter
            level: Audit level filter
            user_id: User ID filter
            patient_id_hash: Patient ID hash filter
            limit: Maximum number of events to return
            
        Returns:
            List of matching audit events
        """
        try:
            # Start with recent events cache
            matching_events = []
            
            for event in reversed(self.recent_events):  # Most recent first
                # Apply filters
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if category and event.category != category:
                    continue
                if level and event.level != level:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if patient_id_hash and event.patient_id_hash != patient_id_hash:
                    continue
                
                matching_events.append(event)
                
                if len(matching_events) >= limit:
                    break
            
            # If we need more events, search audit files
            if len(matching_events) < limit:
                file_events = await self._search_audit_files(
                    start_time=start_time,
                    end_time=end_time,
                    category=category,
                    level=level,
                    user_id=user_id,
                    patient_id_hash=patient_id_hash,
                    limit=limit - len(matching_events)
                )
                matching_events.extend(file_events)
            
            return matching_events[:limit]
            
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    async def get_audit_statistics(
        self,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get audit statistics for the specified period.
        
        Args:
            days_back: Number of days to include in statistics
            
        Returns:
            Audit statistics dictionary
        """
        try:
            start_time = datetime.utcnow() - timedelta(days=days_back)
            events = await self.query_events(start_time=start_time, limit=10000)
            
            # Calculate statistics
            stats = {
                "total_events": len(events),
                "period_days": days_back,
                "events_by_category": {},
                "events_by_level": {},
                "events_by_day": {},
                "unique_users": set(),
                "unique_patients": set()
            }
            
            for event in events:
                # Category statistics
                category = event.category.value
                stats["events_by_category"][category] = stats["events_by_category"].get(category, 0) + 1
                
                # Level statistics
                level = event.level.value
                stats["events_by_level"][level] = stats["events_by_level"].get(level, 0) + 1
                
                # Daily statistics
                day = event.timestamp.date().isoformat()
                stats["events_by_day"][day] = stats["events_by_day"].get(day, 0) + 1
                
                # Unique users and patients
                if event.user_id:
                    stats["unique_users"].add(event.user_id)
                if event.patient_id_hash:
                    stats["unique_patients"].add(event.patient_id_hash)
            
            # Convert sets to counts
            stats["unique_users"] = len(stats["unique_users"])
            stats["unique_patients"] = len(stats["unique_patients"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_events(self) -> int:
        """
        Clean up audit events older than retention period.
        
        Returns:
            Number of events cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            cleanup_count = 0
            
            # Get all audit files
            audit_files = list(self.audit_directory.glob(f"audit_{self.hospital_id}_*.jsonl"))
            
            for file_path in audit_files:
                # Check file modification time
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    # Remove old file
                    file_path.unlink()
                    cleanup_count += 1
                    logger.info(f"Removed old audit file: {file_path}")
            
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old audit events: {e}")
            return 0
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return f"audit_{self.hospital_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    def _sanitize_additional_data(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Sanitize additional data to remove potential PHI."""
        if not data:
            return {}
        
        sanitized = {}
        
        # Fields that might contain PHI
        phi_fields = [
            "patient_name", "patient_address", "patient_phone", "patient_email",
            "ssn", "medical_record_number", "date_of_birth", "full_name"
        ]
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Skip fields that might contain PHI
            if any(phi_field in key_lower for phi_field in phi_fields):
                sanitized[key] = "[REDACTED_FOR_PRIVACY]"
            elif isinstance(value, str) and len(value) > 500:
                # Truncate very long strings that might contain PHI
                sanitized[key] = value[:200] + "...[TRUNCATED]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _initialize_audit_file(self) -> None:
        """Initialize current audit file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{self.hospital_id}_{timestamp}.jsonl"
        self.current_audit_file = self.audit_directory / filename
    
    async def _write_event_to_file(self, event: AuditEvent) -> None:
        """Write audit event to file."""
        try:
            with open(self.current_audit_file, 'a') as f:
                json.dump(event.to_dict(), f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit event to file: {e}")
            raise
    
    async def _check_file_rotation(self) -> None:
        """Check if audit file rotation is needed."""
        try:
            if self.current_audit_file.exists():
                file_size = self.current_audit_file.stat().st_size
                
                if file_size > self.max_file_size_bytes:
                    logger.info(f"Rotating audit file (size: {file_size} bytes)")
                    self._initialize_audit_file()
                    
        except Exception as e:
            logger.error(f"Failed to check file rotation: {e}")
    
    async def _search_audit_files(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        category: Optional[AuditCategory] = None,
        level: Optional[AuditLevel] = None,
        user_id: Optional[str] = None,
        patient_id_hash: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Search audit files for matching events."""
        try:
            matching_events = []
            
            # Get all audit files
            audit_files = sorted(
                self.audit_directory.glob(f"audit_{self.hospital_id}_*.jsonl"),
                reverse=True  # Most recent first
            )
            
            for file_path in audit_files:
                if len(matching_events) >= limit:
                    break
                
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            if len(matching_events) >= limit:
                                break
                            
                            try:
                                event_data = json.loads(line.strip())
                                event = AuditEvent.from_dict(event_data)
                                
                                # Apply filters
                                if start_time and event.timestamp < start_time:
                                    continue
                                if end_time and event.timestamp > end_time:
                                    continue
                                if category and event.category != category:
                                    continue
                                if level and event.level != level:
                                    continue
                                if user_id and event.user_id != user_id:
                                    continue
                                if patient_id_hash and event.patient_id_hash != patient_id_hash:
                                    continue
                                
                                matching_events.append(event)
                                
                            except json.JSONDecodeError:
                                continue  # Skip malformed lines
                                
                except Exception as e:
                    logger.warning(f"Failed to read audit file {file_path}: {e}")
                    continue
            
            return matching_events
            
        except Exception as e:
            logger.error(f"Failed to search audit files: {e}")
            return []