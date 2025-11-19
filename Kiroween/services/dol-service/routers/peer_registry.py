"""
Peer Registry API routes.

This module handles peer hospital registry management for secure
multi-hospital communication and federated learning coordination.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..schemas import (
    PeerRegistrationRequest,
    PeerRegistrationResponse,
    PeerListResponse,
    PeerStatusResponse,
    RegistryStatusResponse
)
from ..services.peer_registry import PeerRegistryService, PeerCapability, PeerStatus
from ..services.audit_storage import AuditStorageService, AuditCategory, AuditLevel
from ..dependencies import get_peer_registry, get_audit_storage, get_current_hospital_id

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/register", response_model=PeerRegistrationResponse)
async def register_peer_hospital(
    registration: PeerRegistrationRequest,
    background_tasks: BackgroundTasks,
    peer_registry: PeerRegistryService = Depends(get_peer_registry),
    audit_storage: AuditStorageService = Depends(get_audit_storage),
    current_hospital_id: str = Depends(get_current_hospital_id)
):
    """
    Register a new peer hospital in the registry.
    
    This endpoint allows hospitals to register with each other for
    secure communication and federated learning participation.
    """
    try:
        logger.info(f"Registering peer hospital: {registration.hospital_id}")
        
        # Validate registration request
        if registration.hospital_id == current_hospital_id:
            raise HTTPException(
                status_code=400,
                detail="Cannot register self as peer hospital"
            )
        
        # Register the peer
        success = await peer_registry.register_peer(
            hospital_id=registration.hospital_id,
            hospital_name=registration.hospital_name,
            public_key_fingerprint=registration.public_key_fingerprint,
            api_endpoint=registration.api_endpoint,
            capabilities=registration.capabilities,
            auto_approve=registration.auto_approve
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to register peer hospital"
            )
        
        # Log audit event
        background_tasks.add_task(
            audit_storage.log_event,
            category=AuditCategory.SYSTEM_ADMIN,
            level=AuditLevel.INFO,
            message=f"Registered peer hospital {registration.hospital_id}",
            additional_data={
                "peer_hospital_id": registration.hospital_id,
                "capabilities": registration.capabilities,
                "auto_approved": registration.auto_approve
            }
        )
        
        return PeerRegistrationResponse(
            success=True,
            hospital_id=registration.hospital_id,
            status="active" if registration.auto_approve else "pending",
            registration_timestamp=datetime.utcnow(),
            message="Peer hospital registered successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Peer registration failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Peer registration failed: {str(e)}"
        )


@router.post("/approve/{hospital_id}")
async def approve_peer_hospital(
    hospital_id: str,
    background_tasks: BackgroundTasks,
    peer_registry: PeerRegistryService = Depends(get_peer_registry),
    audit_storage: AuditStorageService = Depends(get_audit_storage)
):
    """
    Approve a pending peer hospital registration.
    
    This endpoint allows administrators to approve peer hospitals
    that are pending in the registry.
    """
    try:
        logger.info(f"Approving peer hospital: {hospital_id}")
        
        success = await peer_registry.approve_peer(hospital_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Peer hospital {hospital_id} not found or not pending"
            )
        
        # Log audit event
        background_tasks.add_task(
            audit_storage.log_event,
            category=AuditCategory.SYSTEM_ADMIN,
            level=AuditLevel.INFO,
            message=f"Approved peer hospital {hospital_id}",
            additional_data={"peer_hospital_id": hospital_id}
        )
        
        return {
            "success": True,
            "hospital_id": hospital_id,
            "status": "active",
            "approval_timestamp": datetime.utcnow(),
            "message": "Peer hospital approved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Peer approval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Peer approval failed: {str(e)}"
        )


@router.post("/suspend/{hospital_id}")
async def suspend_peer_hospital(
    hospital_id: str,
    reason: str,
    background_tasks: BackgroundTasks,
    peer_registry: PeerRegistryService = Depends(get_peer_registry),
    audit_storage: AuditStorageService = Depends(get_audit_storage)
):
    """
    Suspend a peer hospital.
    
    This endpoint allows administrators to suspend peer hospitals
    for security or compliance reasons.
    """
    try:
        logger.info(f"Suspending peer hospital: {hospital_id}")
        
        success = await peer_registry.suspend_peer(hospital_id, reason)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Peer hospital {hospital_id} not found"
            )
        
        # Log audit event
        background_tasks.add_task(
            audit_storage.log_event,
            category=AuditCategory.SECURITY_EVENT,
            level=AuditLevel.WARNING,
            message=f"Suspended peer hospital {hospital_id}: {reason}",
            additional_data={
                "peer_hospital_id": hospital_id,
                "suspension_reason": reason
            }
        )
        
        return {
            "success": True,
            "hospital_id": hospital_id,
            "status": "suspended",
            "suspension_timestamp": datetime.utcnow(),
            "reason": reason,
            "message": "Peer hospital suspended successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Peer suspension failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Peer suspension failed: {str(e)}"
        )


@router.get("/list", response_model=PeerListResponse)
async def list_peer_hospitals(
    status: Optional[str] = None,
    capability: Optional[str] = None,
    peer_registry: PeerRegistryService = Depends(get_peer_registry)
):
    """
    List peer hospitals in the registry.
    
    This endpoint returns a list of peer hospitals with optional
    filtering by status and capability.
    """
    try:
        logger.info("Listing peer hospitals")
        
        if status and capability:
            # Filter by both status and capability
            if status == "active":
                peers = await peer_registry.get_peers_with_capability(capability)
            else:
                # Get all peers and filter by status
                all_peers = list(peer_registry.peers.values())
                peers = [p for p in all_peers if p.status.value == status]
                
                # Further filter by capability
                if capability:
                    try:
                        cap_enum = PeerCapability(capability)
                        peers = [p for p in peers if cap_enum in p.capabilities]
                    except ValueError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid capability: {capability}"
                        )
        elif status:
            # Filter by status only
            if status == "active":
                peers = await peer_registry.get_active_peers()
            else:
                all_peers = list(peer_registry.peers.values())
                peers = [p for p in all_peers if p.status.value == status]
        elif capability:
            # Filter by capability only
            peers = await peer_registry.get_peers_with_capability(capability)
        else:
            # Return all peers
            peers = list(peer_registry.peers.values())
        
        # Convert to response format (without internal hospital names)
        peer_list = []
        for peer in peers:
            peer_info = {
                "hospital_id": peer.hospital_id,
                "public_key_fingerprint": peer.public_key_fingerprint,
                "api_endpoint": peer.api_endpoint,
                "capabilities": [cap.value for cap in peer.capabilities],
                "status": peer.status.value,
                "trust_score": peer.trust_score,
                "registered_at": peer.registered_at.isoformat(),
                "last_seen": peer.last_seen.isoformat() if peer.last_seen else None
            }
            peer_list.append(peer_info)
        
        return PeerListResponse(
            peers=peer_list,
            total_count=len(peer_list),
            filter_status=status,
            filter_capability=capability,
            query_timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list peer hospitals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list peer hospitals: {str(e)}"
        )


@router.get("/status/{hospital_id}", response_model=PeerStatusResponse)
async def get_peer_hospital_status(
    hospital_id: str,
    peer_registry: PeerRegistryService = Depends(get_peer_registry)
):
    """
    Get detailed status of a specific peer hospital.
    
    This endpoint returns comprehensive status information
    for a registered peer hospital.
    """
    try:
        logger.info(f"Getting status for peer hospital: {hospital_id}")
        
        peer = await peer_registry.get_peer(hospital_id)
        
        if not peer:
            raise HTTPException(
                status_code=404,
                detail=f"Peer hospital {hospital_id} not found"
            )
        
        return PeerStatusResponse(
            hospital_id=peer.hospital_id,
            status=peer.status.value,
            capabilities=[cap.value for cap in peer.capabilities],
            trust_score=peer.trust_score,
            registered_at=peer.registered_at,
            last_seen=peer.last_seen,
            last_communication=peer.last_communication,
            successful_communications=peer.successful_communications,
            failed_communications=peer.failed_communications,
            total_profiles_exchanged=peer.total_profiles_exchanged,
            is_trusted=await peer_registry.verify_peer_trust(hospital_id)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get peer status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get peer status: {str(e)}"
        )


@router.get("/registry/status", response_model=RegistryStatusResponse)
async def get_registry_status(
    peer_registry: PeerRegistryService = Depends(get_peer_registry)
):
    """
    Get overall registry status and statistics.
    
    This endpoint returns comprehensive information about
    the peer hospital registry.
    """
    try:
        logger.info("Getting registry status")
        
        status = await peer_registry.get_registry_status()
        
        return RegistryStatusResponse(
            hospital_id=status["hospital_id"],
            total_peers=status["total_peers"],
            active_peers=status["active_peers"],
            pending_peers=status["pending_peers"],
            suspended_peers=status["suspended_peers"],
            last_updated=datetime.fromisoformat(status["last_updated"]),
            registry_healthy=True  # TODO: Add health checks
        )
        
    except Exception as e:
        logger.error(f"Failed to get registry status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get registry status: {str(e)}"
        )


@router.post("/verify/{hospital_id}")
async def verify_peer_trust(
    hospital_id: str,
    peer_registry: PeerRegistryService = Depends(get_peer_registry)
):
    """
    Verify trust status of a peer hospital.
    
    This endpoint checks if a peer hospital is trusted
    for secure communication.
    """
    try:
        logger.info(f"Verifying trust for peer hospital: {hospital_id}")
        
        is_trusted = await peer_registry.verify_peer_trust(hospital_id)
        peer = await peer_registry.get_peer(hospital_id)
        
        if not peer:
            raise HTTPException(
                status_code=404,
                detail=f"Peer hospital {hospital_id} not found"
            )
        
        return {
            "hospital_id": hospital_id,
            "is_trusted": is_trusted,
            "trust_score": peer.trust_score,
            "status": peer.status.value,
            "verification_timestamp": datetime.utcnow(),
            "trust_factors": {
                "status_active": peer.status == PeerStatus.ACTIVE,
                "trust_score_sufficient": peer.trust_score >= peer_registry.min_trust_score,
                "communication_success_rate": (
                    peer.successful_communications / 
                    max(1, peer.successful_communications + peer.failed_communications)
                ),
                "recently_seen": peer.last_seen is not None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify peer trust: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify peer trust: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_inactive_peers(
    background_tasks: BackgroundTasks,
    peer_registry: PeerRegistryService = Depends(get_peer_registry),
    audit_storage: AuditStorageService = Depends(get_audit_storage)
):
    """
    Clean up inactive peer hospitals.
    
    This endpoint removes peer hospitals that haven't been
    seen for an extended period.
    """
    try:
        logger.info("Cleaning up inactive peers")
        
        cleanup_count = await peer_registry.cleanup_inactive_peers()
        
        # Log audit event
        background_tasks.add_task(
            audit_storage.log_event,
            category=AuditCategory.SYSTEM_ADMIN,
            level=AuditLevel.INFO,
            message=f"Cleaned up {cleanup_count} inactive peer hospitals",
            additional_data={"cleanup_count": cleanup_count}
        )
        
        return {
            "success": True,
            "cleanup_count": cleanup_count,
            "cleanup_timestamp": datetime.utcnow(),
            "message": f"Cleaned up {cleanup_count} inactive peer hospitals"
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup inactive peers: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup inactive peers: {str(e)}"
        )


@router.get("/export")
async def export_peer_list(
    include_internal_names: bool = False,
    peer_registry: PeerRegistryService = Depends(get_peer_registry)
):
    """
    Export peer list for sharing with other hospitals.
    
    This endpoint exports the peer registry in a format
    suitable for sharing with other hospitals.
    """
    try:
        logger.info("Exporting peer list")
        
        peer_list = await peer_registry.export_peer_list(
            include_internal_names=include_internal_names
        )
        
        return {
            "success": True,
            "peer_count": len(peer_list),
            "peers": peer_list,
            "export_timestamp": datetime.utcnow(),
            "includes_internal_names": include_internal_names
        }
        
    except Exception as e:
        logger.error(f"Failed to export peer list: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export peer list: {str(e)}"
        )