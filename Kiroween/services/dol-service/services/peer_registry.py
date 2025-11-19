"""
Peer Registry Service for DOL.

This service manages the registry of trusted peer hospitals for secure
multi-hospital communication and federated learning coordination.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class PeerStatus(str, Enum):
    """Status of peer hospital in registry."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class PeerCapability(str, Enum):
    """Capabilities supported by peer hospitals."""
    PROFILE_IMPORT = "profile_import"
    PROFILE_EXPORT = "profile_export"
    TIMELINE_SYNC = "timeline_sync"
    FEDERATED_LEARNING = "federated_learning"
    EMERGENCY_ACCESS = "emergency_access"


class PeerHospital:
    """Represents a peer hospital in the registry."""
    
    def __init__(
        self,
        hospital_id: str,
        hospital_name: str,
        public_key_fingerprint: str,
        api_endpoint: str,
        capabilities: List[PeerCapability],
        status: PeerStatus = PeerStatus.PENDING
    ):
        self.hospital_id = hospital_id
        self.hospital_name = hospital_name  # For internal use only, never exposed
        self.public_key_fingerprint = public_key_fingerprint
        self.api_endpoint = api_endpoint
        self.capabilities = capabilities
        self.status = status
        
        # Metadata
        self.registered_at = datetime.utcnow()
        self.last_seen = None
        self.last_communication = None
        self.trust_score = 1.0  # Initial trust score
        
        # Communication statistics
        self.successful_communications = 0
        self.failed_communications = 0
        self.total_profiles_exchanged = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hospital_id": self.hospital_id,
            "public_key_fingerprint": self.public_key_fingerprint,
            "api_endpoint": self.api_endpoint,
            "capabilities": [cap.value for cap in self.capabilities],
            "status": self.status.value,
            "registered_at": self.registered_at.isoformat(),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "trust_score": self.trust_score,
            "successful_communications": self.successful_communications,
            "failed_communications": self.failed_communications,
            "total_profiles_exchanged": self.total_profiles_exchanged
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PeerHospital':
        """Create from dictionary."""
        peer = cls(
            hospital_id=data["hospital_id"],
            hospital_name=data.get("hospital_name", "Unknown"),
            public_key_fingerprint=data["public_key_fingerprint"],
            api_endpoint=data["api_endpoint"],
            capabilities=[PeerCapability(cap) for cap in data["capabilities"]],
            status=PeerStatus(data["status"])
        )
        
        # Restore metadata
        peer.registered_at = datetime.fromisoformat(data["registered_at"])
        if data.get("last_seen"):
            peer.last_seen = datetime.fromisoformat(data["last_seen"])
        peer.trust_score = data.get("trust_score", 1.0)
        peer.successful_communications = data.get("successful_communications", 0)
        peer.failed_communications = data.get("failed_communications", 0)
        peer.total_profiles_exchanged = data.get("total_profiles_exchanged", 0)
        
        return peer


class PeerRegistryService:
    """Service for managing peer hospital registry."""
    
    def __init__(self, hospital_id: str, registry_file_path: Optional[str] = None):
        self.hospital_id = hospital_id
        self.registry_file_path = registry_file_path or f"./data/peer_registry_{hospital_id}.json"
        
        # In-memory registry
        self.peers: Dict[str, PeerHospital] = {}
        
        # Trust and security settings
        self.min_trust_score = 0.5
        self.max_failed_communications = 10
        self.peer_timeout_hours = 24
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"Initialized PeerRegistryService for hospital {hospital_id}")
    
    async def register_peer(
        self,
        hospital_id: str,
        hospital_name: str,
        public_key_fingerprint: str,
        api_endpoint: str,
        capabilities: List[str],
        auto_approve: bool = False
    ) -> bool:
        """
        Register a new peer hospital.
        
        Args:
            hospital_id: Unique hospital identifier
            hospital_name: Hospital name (internal use only)
            public_key_fingerprint: Public key fingerprint for verification
            api_endpoint: API endpoint for communication
            capabilities: List of supported capabilities
            auto_approve: Whether to auto-approve the peer
            
        Returns:
            True if registration successful
        """
        try:
            if hospital_id in self.peers:
                logger.warning(f"Hospital {hospital_id} already registered")
                return False
            
            # Validate capabilities
            peer_capabilities = []
            for cap in capabilities:
                try:
                    peer_capabilities.append(PeerCapability(cap))
                except ValueError:
                    logger.warning(f"Invalid capability: {cap}")
                    continue
            
            # Create peer hospital
            peer = PeerHospital(
                hospital_id=hospital_id,
                hospital_name=hospital_name,
                public_key_fingerprint=public_key_fingerprint,
                api_endpoint=api_endpoint,
                capabilities=peer_capabilities,
                status=PeerStatus.ACTIVE if auto_approve else PeerStatus.PENDING
            )
            
            # Add to registry
            self.peers[hospital_id] = peer
            
            # Save registry
            await self._save_registry()
            
            logger.info(f"Registered peer hospital {hospital_id} with status {peer.status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register peer {hospital_id}: {e}")
            return False
    
    async def approve_peer(self, hospital_id: str) -> bool:
        """
        Approve a pending peer hospital.
        
        Args:
            hospital_id: Hospital ID to approve
            
        Returns:
            True if approval successful
        """
        try:
            if hospital_id not in self.peers:
                logger.error(f"Hospital {hospital_id} not found in registry")
                return False
            
            peer = self.peers[hospital_id]
            if peer.status != PeerStatus.PENDING:
                logger.warning(f"Hospital {hospital_id} is not pending approval")
                return False
            
            peer.status = PeerStatus.ACTIVE
            await self._save_registry()
            
            logger.info(f"Approved peer hospital {hospital_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to approve peer {hospital_id}: {e}")
            return False
    
    async def suspend_peer(self, hospital_id: str, reason: str) -> bool:
        """
        Suspend a peer hospital.
        
        Args:
            hospital_id: Hospital ID to suspend
            reason: Reason for suspension
            
        Returns:
            True if suspension successful
        """
        try:
            if hospital_id not in self.peers:
                logger.error(f"Hospital {hospital_id} not found in registry")
                return False
            
            peer = self.peers[hospital_id]
            peer.status = PeerStatus.SUSPENDED
            await self._save_registry()
            
            logger.warning(f"Suspended peer hospital {hospital_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to suspend peer {hospital_id}: {e}")
            return False
    
    async def get_peer(self, hospital_id: str) -> Optional[PeerHospital]:
        """
        Get peer hospital information.
        
        Args:
            hospital_id: Hospital ID to retrieve
            
        Returns:
            PeerHospital object or None if not found
        """
        return self.peers.get(hospital_id)
    
    async def get_active_peers(self) -> List[PeerHospital]:
        """
        Get all active peer hospitals.
        
        Returns:
            List of active peer hospitals
        """
        return [
            peer for peer in self.peers.values()
            if peer.status == PeerStatus.ACTIVE
        ]
    
    async def get_peers_with_capability(self, capability: str) -> List[PeerHospital]:
        """
        Get peers that support a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of peers with the capability
        """
        try:
            cap_enum = PeerCapability(capability)
            return [
                peer for peer in self.peers.values()
                if peer.status == PeerStatus.ACTIVE and cap_enum in peer.capabilities
            ]
        except ValueError:
            logger.error(f"Invalid capability: {capability}")
            return []
    
    async def verify_peer_trust(self, hospital_id: str) -> bool:
        """
        Verify that a peer hospital is trusted for communication.
        
        Args:
            hospital_id: Hospital ID to verify
            
        Returns:
            True if peer is trusted
        """
        try:
            peer = self.peers.get(hospital_id)
            if not peer:
                logger.warning(f"Unknown peer hospital: {hospital_id}")
                return False
            
            # Check status
            if peer.status != PeerStatus.ACTIVE:
                logger.warning(f"Peer {hospital_id} is not active: {peer.status.value}")
                return False
            
            # Check trust score
            if peer.trust_score < self.min_trust_score:
                logger.warning(f"Peer {hospital_id} trust score too low: {peer.trust_score}")
                return False
            
            # Check failed communications
            if peer.failed_communications > self.max_failed_communications:
                logger.warning(f"Peer {hospital_id} has too many failed communications: {peer.failed_communications}")
                return False
            
            # Check if peer has been seen recently
            if peer.last_seen:
                time_since_seen = datetime.utcnow() - peer.last_seen
                if time_since_seen > timedelta(hours=self.peer_timeout_hours):
                    logger.warning(f"Peer {hospital_id} not seen for {time_since_seen}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify peer trust for {hospital_id}: {e}")
            return False
    
    async def record_communication(
        self,
        hospital_id: str,
        success: bool,
        operation_type: str = "unknown"
    ) -> None:
        """
        Record communication attempt with peer hospital.
        
        Args:
            hospital_id: Hospital ID
            success: Whether communication was successful
            operation_type: Type of operation performed
        """
        try:
            peer = self.peers.get(hospital_id)
            if not peer:
                logger.warning(f"Recording communication for unknown peer: {hospital_id}")
                return
            
            # Update communication statistics
            if success:
                peer.successful_communications += 1
                peer.last_communication = datetime.utcnow()
                peer.last_seen = datetime.utcnow()
                
                # Improve trust score for successful communications
                peer.trust_score = min(1.0, peer.trust_score + 0.01)
            else:
                peer.failed_communications += 1
                
                # Decrease trust score for failed communications
                peer.trust_score = max(0.0, peer.trust_score - 0.05)
            
            # Auto-suspend if too many failures
            if peer.failed_communications > self.max_failed_communications:
                await self.suspend_peer(hospital_id, "Too many failed communications")
            
            await self._save_registry()
            
            logger.info(f"Recorded {operation_type} communication with {hospital_id}: {'success' if success else 'failure'}")
            
        except Exception as e:
            logger.error(f"Failed to record communication for {hospital_id}: {e}")
    
    async def get_registry_status(self) -> Dict[str, Any]:
        """
        Get registry status information.
        
        Returns:
            Registry status dictionary
        """
        try:
            active_peers = len([p for p in self.peers.values() if p.status == PeerStatus.ACTIVE])
            pending_peers = len([p for p in self.peers.values() if p.status == PeerStatus.PENDING])
            suspended_peers = len([p for p in self.peers.values() if p.status == PeerStatus.SUSPENDED])
            
            return {
                "hospital_id": self.hospital_id,
                "total_peers": len(self.peers),
                "active_peers": active_peers,
                "pending_peers": pending_peers,
                "suspended_peers": suspended_peers,
                "registry_file": self.registry_file_path,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get registry status: {e}")
            return {"error": str(e)}
    
    async def cleanup_inactive_peers(self) -> int:
        """
        Clean up inactive peers that haven't been seen recently.
        
        Returns:
            Number of peers cleaned up
        """
        try:
            cleanup_count = 0
            current_time = datetime.utcnow()
            
            peers_to_remove = []
            for hospital_id, peer in self.peers.items():
                if peer.last_seen:
                    time_since_seen = current_time - peer.last_seen
                    if time_since_seen > timedelta(days=30):  # 30 days inactive
                        peers_to_remove.append(hospital_id)
                        cleanup_count += 1
            
            # Remove inactive peers
            for hospital_id in peers_to_remove:
                del self.peers[hospital_id]
                logger.info(f"Cleaned up inactive peer: {hospital_id}")
            
            if cleanup_count > 0:
                await self._save_registry()
            
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup inactive peers: {e}")
            return 0
    
    def _load_registry(self) -> None:
        """Load peer registry from file."""
        try:
            import os
            if os.path.exists(self.registry_file_path):
                with open(self.registry_file_path, 'r') as f:
                    data = json.load(f)
                
                for hospital_id, peer_data in data.get("peers", {}).items():
                    peer = PeerHospital.from_dict(peer_data)
                    self.peers[hospital_id] = peer
                
                logger.info(f"Loaded {len(self.peers)} peers from registry")
            else:
                logger.info("No existing registry file found, starting with empty registry")
                
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.peers = {}
    
    async def _save_registry(self) -> None:
        """Save peer registry to file."""
        try:
            import os
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.registry_file_path), exist_ok=True)
            
            # Prepare data for serialization
            registry_data = {
                "hospital_id": self.hospital_id,
                "last_updated": datetime.utcnow().isoformat(),
                "peers": {
                    hospital_id: peer.to_dict()
                    for hospital_id, peer in self.peers.items()
                }
            }
            
            # Save to file
            with open(self.registry_file_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.debug(f"Saved registry with {len(self.peers)} peers")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    async def export_peer_list(self, include_internal_names: bool = False) -> List[Dict[str, Any]]:
        """
        Export peer list for sharing with other hospitals.
        
        Args:
            include_internal_names: Whether to include hospital names (for internal use only)
            
        Returns:
            List of peer information dictionaries
        """
        try:
            peer_list = []
            
            for peer in self.peers.values():
                if peer.status == PeerStatus.ACTIVE:
                    peer_info = {
                        "hospital_id": peer.hospital_id,
                        "public_key_fingerprint": peer.public_key_fingerprint,
                        "api_endpoint": peer.api_endpoint,
                        "capabilities": [cap.value for cap in peer.capabilities],
                        "trust_score": peer.trust_score
                    }
                    
                    # Only include hospital name for internal use
                    if include_internal_names:
                        peer_info["hospital_name"] = peer.hospital_name
                    
                    peer_list.append(peer_info)
            
            return peer_list
            
        except Exception as e:
            logger.error(f"Failed to export peer list: {e}")
            return []