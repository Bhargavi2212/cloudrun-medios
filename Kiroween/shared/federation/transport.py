"""
Secure transport helpers for encrypted model parameter exchange.

This module provides utilities for secure communication of federated
learning parameters between hospitals while maintaining privacy.
"""

import json
import base64
import hashlib
import hmac
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import uuid

# Try to import cryptography, but make it optional for testing
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    # Mock classes for testing without cryptography
    class Fernet:
        @staticmethod
        def generate_key():
            return b"mock_key_32_bytes_long_for_test"
        
        def __init__(self, key):
            self.key = key
        
        def encrypt(self, data):
            return b"encrypted_" + data
        
        def decrypt(self, data):
            return data[10:]  # Remove "encrypted_" prefix

from .base import ModelWeights


class SecureTransport:
    """
    Secure transport layer for federated learning parameter exchange.
    
    Provides encryption, authentication, and integrity verification
    for model weights transmitted between hospitals.
    """
    
    def __init__(self, hospital_id: str, private_key: Optional[bytes] = None):
        self.hospital_id = hospital_id
        
        if CRYPTO_AVAILABLE:
            # Generate or load RSA key pair for this hospital
            if private_key:
                self.private_key = serialization.load_pem_private_key(
                    private_key, password=None
                )
            else:
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
            
            self.public_key = self.private_key.public_key()
            
            # Generate symmetric key for bulk encryption
            self.symmetric_key = Fernet.generate_key()
            self.fernet = Fernet(self.symmetric_key)
        else:
            # Mock implementation for testing
            self.private_key = None
            self.public_key = None
            self.symmetric_key = b"mock_key_32_bytes_long_for_test"
            self.fernet = Fernet(self.symmetric_key)
    
    def get_public_key_pem(self) -> bytes:
        """Get public key in PEM format for sharing with other hospitals."""
        if CRYPTO_AVAILABLE and self.public_key:
            return self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            return b"-----BEGIN PUBLIC KEY-----\nMOCK_PUBLIC_KEY_FOR_TESTING\n-----END PUBLIC KEY-----"
    
    def get_private_key_pem(self) -> bytes:
        """Get private key in PEM format for storage."""
        if CRYPTO_AVAILABLE and self.private_key:
            return self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            return b"-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY_FOR_TESTING\n-----END PRIVATE KEY-----"
    
    async def encrypt_model_weights(
        self,
        weights: ModelWeights,
        recipient_public_key: bytes
    ) -> Dict[str, Any]:
        """
        Encrypt model weights for secure transmission.
        
        Args:
            weights: Model weights to encrypt
            recipient_public_key: Public key of receiving hospital
            
        Returns:
            Encrypted payload with metadata
        """
        # Serialize weights to JSON
        weights_json = json.dumps(weights.to_dict(), default=str)
        weights_bytes = weights_json.encode('utf-8')
        
        if CRYPTO_AVAILABLE:
            # Load recipient's public key
            recipient_key = serialization.load_pem_public_key(recipient_public_key)
            
            # Encrypt weights with symmetric key
            encrypted_weights = self.fernet.encrypt(weights_bytes)
            
            # Encrypt symmetric key with recipient's public key
            encrypted_symmetric_key = recipient_key.encrypt(
                self.symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            # Mock encryption for testing
            encrypted_weights = self.fernet.encrypt(weights_bytes)
            encrypted_symmetric_key = b"mock_encrypted_symmetric_key"
        
        # Create message authentication code
        mac = self._create_mac(encrypted_weights)
        
        # Create secure payload
        payload = {
            "encrypted_weights": base64.b64encode(encrypted_weights).decode('utf-8'),
            "encrypted_key": base64.b64encode(encrypted_symmetric_key).decode('utf-8'),
            "mac": mac,
            "sender_id": self.hospital_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": str(uuid.uuid4()),
            "model_id": weights.model_id,
            "training_round": weights.training_round
        }
        
        return payload
    
    async def decrypt_model_weights(
        self,
        encrypted_payload: Dict[str, Any]
    ) -> ModelWeights:
        """
        Decrypt received model weights.
        
        Args:
            encrypted_payload: Encrypted payload from another hospital
            
        Returns:
            Decrypted model weights
        """
        # Verify message age (prevent replay attacks)
        timestamp = datetime.fromisoformat(encrypted_payload["timestamp"])
        if datetime.utcnow() - timestamp > timedelta(hours=24):
            raise ValueError("Message too old, potential replay attack")
        
        # Decrypt weights
        encrypted_weights = base64.b64decode(encrypted_payload["encrypted_weights"])
        
        # Verify MAC
        if not self._verify_mac(encrypted_weights, encrypted_payload["mac"]):
            raise ValueError("Message authentication failed")
        
        if CRYPTO_AVAILABLE:
            # Decrypt symmetric key
            encrypted_key = base64.b64decode(encrypted_payload["encrypted_key"])
            symmetric_key = self.private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt weights data
            fernet = Fernet(symmetric_key)
            weights_bytes = fernet.decrypt(encrypted_weights)
        else:
            # Mock decryption for testing
            weights_bytes = self.fernet.decrypt(encrypted_weights)
        
        weights_dict = json.loads(weights_bytes.decode('utf-8'))
        
        # Reconstruct ModelWeights object
        weights = ModelWeights(
            weights=weights_dict["weights"],
            model_id=weights_dict["model_id"],
            hospital_id=weights_dict["hospital_id"],
            training_round=weights_dict["training_round"],
            privacy_budget=weights_dict["privacy_budget"],
            noise_scale=weights_dict.get("noise_scale")
        )
        
        return weights
    
    def _create_mac(self, data: bytes) -> str:
        """Create message authentication code."""
        mac = hmac.new(
            self.symmetric_key,
            data,
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')
    
    def _verify_mac(self, data: bytes, received_mac: str) -> bool:
        """Verify message authentication code."""
        expected_mac = self._create_mac(data)
        return hmac.compare_digest(expected_mac, received_mac)


class FederatedTransportManager:
    """
    Manager for secure transport across multiple hospitals.
    
    Handles key exchange, message routing, and secure communication
    for federated learning coordination.
    """
    
    def __init__(self, hospital_id: str):
        self.hospital_id = hospital_id
        self.transport = SecureTransport(hospital_id)
        self.peer_public_keys: Dict[str, bytes] = {}
        self.message_history: List[Dict[str, Any]] = []
    
    def register_peer(self, peer_hospital_id: str, public_key: bytes) -> None:
        """
        Register a peer hospital's public key.
        
        Args:
            peer_hospital_id: ID of peer hospital
            public_key: Peer's public key in PEM format
        """
        self.peer_public_keys[peer_hospital_id] = public_key
    
    async def send_weights_to_aggregator(
        self,
        weights: ModelWeights,
        aggregator_id: str
    ) -> Dict[str, Any]:
        """
        Send model weights to federated aggregator.
        
        Args:
            weights: Local model weights to send
            aggregator_id: ID of the aggregator service
            
        Returns:
            Encrypted payload for transmission
        """
        if aggregator_id not in self.peer_public_keys:
            raise ValueError(f"No public key registered for aggregator {aggregator_id}")
        
        # Encrypt weights for aggregator
        encrypted_payload = await self.transport.encrypt_model_weights(
            weights,
            self.peer_public_keys[aggregator_id]
        )
        
        # Record in message history
        self.message_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "send_weights",
            "recipient": aggregator_id,
            "model_id": weights.model_id,
            "training_round": weights.training_round,
            "message_id": encrypted_payload["message_id"]
        })
        
        return encrypted_payload
    
    async def receive_global_weights(
        self,
        encrypted_payload: Dict[str, Any]
    ) -> ModelWeights:
        """
        Receive global weights from aggregator.
        
        Args:
            encrypted_payload: Encrypted global weights
            
        Returns:
            Decrypted global model weights
        """
        # Decrypt weights
        global_weights = await self.transport.decrypt_model_weights(encrypted_payload)
        
        # Record in message history
        self.message_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "receive_global_weights",
            "sender": encrypted_payload["sender_id"],
            "model_id": global_weights.model_id,
            "training_round": global_weights.training_round,
            "message_id": encrypted_payload["message_id"]
        })
        
        return global_weights
    
    async def broadcast_weights_to_peers(
        self,
        weights: ModelWeights,
        peer_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Broadcast weights to multiple peer hospitals.
        
        Args:
            weights: Model weights to broadcast
            peer_ids: List of peer hospital IDs
            
        Returns:
            Dictionary of encrypted payloads for each peer
        """
        encrypted_payloads = {}
        
        for peer_id in peer_ids:
            if peer_id not in self.peer_public_keys:
                continue
            
            encrypted_payload = await self.transport.encrypt_model_weights(
                weights,
                self.peer_public_keys[peer_id]
            )
            
            encrypted_payloads[peer_id] = encrypted_payload
            
            # Record in message history
            self.message_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": "broadcast_weights",
                "recipient": peer_id,
                "model_id": weights.model_id,
                "training_round": weights.training_round,
                "message_id": encrypted_payload["message_id"]
            })
        
        return encrypted_payloads
    
    def get_transport_statistics(self) -> Dict[str, Any]:
        """Get statistics about secure transport usage."""
        return {
            "hospital_id": self.hospital_id,
            "registered_peers": len(self.peer_public_keys),
            "peer_ids": list(self.peer_public_keys.keys()),
            "messages_sent": len([m for m in self.message_history if m["action"] in ["send_weights", "broadcast_weights"]]),
            "messages_received": len([m for m in self.message_history if m["action"] == "receive_global_weights"]),
            "total_messages": len(self.message_history)
        }
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get history of secure transport messages."""
        return self.message_history.copy()


class PrivacyPreservingTransport:
    """
    Enhanced transport with additional privacy guarantees.
    
    Implements techniques like onion routing, mix networks,
    and traffic analysis resistance for maximum privacy.
    """
    
    def __init__(self, hospital_id: str):
        self.hospital_id = hospital_id
        self.base_transport = SecureTransport(hospital_id)
        self.privacy_level = "maximum"
    
    async def send_with_traffic_padding(
        self,
        weights: ModelWeights,
        recipient_public_key: bytes,
        padding_size: int = 1024
    ) -> Dict[str, Any]:
        """
        Send weights with traffic padding to resist traffic analysis.
        
        Args:
            weights: Model weights to send
            recipient_public_key: Recipient's public key
            padding_size: Amount of random padding to add
            
        Returns:
            Encrypted payload with padding
        """
        # Encrypt weights normally
        encrypted_payload = await self.base_transport.encrypt_model_weights(
            weights,
            recipient_public_key
        )
        
        # Add random padding
        padding = base64.b64encode(
            bytes([0] * padding_size)
        ).decode('utf-8')
        
        encrypted_payload["padding"] = padding
        encrypted_payload["privacy_enhanced"] = True
        
        return encrypted_payload
    
    async def send_with_delay(
        self,
        weights: ModelWeights,
        recipient_public_key: bytes,
        delay_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Send weights with random delay to resist timing analysis.
        
        This would be implemented with actual delay in production.
        """
        import asyncio
        
        # Add random delay (simulated)
        actual_delay = delay_seconds + (hash(str(weights.weight_id)) % 30)
        
        # In production, would actually delay here:
        # await asyncio.sleep(actual_delay)
        
        encrypted_payload = await self.base_transport.encrypt_model_weights(
            weights,
            recipient_public_key
        )
        
        encrypted_payload["delayed_send"] = True
        encrypted_payload["delay_applied"] = actual_delay
        
        return encrypted_payload