"""
Cryptographic service for DOL.

This service handles cryptographic signing, verification, and encryption
for portable patient profiles while maintaining hospital anonymity.
"""

import hashlib
import hmac
import base64
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets

from ..schemas import ClinicalTimelineEntry

logger = logging.getLogger(__name__)


class CryptographicService:
    """Service for cryptographic operations on patient profiles."""
    
    def __init__(
        self,
        hospital_id: str,
        private_key_path: Optional[str] = None,
        public_key_path: Optional[str] = None
    ):
        self.hospital_id = hospital_id
        self.private_key_path = private_key_path
        self.public_key_path = public_key_path
        
        # Initialize cryptographic keys
        self.private_key = None
        self.public_key = None
        
        # For demo purposes, generate keys in memory
        # In production, these would be loaded from secure storage
        self._initialize_keys()
        
        logger.info(f"Initialized CryptographicService for hospital {hospital_id}")
    
    def _initialize_keys(self):
        """Initialize cryptographic keys for signing and verification."""
        try:
            # Generate RSA key pair for demo
            # In production, load from secure key management system
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.public_key = self.private_key.public_key()
            
            logger.info("Generated RSA key pair for cryptographic operations")
            
        except Exception as e:
            logger.error(f"Failed to initialize cryptographic keys: {e}")
            raise
    
    async def sign_and_encrypt_profile(
        self,
        profile_data: Dict[str, Any],
        export_format: str = "json"
    ) -> Dict[str, str]:
        """
        Sign and encrypt a portable profile for secure transport.
        
        Args:
            profile_data: Profile data to sign and encrypt
            export_format: Export format (json, fhir, etc.)
            
        Returns:
            Dictionary with encrypted_data and signature
        """
        try:
            logger.info("Signing and encrypting portable profile")
            
            # Serialize profile data
            profile_json = json.dumps(profile_data, default=str, sort_keys=True)
            
            # Create cryptographic signature
            signature = await self._create_profile_signature(profile_json)
            
            # Encrypt profile data
            encrypted_data = await self._encrypt_profile_data(profile_json)
            
            return {
                "encrypted_data": encrypted_data,
                "signature": signature,
                "algorithm": "RSA-SHA256",
                "hospital_fingerprint": await self._get_hospital_fingerprint()
            }
            
        except Exception as e:
            logger.error(f"Failed to sign and encrypt profile: {e}")
            raise
    
    async def decrypt_and_verify_profile(
        self,
        encrypted_data: str,
        signature: str
    ) -> Optional[str]:
        """
        Decrypt and verify a portable profile.
        
        Args:
            encrypted_data: Encrypted profile data
            signature: Cryptographic signature
            
        Returns:
            Decrypted profile data if verification succeeds, None otherwise
        """
        try:
            logger.info("Decrypting and verifying portable profile")
            
            # Decrypt profile data
            decrypted_data = await self._decrypt_profile_data(encrypted_data)
            
            if not decrypted_data:
                logger.error("Failed to decrypt profile data")
                return None
            
            # Verify signature
            signature_valid = await self._verify_profile_signature(decrypted_data, signature)
            
            if not signature_valid:
                logger.error("Profile signature verification failed")
                return None
            
            logger.info("Profile decryption and verification successful")
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Failed to decrypt and verify profile: {e}")
            return None
    
    async def verify_profile_signature(
        self,
        profile_data: str,
        signature: str
    ) -> bool:
        """
        Verify the cryptographic signature of a profile.
        
        Args:
            profile_data: Profile data to verify
            signature: Cryptographic signature
            
        Returns:
            True if signature is valid
        """
        return await self._verify_profile_signature(profile_data, signature)
    
    async def sign_timeline_entry(self, entry: ClinicalTimelineEntry) -> str:
        """
        Create cryptographic signature for a timeline entry.
        
        Args:
            entry: Timeline entry to sign
            
        Returns:
            Cryptographic signature
        """
        try:
            # Create canonical representation of timeline entry
            entry_data = {
                "entry_id": entry.entry_id,
                "patient_id": entry.patient_id,
                "timestamp": entry.timestamp.isoformat(),
                "event_type": entry.event_type,
                "clinical_summary": entry.clinical_summary,
                "structured_data": entry.structured_data
            }
            
            entry_json = json.dumps(entry_data, sort_keys=True)
            signature = await self._create_profile_signature(entry_json)
            
            logger.info(f"Created signature for timeline entry {entry.entry_id}")
            return signature
            
        except Exception as e:
            logger.error(f"Failed to sign timeline entry: {e}")
            raise
    
    async def verify_timeline_entry_signature(self, entry: Dict[str, Any]) -> bool:
        """
        Verify the signature of a timeline entry.
        
        Args:
            entry: Timeline entry with signature
            
        Returns:
            True if signature is valid
        """
        try:
            if "cryptographic_signature" not in entry:
                return False
            
            # Reconstruct entry data for verification
            entry_data = {
                "entry_id": entry["entry_id"],
                "patient_id": entry["patient_id"],
                "timestamp": entry["timestamp"],
                "event_type": entry["event_type"],
                "clinical_summary": entry["clinical_summary"],
                "structured_data": entry["structured_data"]
            }
            
            entry_json = json.dumps(entry_data, sort_keys=True)
            return await self._verify_profile_signature(
                entry_json,
                entry["cryptographic_signature"]
            )
            
        except Exception as e:
            logger.error(f"Failed to verify timeline entry signature: {e}")
            return False
    
    async def sign_model_parameters(
        self,
        model_parameters: Dict[str, Any],
        model_type: str,
        training_round: int
    ) -> str:
        """
        Sign model parameters for federated learning.
        
        Args:
            model_parameters: Model parameters to sign
            model_type: Type of model
            training_round: Training round number
            
        Returns:
            Cryptographic signature
        """
        try:
            # Create canonical representation
            parameter_data = {
                "model_type": model_type,
                "training_round": training_round,
                "parameters": model_parameters,
                "timestamp": datetime.utcnow().isoformat(),
                "hospital_fingerprint": await self._get_hospital_fingerprint()
            }
            
            parameter_json = json.dumps(parameter_data, sort_keys=True, default=str)
            signature = await self._create_profile_signature(parameter_json)
            
            logger.info(f"Created signature for {model_type} model parameters")
            return signature
            
        except Exception as e:
            logger.error(f"Failed to sign model parameters: {e}")
            raise
    
    async def verify_global_model_signature(
        self,
        model_parameters: Dict[str, Any],
        signature: str
    ) -> bool:
        """
        Verify signature of global model update.
        
        Args:
            model_parameters: Global model parameters
            signature: Coordinator signature
            
        Returns:
            True if signature is valid
        """
        try:
            # TODO: Implement actual global model signature verification
            # This would verify the federated coordinator's signature
            logger.info("TODO: Verify global model signature from coordinator")
            
            # For demo purposes, always return True
            # In production, this would verify against coordinator's public key
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify global model signature: {e}")
            return False
    
    async def get_verification_details(self, signature: str) -> Dict[str, Any]:
        """
        Get details about a cryptographic signature.
        
        Args:
            signature: Signature to analyze
            
        Returns:
            Signature details
        """
        try:
            # Decode signature to extract metadata
            # This is a simplified implementation
            return {
                "algorithm": "RSA-SHA256",
                "hospital_fingerprint": await self._get_hospital_fingerprint(),
                "signature_length": len(signature),
                "created_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get verification details: {e}")
            return {}
    
    # Private helper methods
    
    async def _create_profile_signature(self, data: str) -> str:
        """Create RSA signature for data."""
        try:
            # Create SHA-256 hash of data
            data_bytes = data.encode('utf-8')
            
            # Sign with private key
            signature = self.private_key.sign(
                data_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Encode signature as base64
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to create signature: {e}")
            raise
    
    async def _verify_profile_signature(self, data: str, signature: str) -> bool:
        """Verify RSA signature for data."""
        try:
            # Decode signature from base64
            signature_bytes = base64.b64decode(signature.encode('utf-8'))
            data_bytes = data.encode('utf-8')
            
            # Verify signature with public key
            self.public_key.verify(
                signature_bytes,
                data_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False
    
    async def _encrypt_profile_data(self, data: str) -> str:
        """Encrypt profile data using AES."""
        try:
            # Generate random AES key and IV
            aes_key = secrets.token_bytes(32)  # 256-bit key
            iv = secrets.token_bytes(16)  # 128-bit IV
            
            # Encrypt data with AES
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Pad data to AES block size
            data_bytes = data.encode('utf-8')
            padding_length = 16 - (len(data_bytes) % 16)
            padded_data = data_bytes + bytes([padding_length] * padding_length)
            
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Encrypt AES key with RSA public key
            encrypted_key = self.public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key, IV, and encrypted data
            combined = encrypted_key + iv + encrypted_data
            return base64.b64encode(combined).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to encrypt profile data: {e}")
            raise
    
    async def _decrypt_profile_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt profile data using AES."""
        try:
            # Decode from base64
            combined = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Extract components
            encrypted_key = combined[:256]  # RSA-2048 encrypted key
            iv = combined[256:272]  # 16-byte IV
            encrypted_content = combined[272:]  # Encrypted data
            
            # Decrypt AES key with RSA private key
            aes_key = self.private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt data with AES
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(encrypted_content) + decryptor.finalize()
            
            # Remove padding
            padding_length = padded_data[-1]
            data = padded_data[:-padding_length]
            
            return data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to decrypt profile data: {e}")
            return None
    
    async def _get_hospital_fingerprint(self) -> str:
        """Get hospital's public key fingerprint for identification."""
        try:
            # Get public key in PEM format
            public_key_pem = self.public_key.public_key_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Create SHA-256 hash of public key
            fingerprint = hashlib.sha256(public_key_pem).hexdigest()
            
            # Return first 16 characters for brevity
            return fingerprint[:16]
            
        except Exception as e:
            logger.error(f"Failed to get hospital fingerprint: {e}")
            return "unknown"