"""
Authentication middleware for DOL Service.

This middleware handles JWT/mTLS authentication for secure
hospital-to-hospital communication using the peer registry.
"""

import logging
import uuid
import jwt
import ssl
from datetime import datetime, timedelta
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..services.peer_registry import PeerRegistryService
from ..config import get_settings

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for DOL API endpoints."""
    
    def __init__(self, app):
        super().__init__(app)
        
        self.settings = get_settings()
        self.peer_registry = None  # Will be initialized on first request
        
        # Endpoints that don't require authentication
        self.public_endpoints = [
            "/health",
            "/",
            "/docs",
            "/openapi.json",
            "/redoc"
        ]
        
        # Endpoints that require peer hospital authentication
        self.peer_authenticated_endpoints = [
            "/api/federated/patient",
            "/api/timeline",
            "/api/model_update"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process authentication for incoming requests."""
        
        # Initialize peer registry if not already done
        if self.peer_registry is None:
            self.peer_registry = PeerRegistryService(self.settings.hospital_id)
        
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        
        try:
            # Authenticate request
            auth_result = await self._authenticate_request(request)
            
            if not auth_result["authenticated"]:
                logger.warning(f"Authentication failed for {request.url.path}: {auth_result.get('reason')}")
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Authentication required",
                        "message": auth_result.get("reason", "Valid authentication credentials required"),
                        "request_id": request_id
                    }
                )
            
            # Verify peer hospital authorization for sensitive endpoints
            if any(endpoint in request.url.path for endpoint in self.peer_authenticated_endpoints):
                peer_hospital_id = auth_result.get("hospital_id")
                
                if not await self.peer_registry.verify_peer(peer_hospital_id):
                    logger.warning(f"Untrusted peer hospital: {peer_hospital_id}")
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Unauthorized hospital",
                            "message": "Hospital not in trusted peer registry",
                            "request_id": request_id
                        }
                    )
                
                # Validate communication authorization
                communication_type = self._get_communication_type(request.url.path)
                if not await self.peer_registry.validate_peer_communication(
                    peer_hospital_id, communication_type, {}
                ):
                    logger.warning(f"Peer {peer_hospital_id} not authorized for {communication_type}")
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Operation not authorized",
                            "message": f"Hospital not authorized for {communication_type}",
                            "request_id": request_id
                        }
                    )
            
            # Add authentication context to request
            request.state.hospital_id = auth_result.get("hospital_id")
            request.state.authenticated = True
            request.state.auth_method = auth_result.get("method")
            request.state.peer_verified = auth_result.get("peer_verified", False)
            
            # Process request
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Hospital-ID"] = auth_result.get("hospital_id", "unknown")
            
            return response
            
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Authentication system error",
                    "message": "Internal authentication error",
                    "request_id": request_id
                }
            )
    
    async def _authenticate_request(self, request: Request) -> dict:
        """
        Authenticate incoming request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Authentication result dictionary
        """
        try:
            # Check for Authorization header
            auth_header = request.headers.get("Authorization")
            
            if not auth_header:
                # Check for mTLS client certificate
                client_cert = await self._get_client_certificate(request)
                if client_cert:
                    return await self._authenticate_mtls(client_cert, request)
                
                return {"authenticated": False, "reason": "No authentication provided"}
            
            # Handle Bearer token authentication
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                return await self._authenticate_jwt(token, request)
            
            # Handle API key authentication
            if auth_header.startswith("ApiKey "):
                api_key = auth_header[7:]
                return await self._authenticate_api_key(api_key, request)
            
            return {"authenticated": False, "reason": "Invalid authentication format"}
            
        except Exception as e:
            logger.error(f"Authentication check failed: {e}")
            return {"authenticated": False, "reason": f"Authentication error: {str(e)}"}
    
    async def _authenticate_jwt(self, token: str, request: Request) -> dict:
        """
        Authenticate JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Authentication result
        """
        try:
            # Decode JWT token
            try:
                payload = jwt.decode(
                    token,
                    self.settings.jwt_secret_key,
                    algorithms=[self.settings.jwt_algorithm]
                )
            except jwt.ExpiredSignatureError:
                return {"authenticated": False, "reason": "JWT token expired"}
            except jwt.InvalidTokenError:
                return {"authenticated": False, "reason": "Invalid JWT token"}
            
            # Extract hospital ID from token
            hospital_id = payload.get("hospital_id")
            if not hospital_id:
                return {"authenticated": False, "reason": "Hospital ID not found in token"}
            
            # Verify token claims
            issued_at = payload.get("iat")
            expires_at = payload.get("exp")
            
            if not issued_at or not expires_at:
                return {"authenticated": False, "reason": "Missing token timestamps"}
            
            # Check if token is not yet valid
            current_time = datetime.utcnow().timestamp()
            if issued_at > current_time:
                return {"authenticated": False, "reason": "Token not yet valid"}
            
            return {
                "authenticated": True,
                "method": "jwt",
                "hospital_id": hospital_id,
                "token_valid": True,
                "issued_at": issued_at,
                "expires_at": expires_at
            }
            
        except Exception as e:
            logger.error(f"JWT authentication failed: {e}")
            return {"authenticated": False, "reason": f"JWT validation error: {str(e)}"}
    
    async def _authenticate_api_key(self, api_key: str, request: Request) -> dict:
        """
        Authenticate API key using peer registry.
        
        Args:
            api_key: API key string
            request: FastAPI request object
            
        Returns:
            Authentication result
        """
        try:
            # Get peer registry from app state
            peer_registry = getattr(request.app.state, 'peer_registry', None)
            
            if not peer_registry:
                logger.warning("Peer registry not available for API key validation")
                # Fallback to basic validation
                if len(api_key) > 10:
                    return {
                        "authenticated": True,
                        "method": "api_key",
                        "hospital_id": "demo_hospital",
                        "key_valid": True
                    }
                return {"authenticated": False, "reason": "Invalid API key"}
            
            # Check API key against all trusted peers
            for hospital_id in peer_registry.trusted_peers.keys():
                if await peer_registry.validate_peer_api_key(hospital_id, api_key):
                    return {
                        "authenticated": True,
                        "method": "api_key",
                        "hospital_id": hospital_id,
                        "key_valid": True,
                        "peer_validated": True
                    }
            
            return {"authenticated": False, "reason": "API key not found in peer registry"}
            
        except Exception as e:
            logger.error(f"API key authentication failed: {e}")
            return {"authenticated": False, "reason": f"API key validation error: {str(e)}"}
    
    async def _authenticate_mtls(self, client_cert: dict, request: Request) -> dict:
        """
        Authenticate mTLS client certificate.
        
        Args:
            client_cert: Client certificate information
            
        Returns:
            Authentication result
        """
        try:
            # TODO: Implement actual mTLS certificate validation
            # This would include:
            # 1. Certificate chain validation
            # 2. Certificate expiration check
            # 3. Hospital identity extraction from certificate
            # 4. Certificate revocation check
            
            logger.info("TODO: Implement mTLS certificate validation")
            
            return {
                "authenticated": True,
                "method": "mtls",
                "hospital_id": client_cert.get("hospital_id", "demo_hospital"),
                "cert_valid": True
            }
            
        except Exception as e:
            logger.error(f"mTLS authentication failed: {e}")
            return {"authenticated": False, "reason": f"mTLS validation error: {str(e)}"}
    
    async def _get_client_certificate(self, request: Request) -> dict:
        """
        Extract client certificate from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client certificate information
        """
        try:
            # Extract client certificate from TLS connection
            # This would typically be handled by a reverse proxy (nginx, traefik)
            # and passed via headers like X-SSL-Client-Cert
            
            client_cert_header = request.headers.get("X-SSL-Client-Cert")
            if client_cert_header:
                # Parse certificate and extract hospital ID
                # In production, this would involve proper X.509 certificate parsing
                return {
                    "certificate": client_cert_header,
                    "hospital_id": self._extract_hospital_id_from_cert(client_cert_header),
                    "verified": True
                }
            
            # Check for other certificate headers
            client_dn = request.headers.get("X-SSL-Client-DN")
            if client_dn:
                hospital_id = self._extract_hospital_id_from_dn(client_dn)
                if hospital_id:
                    return {
                        "distinguished_name": client_dn,
                        "hospital_id": hospital_id,
                        "verified": True
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get client certificate: {e}")
            return None
    
    def _get_communication_type(self, path: str) -> str:
        """
        Determine communication type from request path.
        
        Args:
            path: Request path
            
        Returns:
            Communication type string
        """
        if "/api/federated/patient/import" in path:
            return "profile_import"
        elif "/api/federated/patient/export" in path:
            return "profile_export"
        elif "/api/timeline" in path:
            return "timeline_access"
        elif "/api/model_update/submit" in path:
            return "model_update"
        elif "/api/model_update/receive" in path:
            return "model_receive"
        else:
            return "general_api"
    
    def _extract_hospital_id_from_cert(self, certificate: str) -> Optional[str]:
        """
        Extract hospital ID from X.509 certificate.
        
        Args:
            certificate: Certificate string
            
        Returns:
            Hospital ID or None
        """
        try:
            # TODO: Implement actual certificate parsing
            # This would use cryptography library to parse X.509 certificate
            # and extract hospital ID from subject or SAN extension
            
            # For demo purposes, return a placeholder
            return "cert_hospital_001"
            
        except Exception as e:
            logger.error(f"Failed to extract hospital ID from certificate: {e}")
            return None
    
    def _extract_hospital_id_from_dn(self, distinguished_name: str) -> Optional[str]:
        """
        Extract hospital ID from certificate distinguished name.
        
        Args:
            distinguished_name: Certificate DN string
            
        Returns:
            Hospital ID or None
        """
        try:
            # Parse DN to extract hospital ID
            # Example DN: "CN=hospital_001,OU=Medical,O=Hospital Network,C=US"
            
            parts = distinguished_name.split(",")
            for part in parts:
                part = part.strip()
                if part.startswith("CN="):
                    cn = part[3:]  # Remove "CN=" prefix
                    if cn.startswith("hospital_"):
                        return cn
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract hospital ID from DN: {e}")
            return None