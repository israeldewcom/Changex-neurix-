# app/utils/security.py - ENHANCED SECURITY UTILITIES
import hashlib
import hmac
import secrets
import string
import jwt
import bcrypt
from datetime import datetime, timedelta
from flask import current_app
import logging
import pyotp
import qrcode
import io
import base64

logger = logging.getLogger(__name__)

class SecurityManager:
    """Enhanced security manager"""
    
    @staticmethod
    def generate_secure_token(length=32):
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password):
        """Hash password with bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def verify_password(password, hashed_password):
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except:
            return False
    
    @staticmethod
    def generate_api_key():
        """Generate secure API key"""
        return f"cnx_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def generate_api_secret():
        """Generate secure API secret"""
        return secrets.token_urlsafe(64)
    
    @staticmethod
    def generate_jwt_token(user_id, expires_in=3600, additional_claims=None):
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow(),
            'iss': 'changex_neurix',
            'aud': 'changex_neurix_api'
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(
            payload,
            current_app.config['JWT_SECRET_KEY'],
            algorithm='HS256'
        )
    
    @staticmethod
    def verify_jwt_token(token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token,
                current_app.config['JWT_SECRET_KEY'],
                algorithms=['HS256'],
                audience='changex_neurix_api',
                issuer='changex_neurix'
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    @staticmethod
    def generate_2fa_secret():
        """Generate 2FA secret"""
        return pyotp.random_base32()
    
    @staticmethod
    def generate_2fa_qr_code(secret, username):
        """Generate QR code for 2FA setup"""
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=username,
            issuer_name="ChangeX Neurix"
        )
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'secret': secret,
            'qr_code': f"data:image/png;base64,{img_str}",
            'provisioning_uri': provisioning_uri
        }
    
    @staticmethod
    def verify_2fa_token(secret, token):
        """Verify 2FA token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    
    @staticmethod
    def generate_recovery_codes(count=10):
        """Generate recovery codes for 2FA"""
        codes = []
        for _ in range(count):
            # Generate 8-character codes with dashes
            code = '-'.join([
                secrets.token_hex(2),
                secrets.token_hex(2),
                secrets.token_hex(2)
            ]).upper()
            codes.append(code)
        
        # Hash codes for storage
        hashed_codes = [hashlib.sha256(code.encode()).hexdigest() for code in codes]
        
        return {
            'plain_codes': codes,
            'hashed_codes': hashed_codes
        }
    
    @staticmethod
    def verify_recovery_code(code, hashed_codes):
        """Verify recovery code"""
        code_hash = hashlib.sha256(code.upper().encode()).hexdigest()
        return code_hash in hashed_codes
    
    @staticmethod
    def encrypt_data(data, key=None):
        """Encrypt sensitive data"""
        from cryptography.fernet import Fernet
        import base64
        
        if not key:
            key = current_app.config['SESSION_ENCRYPTION_KEY']
        
        # Ensure key is 32 bytes
        key = hashlib.sha256(key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key)
        
        fernet = Fernet(fernet_key)
        
        if isinstance(data, dict):
            data = json.dumps(data)
        
        encrypted = fernet.encrypt(data.encode())
        return encrypted.decode()
    
    @staticmethod
    def decrypt_data(encrypted_data, key=None):
        """Decrypt sensitive data"""
        from cryptography.fernet import Fernet
        import base64
        
        if not key:
            key = current_app.config['SESSION_ENCRYPTION_KEY']
        
        # Ensure key is 32 bytes
        key = hashlib.sha256(key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key)
        
        fernet = Fernet(fernet_key)
        
        decrypted = fernet.decrypt(encrypted_data.encode())
        return decrypted.decode()
    
    @staticmethod
    def sanitize_input(input_string):
        """Sanitize user input to prevent XSS"""
        import html
        
        if not input_string:
            return input_string
        
        # HTML escape
        sanitized = html.escape(input_string)
        
        # Remove dangerous characters
        dangerous_patterns = [
            ('<script', ''),
            ('</script>', ''),
            ('javascript:', ''),
            ('onerror=', ''),
            ('onload=', ''),
            ('onclick=', ''),
            ('onmouseover=', ''),
            ('eval(', ''),
            ('alert(', ''),
        ]
        
        for pattern, replacement in dangerous_patterns:
            sanitized = sanitized.replace(pattern, replacement)
        
        return sanitized
    
    @staticmethod
    def validate_password_strength(password):
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > 128:
            return False, "Password must be less than 128 characters"
        
        # Check for uppercase
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        # Check for lowercase
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        # Check for digits
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"
        
        # Check for special characters
        special_chars = string.punctuation
        if not any(c in special_chars for c in password):
            return False, "Password must contain at least one special character"
        
        # Check for common passwords
        common_passwords = [
            'password', '123456', 'qwerty', 'admin', 'welcome',
            'password123', 'changeme', 'letmein', '12345678'
        ]
        
        if password.lower() in common_passwords:
            return False, "Password is too common"
        
        return True, "Password is strong"
    
    @staticmethod
    def generate_secure_filename(filename):
        """Generate secure filename"""
        import uuid
        import os
        
        # Get file extension
        _, ext = os.path.splitext(filename)
        
        # Generate secure filename
        secure_name = f"{uuid.uuid4().hex}{ext.lower()}"
        
        return secure_name
    
    @staticmethod
    def check_file_type(file_stream, allowed_extensions):
        """Check file type using magic numbers"""
        import magic
        
        # Read first 2048 bytes for magic number detection
        header = file_stream.read(2048)
        file_stream.seek(0)
        
        mime = magic.Magic(mime=True)
        file_mime = mime.from_buffer(header)
        
        # Map MIME types to extensions
        mime_to_ext = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'video/mp4': '.mp4',
            'video/webm': '.webm',
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'audio/ogg': '.ogg',
            'application/pdf': '.pdf',
            'text/plain': '.txt',
            'application/json': '.json'
        }
        
        if file_mime in mime_to_ext:
            detected_ext = mime_to_ext[file_mime]
            if detected_ext in allowed_extensions:
                return True
        
        return False
    
    @staticmethod
    def generate_hmac_signature(data, secret_key):
        """Generate HMAC signature for data verification"""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        
        return hmac.new(
            secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def verify_hmac_signature(data, signature, secret_key):
        """Verify HMAC signature"""
        expected_signature = SecurityManager.generate_hmac_signature(data, secret_key)
        return hmac.compare_digest(expected_signature, signature)
    
    @staticmethod
    def rate_limit_key(user_id, endpoint):
        """Generate rate limit key"""
        if user_id:
            return f"ratelimit:user:{user_id}:{endpoint}"
        else:
            return f"ratelimit:ip:{request.remote_addr}:{endpoint}"
    
    @staticmethod
    def check_rate_limit(key, limit, period):
        """Check rate limit"""
        from app import redis_client
        
        current = redis_client.get(key)
        if current and int(current) >= limit:
            return False
        
        # Increment counter
        redis_client.incr(key)
        redis_client.expire(key, period)
        
        return True
    
    @staticmethod
    def generate_csrf_token():
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def verify_csrf_token(token, session_token):
        """Verify CSRF token"""
        return hmac.compare_digest(token, session_token)
    
    @staticmethod
    def sanitize_sql_query(query):
        """Sanitize SQL query to prevent injection"""
        # Remove dangerous SQL keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER',
            'CREATE', 'EXEC', 'UNION', 'OR', 'AND'
        ]
        
        for keyword in dangerous_keywords:
            query = query.replace(keyword, '')
        
        return query
    
    @staticmethod
    def generate_secure_session_id():
        """Generate secure session ID"""
        return secrets.token_urlsafe(64)
    
    @staticmethod
    def validate_email(email):
        """Validate email address"""
        import re
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def generate_password_reset_token():
        """Generate password reset token"""
        return secrets.token_urlsafe(48)
