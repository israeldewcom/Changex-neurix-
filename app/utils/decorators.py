# app/utils/decorators.py - ENHANCED WITH ROLE PERMISSIONS
from functools import wraps
from flask import jsonify, request, current_app
from flask_login import current_user
from app import limiter
import jwt
from datetime import datetime

def admin_required(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not current_user.is_administrator:
            return jsonify({'error': 'Admin privileges required'}), 403
        
        return f(*args, **kwargs)
    return decorated_function

def role_required(role):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return jsonify({'error': 'Authentication required'}), 401
            
            # Check role permissions
            roles_permissions = {
                'super_admin': current_user.is_administrator,
                'admin': current_user.is_administrator,
                'moderator': current_user.is_moderator or current_user.is_administrator,
                'support': current_user.is_support or current_user.is_moderator or current_user.is_administrator
            }
            
            if not roles_permissions.get(role, False):
                return jsonify({'error': f'{role.capitalize()} role required'}), 403
            
            # app/utils/decorators.py - CONTINUED
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def subscription_required(tier=None):
    """Decorator to require subscription tier"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return jsonify({'error': 'Authentication required'}), 401
            
            # Check if user has premium access
            if not current_user.has_premium_access:
                return jsonify({'error': 'Premium subscription required'}), 403
            
            # Check specific tier if specified
            if tier and current_user.subscription_tier != tier:
                return jsonify({'error': f'{tier.capitalize()} subscription required'}), 403
            
            # Check if subscription is active
            if current_user.subscription and current_user.subscription.status != 'active':
                return jsonify({'error': 'Subscription is not active'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def rate_limit_by_user(limit, period):
    """Decorator for user-specific rate limiting"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if current_user.is_authenticated:
                key = f"ratelimit:{current_user.id}:{request.endpoint}"
            else:
                key = f"ratelimit:anonymous:{request.remote_addr}:{request.endpoint}"
            
            # Check rate limit
            try:
                limiter.check(f"{limit}/{period}", key=key)
            except Exception as e:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'limit': limit,
                    'period': period,
                    'retry_after': getattr(e, 'retry_after', None)
                }), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_request(schema_class):
    """Decorator to validate request data with Marshmallow schema"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                schema = schema_class()
                
                # Parse request data
                if request.is_json:
                    data = request.get_json()
                elif request.form:
                    data = request.form.to_dict()
                else:
                    data = {}
                
                # Merge with query params for GET requests
                if request.method == 'GET':
                    data.update(request.args.to_dict())
                
                # Validate data
                validated_data = schema.load(data)
                
                # Add validated data to request object
                request.validated_data = validated_data
                
                return f(*args, **kwargs)
                
            except ValidationError as e:
                return jsonify({
                    'error': 'Validation failed',
                    'errors': e.messages
                }), 400
            except Exception as e:
                current_app.logger.error(f"Validation error: {e}")
                return jsonify({'error': 'Invalid request data'}), 400
        
        return decorated_function
    return decorator

def cache_response(timeout=300):
    """Decorator to cache responses"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate cache key
            cache_key = f"view_cache:{request.path}:{hash(frozenset(request.args.items()))}"
            
            # Check cache
            cached_response = current_app.cache.get(cache_key)
            if cached_response and not request.args.get('refresh'):
                return cached_response
            
            # Execute function
            response = f(*args, **kwargs)
            
            # Cache response if successful
            if response.status_code == 200:
                current_app.cache.set(cache_key, response, timeout=timeout)
            
            return response
        return decorated_function
    return decorator

def async_task(timeout=60):
    """Decorator to run function as async task"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract task parameters
            task_name = f.__name__
            task_args = args
            task_kwargs = kwargs
            
            # Get user context
            user_context = {
                'user_id': current_user.id if current_user.is_authenticated else None,
                'ip': request.remote_addr if request else None,
                'user_agent': request.user_agent.string if request else None
            }
            
            # Queue task
            from app import task_queue
            job = task_queue.enqueue(
                f,
                args=task_args,
                kwargs=task_kwargs,
                job_timeout=timeout,
                result_ttl=86400,  # Keep results for 24 hours
                failure_ttl=604800,  # Keep failed jobs for 7 days
                meta={'user_context': user_context}
            )
            
            # Return job ID
            return jsonify({
                'success': True,
                'job_id': job.get_id(),
                'status_url': f'/api/v2/tasks/{job.get_id()}/status',
                'message': f'Task {task_name} queued for processing'
            })
        
        return decorated_function
    return decorator

def require_2fa(f):
    """Decorator to require 2FA authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        
        if current_user.two_factor_enabled and not session.get('2fa_verified'):
            # Check for 2FA token in request
            token = request.headers.get('X-2FA-Token') or request.json.get('two_factor_token')
            if not token:
                return jsonify({
                    'error': '2FA required',
                    'message': 'Two-factor authentication is enabled for your account',
                    'requires_2fa': True
                }), 403
            
            # Verify 2FA token
            from app.utils.security import verify_2fa_token
            if not verify_2fa_token(current_user.two_factor_secret, token):
                return jsonify({'error': 'Invalid 2FA token'}), 403
            
            # Mark as verified for this session
            session['2fa_verified'] = True
        
        return f(*args, **kwargs)
    return decorated_function

def audit_log(action_type):
    """Decorator to log admin actions"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from app.models.user import AdminAction
            
            # Execute the function
            result = f(*args, **kwargs)
            
            # Log the action if user is admin
            if current_user.is_authenticated and current_user.is_administrator:
                try:
                    # Extract target info from result or request
                    target_id = None
                    target_type = None
                    
                    if hasattr(result, 'json') and result.json:
                        data = result.json
                        if isinstance(data, dict):
                            target_id = data.get('id') or data.get('user_id') or data.get('transaction_id')
                    
                    # Determine target type from endpoint
                    if 'user' in request.endpoint:
                        target_type = 'user'
                    elif 'transaction' in request.endpoint:
                        target_type = 'transaction'
                    elif 'withdrawal' in request.endpoint:
                        target_type = 'withdrawal'
                    elif 'subscription' in request.endpoint:
                        target_type = 'subscription'
                    
                    # Create audit log
                    action = AdminAction(
                        admin_id=current_user.id,
                        action_type=action_type,
                        target_type=target_type,
                        target_id=target_id,
                        details=f"Action: {action_type} on {target_type} {target_id}",
                        ip_address=request.remote_addr,
                        user_agent=request.user_agent.string
                    )
                    
                    db.session.add(action)
                    db.session.commit()
                    
                except Exception as e:
                    current_app.logger.error(f"Failed to create audit log: {e}")
            
            return result
        
        return decorated_function
    return decorator

def maintenance_mode(f):
    """Decorator to check maintenance mode"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_app.config.get('MAINTENANCE_MODE', False):
            # Allow admins during maintenance
            if not current_user.is_authenticated or not current_user.is_administrator:
                return jsonify({
                    'error': 'Maintenance mode',
                    'message': 'The system is currently undergoing maintenance. Please try again later.',
                    'estimated_recovery': current_app.config.get('MAINTENANCE_ETA')
                }), 503
        
        return f(*args, **kwargs)
    return decorated_function
