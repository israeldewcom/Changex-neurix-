# app/models/user.py - ENHANCED WITH ADMIN APPROVAL FEATURES
from datetime import datetime
from app import db, login_manager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import json
import uuid

class User(UserMixin, db.Model):
    """Enhanced User model with premium approval tracking"""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256))
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    company = db.Column(db.String(128))
    phone = db.Column(db.String(20))
    country = db.Column(db.String(64))
    timezone = db.Column(db.String(64), default='UTC')
    
    # Account status
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    is_administrator = db.Column(db.Boolean, default=False)
    is_moderator = db.Column(db.Boolean, default=False)
    is_support = db.Column(db.Boolean, default=False)
    
    # Premium features
    has_premium_access = db.Column(db.Boolean, default=False)
    premium_approved = db.Column(db.Boolean, default=False)
    premium_approved_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    premium_approved_at = db.Column(db.DateTime)
    premium_rejection_reason = db.Column(db.Text)
    
    # Subscription info
    subscription_tier = db.Column(db.String(32), default='free')
    subscription_status = db.Column(db.String(32), default='active')
    subscription_id = db.Column(db.String(128))
    subscription_updated_at = db.Column(db.DateTime)
    
    # API access
    api_key = db.Column(db.String(128), unique=True, default=lambda: str(uuid.uuid4()))
    api_secret = db.Column(db.String(256), default=lambda: str(uuid.uuid4()))
    api_rate_limit = db.Column(db.Integer, default=1000)
    
    # Usage tracking
    total_requests = db.Column(db.Integer, default=0)
    total_images_generated = db.Column(db.Integer, default=0)
    total_videos_generated = db.Column(db.Integer, default=0)
    total_audio_processed = db.Column(db.Integer, default=0)
    storage_used = db.Column(db.BigInteger, default=0)  # in bytes
    
    # Security
    two_factor_enabled = db.Column(db.Boolean, default=False)
    two_factor_secret = db.Column(db.String(32))
    last_password_change = db.Column(db.DateTime)
    failed_login_attempts = db.Column(db.Integer, default=0)
    account_locked_until = db.Column(db.DateTime)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_seen = db.Column(db.DateTime)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    subscription = db.relationship('Subscription', backref='user', uselist=False, lazy=True)
    transactions = db.relationship('Transaction', backref='user', lazy='dynamic')
    api_keys = db.relationship('APIKey', backref='user', lazy='dynamic')
    generated_images = db.relationship('GeneratedImage', backref='user', lazy='dynamic')
    generated_videos = db.relationship('GeneratedVideo', backref='user', lazy='dynamic')
    audio_files = db.relationship('AudioFile', backref='user', lazy='dynamic')
    iot_devices = db.relationship('IoTDevice', backref='user', lazy='dynamic')
    learning_sessions = db.relationship('SelfLearningSession', backref='user', lazy='dynamic')
    affiliate = db.relationship('Affiliate', backref='user', uselist=False, lazy=True)
    admin_actions = db.relationship('AdminAction', backref='admin', lazy='dynamic', foreign_keys='AdminAction.admin_id')
    
    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        if not self.api_key:
            self.api_key = str(uuid.uuid4())
        if not self.api_secret:
            self.api_secret = str(uuid.uuid4())
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        self.last_password_change = datetime.utcnow()
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def generate_api_key(self):
        self.api_key = str(uuid.uuid4())
        self.api_secret = str(uuid.uuid4())
        return self.api_key, self.api_secret
    
    def approve_premium_access(self, admin_user, notes=None):
        """Approve premium access for user"""
        self.has_premium_access = True
        self.premium_approved = True
        self.premium_approved_by = admin_user.id
        self.premium_approved_at = datetime.utcnow()
        
        # Create admin action log
        action = AdminAction(
            admin_id=admin_user.id,
            action_type='premium_approval',
            target_type='user',
            target_id=self.id,
            details=f"Premium access approved. Notes: {notes or 'No notes'}",
            ip_address=request.remote_addr if request else None
        )
        db.session.add(action)
        db.session.commit()
        
        # Send notification
        from app.utils.notifications import send_premium_approval_notification
        send_premium_approval_notification(self)
        
        return True
    
    def reject_premium_access(self, admin_user, reason):
        """Reject premium access request"""
        self.premium_approved = False
        self.premium_rejection_reason = reason
        
        # Create admin action log
        action = AdminAction(
            admin_id=admin_user.id,
            action_type='premium_rejection',
            target_type='user',
            target_id=self.id,
            details=f"Premium access rejected. Reason: {reason}",
            ip_address=request.remote_addr if request else None
        )
        db.session.add(action)
        db.session.commit()
        
        # Send notification
        from app.utils.notifications import send_premium_rejection_notification
        send_premium_rejection_notification(self, reason)
        
        return True
    
    def get_usage_stats(self):
        """Get user usage statistics"""
        return {
            'total_requests': self.total_requests,
            'total_images': self.total_images_generated,
            'total_videos': self.total_videos_generated,
            'total_audio': self.total_audio_processed,
            'storage_used': self.storage_used,
            'storage_used_human': self._bytes_to_human(self.storage_used)
        }
    
    def _bytes_to_human(self, bytes):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} PB"
    
    def to_dict(self, include_sensitive=False):
        """Convert user to dictionary"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'company': self.company,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'is_administrator': self.is_administrator,
            'has_premium_access': self.has_premium_access,
            'premium_approved': self.premium_approved,
            'premium_approved_at': self.premium_approved_at.isoformat() if self.premium_approved_at else None,
            'subscription_tier': self.subscription_tier,
            'subscription_status': self.subscription_status,
            'total_requests': self.total_requests,
            'storage_used': self.storage_used,
            'storage_used_human': self._bytes_to_human(self.storage_used),
            'created_at': self.created_at.isoformat(),
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
        
        if include_sensitive:
            data.update({
                'api_key': self.api_key,
                'two_factor_enabled': self.two_factor_enabled,
                'premium_rejection_reason': self.premium_rejection_reason
            })
        
        return data
    
    def __repr__(self):
        return f'<User {self.username}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class AdminAction(db.Model):
    """Track all admin actions for audit trail"""
    
    __tablename__ = 'admin_actions'
    
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action_type = db.Column(db.String(64), nullable=False)  # premium_approval, transaction_approval, withdrawal_approval, etc.
    target_type = db.Column(db.String(64), nullable=False)  # user, transaction, withdrawal, etc.
    target_id = db.Column(db.Integer, nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(256))
    action_timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Indexes
    __table_args__ = (
        db.Index('idx_admin_actions_composite', 'admin_id', 'action_type', 'action_timestamp'),
        db.Index('idx_admin_actions_target', 'target_type', 'target_id'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'admin_id': self.admin_id,
            'admin_username': self.admin.username if self.admin else None,
            'action_type': self.action_type,
            'target_type': self.target_type,
            'target_id': self.target_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'action_timestamp': self.action_timestamp.isoformat()
        }
    
    def __repr__(self):
        return f'<AdminAction {self.action_type} by User {self.admin_id}>'

class Subscription(db.Model):
    """Enhanced Subscription model with approval tracking"""
    
    __tablename__ = 'subscriptions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    tier = db.Column(db.String(32), default='free')
    status = db.Column(db.String(32), default='active')  # active, pending, canceled, expired, requires_approval
    amount = db.Column(db.Numeric(10, 2), default=0)
    currency = db.Column(db.String(3), default='USD')
    
    # Payment info
    payment_method = db.Column(db.String(32))  # stripe, flutterwave, manual
    payment_id = db.Column(db.String(128))
    payment_status = db.Column(db.String(32))  # pending, completed, failed, requires_approval
    
    # Approval tracking
    requires_approval = db.Column(db.Boolean, default=False)
    approved_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    approved_at = db.Column(db.DateTime)
    rejection_reason = db.Column(db.Text)
    
    # Billing cycle
    billing_cycle = db.Column(db.String(32), default='monthly')  # monthly, yearly, lifetime
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime)
    next_billing_date = db.Column(db.DateTime)
    auto_renew = db.Column(db.Boolean, default=True)
    
    # Trial info
    is_trial = db.Column(db.Boolean, default=False)
    trial_ends_at = db.Column(db.DateTime)
    
    # Usage limits
    max_requests = db.Column(db.Integer, default=1000)
    max_images = db.Column(db.Integer, default=10)
    max_videos = db.Column(db.Integer, default=5)
    max_audio = db.Column(db.Integer, default=20)
    max_storage = db.Column(db.BigInteger, default=1073741824)  # 1GB in bytes
    
    # Current usage
    used_requests = db.Column(db.Integer, default=0)
    used_images = db.Column(db.Integer, default=0)
    used_videos = db.Column(db.Integer, default=0)
    used_audio = db.Column(db.Integer, default=0)
    used_storage = db.Column(db.BigInteger, default=0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    canceled_at = db.Column(db.DateTime)
    
    def approve(self, admin_user, notes=None):
        """Approve subscription"""
        self.status = 'active'
        self.payment_status = 'completed'
        self.requires_approval = False
        self.approved_by = admin_user.id
        self.approved_at = datetime.utcnow
        
        # Update user premium access
        self.user.has_premium_access = True
        self.user.premium_approved = True
        self.user.premium_approved_by = admin_user.id
        self.user.premium_approved_at = datetime.utcnow
        
        # Log action
        action = AdminAction(
            admin_id=admin_user.id,
            action_type='subscription_approval',
            target_type='subscription',
            target_id=self.id,
            details=f"Subscription approved. Tier: {self.tier}, Amount: {self.amount}. Notes: {notes or 'No notes'}",
            ip_address=request.remote_addr if request else None
        )
        db.session.add(action)
        db.session.commit()
        
        # Send notification
        from app.utils.notifications import send_subscription_approval_notification
        send_subscription_approval_notification(self.user)
        
        return True
    
    def reject(self, admin_user, reason):
        """Reject subscription"""
        self.status = 'canceled'
        self.payment_status = 'failed'
        self.rejection_reason = reason
        
        # Log action
        action = AdminAction(
            admin_id=admin_user.id,
            action_type='subscription_rejection',
            target_type='subscription',
            target_id=self.id,
            details=f"Subscription rejected. Reason: {reason}",
            ip_address=request.remote_addr if request else None
        )
        db.session.add(action)
        db.session.commit()
        
        # Send notification
        from app.utils.notifications import send_subscription_rejection_notification
        send_subscription_rejection_notification(self.user, reason)
        
        return True
    
    def update_usage(self, requests=0, images=0, videos=0, audio=0, storage=0):
        """Update usage counters"""
        self.used_requests += requests
        self.used_images += images
        self.used_videos += videos
        self.used_audio += audio
        self.used_storage += storage
        
        # Check if limits exceeded
        if (self.used_requests > self.max_requests or
            self.used_images > self.max_images or
            self.used_videos > self.max_videos or
            self.used_audio > self.max_audio or
            self.used_storage > self.max_storage):
            self.status = 'limit_exceeded'
        
        db.session.commit()
        return True
    
    def get_usage_percentage(self):
        """Get usage percentages"""
        return {
            'requests': min(100, (self.used_requests / self.max_requests * 100) if self.max_requests > 0 else 0),
            'images': min(100, (self.used_images / self.max_images * 100) if self.max_images > 0 else 0),
            'videos': min(100, (self.used_videos / self.max_videos * 100) if self.max_videos > 0 else 0),
            'audio': min(100, (self.used_audio / self.max_audio * 100) if self.max_audio > 0 else 0),
            'storage': min(100, (self.used_storage / self.max_storage * 100) if self.max_storage > 0 else 0)
        }
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'tier': self.tier,
            'status': self.status,
            'amount': float(self.amount) if self.amount else 0,
            'currency': self.currency,
            'payment_method': self.payment_method,
            'payment_status': self.payment_status,
            'requires_approval': self.requires_approval,
            'billing_cycle': self.billing_cycle,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'next_billing_date': self.next_billing_date.isoformat() if self.next_billing_date else None,
            'auto_renew': self.auto_renew,
            'is_trial': self.is_trial,
            'trial_ends_at': self.trial_ends_at.isoformat() if self.trial_ends_at else None,
            'usage': self.get_usage_percentage(),
            'limits': {
                'max_requests': self.max_requests,
                'max_images': self.max_images,
                'max_videos': self.max_videos,
                'max_audio': self.max_audio,
                'max_storage': self.max_storage
            },
            'used': {
                'requests': self.used_requests,
                'images': self.used_images,
                'videos': self.used_videos,
                'audio': self.used_audio,
                'storage': self.used_storage
            },
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<Subscription {self.tier} for User {self.user_id}>'
