# app/models/affiliate.py - ENHANCED WITH WITHDRAWAL APPROVAL
from datetime import datetime
from app import db
import json
import uuid

class Affiliate(db.Model):
    """Enhanced Affiliate model with withdrawal tracking"""
    
    __tablename__ = 'affiliates'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    affiliate_code = db.Column(db.String(32), unique=True, nullable=False, index=True)
    
    # Commission settings
    commission_rate = db.Column(db.Float, default=0.15)  # 15% default
    custom_rate = db.Column(db.Float)
    min_payout = db.Column(db.Numeric(10, 2), default=50.00)
    
    # Earnings tracking
    balance = db.Column(db.Numeric(10, 2), default=0)
    total_earned = db.Column(db.Numeric(10, 2), default=0)
    total_withdrawn = db.Column(db.Numeric(10, 2), default=0)
    pending_withdrawals = db.Column(db.Numeric(10, 2), default=0)
    
    # Performance metrics
    total_referrals = db.Column(db.Integer, default=0)
    active_referrals = db.Column(db.Integer, default=0)
    conversion_rate = db.Column(db.Float, default=0)
    
    # Payment info
    payment_method = db.Column(db.String(32))  # bank_transfer, paypal, stripe
    payment_details = db.Column(db.JSON)  # Bank account, PayPal email, etc.
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    verified_at = db.Column(db.DateTime)
    verified_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    commissions = db.relationship('Commission', backref='affiliate', lazy='dynamic')
    referrals = db.relationship('AffiliateReferral', backref='affiliate', lazy='dynamic')
    withdrawal_requests = db.relationship('WithdrawalRequest', backref='affiliate', lazy='dynamic')
    
    def __init__(self, **kwargs):
        super(Affiliate, self).__init__(**kwargs)
        if not self.affiliate_code:
            self.affiliate_code = self._generate_affiliate_code()
    
    def _generate_affiliate_code(self):
        """Generate unique affiliate code"""
        import random
        import string
        
        while True:
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            if not Affiliate.query.filter_by(affiliate_code=code).first():
                return code
    
    def request_withdrawal(self, amount, payment_method=None, details=None):
        """Request withdrawal with approval requirement"""
        
        if amount < self.min_payout:
            raise ValueError(f"Minimum withdrawal amount is {self.min_payout}")
        
        if amount > self.balance:
            raise ValueError("Insufficient balance")
        
        # Create withdrawal request
        withdrawal = WithdrawalRequest(
            affiliate_id=self.id,
            amount=amount,
            currency='USD',
            payment_method=payment_method or self.payment_method,
            payment_details=details or self.payment_details,
            status='pending'  # Requires admin approval
        )
        
        # Reserve balance
        self.balance -= amount
        self.pending_withdrawals += amount
        
        db.session.add(withdrawal)
        db.session.commit()
        
        return withdrawal
    
    def get_performance_stats(self, days=30):
        """Get affiliate performance statistics"""
        from datetime import timedelta
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get commissions in period
        period_commissions = self.commissions.filter(
            Commission.created_at >= start_date,
            Commission.status == 'paid'
        ).all()
        
        total_commission = sum(float(c.amount) for c in period_commissions)
        
        # Get referrals in period
        period_referrals = self.referrals.filter(
            AffiliateReferral.created_at >= start_date
        ).all()
        
        total_referrals = len(period_referrals)
        converted_referrals = len([r for r in period_referrals if r.converted])
        
        return {
            'period_days': days,
            'total_commission': total_commission,
            'total_referrals': total_referrals,
            'converted_referrals': converted_referrals,
            'conversion_rate': (converted_referrals / total_referrals * 100) if total_referrals > 0 else 0,
            'average_commission': total_commission / len(period_commissions) if period_commissions else 0
        }
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'affiliate_code': self.affiliate_code,
            'commission_rate': self.commission_rate,
            'balance': float(self.balance) if self.balance else 0,
            'total_earned': float(self.total_earned) if self.total_earned else 0,
            'total_withdrawn': float(self.total_withdrawn) if self.total_withdrawn else 0,
            'pending_withdrawals': float(self.pending_withdrawals) if self.pending_withdrawals else 0,
            'total_referrals': self.total_referrals,
            'active_referrals': self.active_referrals,
            'conversion_rate': self.conversion_rate,
            'payment_method': self.payment_method,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'min_payout': float(self.min_payout) if self.min_payout else 0,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<Affiliate {self.affiliate_code} - User {self.user_id}>'

class WithdrawalRequest(db.Model):
    """Withdrawal request model with approval system"""
    
    __tablename__ = 'withdrawal_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(128), unique=True, nullable=False, index=True)
    affiliate_id = db.Column(db.Integer, db.ForeignKey('affiliates.id'), nullable=False, index=True)
    
    # Withdrawal details
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    currency = db.Column(db.String(3), default='USD')
    payment_method = db.Column(db.String(32), nullable=False)
    payment_details = db.Column(db.JSON)  # Bank account, transaction ID, etc.
    
    # Status and approval
    status = db.Column(db.String(32), default='pending')  # pending, approved, rejected, paid
    requires_approval = db.Column(db.Boolean, default=True)
    approved_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    approved_at = db.Column(db.DateTime)
    rejection_reason = db.Column(db.Text)
    
    # Processing
    processed_at = db.Column(db.DateTime)
    processed_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    payment_proof = db.Column(db.String(512))  # URL to payment proof
    notes = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        db.Index('idx_withdrawal_status', 'status'),
        db.Index('idx_withdrawal_affiliate_status', 'affiliate_id', 'status'),
        db.Index('idx_withdrawal_created', 'created_at'),
    )
    
    def __init__(self, **kwargs):
        super(WithdrawalRequest, self).__init__(**kwargs)
        if not self.request_id:
            self.request_id = f"wd_{uuid.uuid4().hex[:16]}"
        
        # Check if approval is required
        if current_app.config.get('WITHDRAWAL_APPROVAL_REQUIRED', True):
            self.requires_approval = True
            self.status = 'pending'
    
    def approve(self, admin_user, notes=None):
        """Approve withdrawal request"""
        if self.status != 'pending':
            return False
        
        self.status = 'approved'
        self.approved_by = admin_user.id
        self.approved_at = datetime.utcnow()
        self.processed_at = datetime.utcnow()
        self.processed_by = admin_user.id
        self.notes = notes
        
        # Update affiliate pending withdrawals
        self.affiliate.pending_withdrawals -= self.amount
        
        db.session.commit()
        
        # Create admin action log
        from app.models.user import AdminAction
        action = AdminAction(
            admin_id=admin_user.id,
            action_type='withdrawal_approval',
            target_type='withdrawal',
            target_id=self.id,
            details=f"Withdrawal approved. Amount: {self.amount} {self.currency}. Notes: {notes or 'No notes'}",
            ip_address=request.remote_addr if request else None
        )
        db.session.add(action)
        db.session.commit()
        
        return True
    
    def reject(self, admin_user, reason):
        """Reject withdrawal request"""
        if self.status != 'pending':
            return False
        
        self.status = 'rejected'
        self.rejection_reason = reason
        self.processed_at = datetime.utcnow()
        self.processed_by = admin_user.id
        
        # Return amount to affiliate balance
        self.affiliate.balance += self.amount
        self.affiliate.pending_withdrawals -= self.amount
        
        db.session.commit()
        
        # Create admin action log
        from app.models.user import AdminAction
        action = AdminAction(
            admin_id=admin_user.id,
            action_type='withdrawal_rejection',
            target_type='withdrawal',
            target_id=self.id,
            details=f"Withdrawal rejected. Reason: {reason}",
            ip_address=request.remote_addr if request else None
        )
        db.session.add(action)
        db.session.commit()
        
        return True
    
    def mark_as_paid(self, admin_user, payment_proof=None, notes=None):
        """Mark withdrawal as paid"""
        if self.status != 'approved':
            return False
        
        self.status = 'paid'
        self.payment_proof = payment_proof
        self.notes = notes
        self.processed_at = datetime.utcnow()
        self.processed_by = admin_user.id
        
        # Update affiliate total withdrawn
        self.affiliate.total_withdrawn += self.amount
        
        db.session.commit()
        
        # Create admin action log
        from app.models.user import AdminAction
        action = AdminAction(
            admin_id=admin_user.id,
            action_type='withdrawal_paid',
            target_type='withdrawal',
            target_id=self.id,
            details=f"Withdrawal marked as paid. Amount: {self.amount} {self.currency}",
            ip_address=request.remote_addr if request else None
        )
        db.session.add(action)
        db.session.commit()
        
        return True
    
    def to_dict(self):
        return {
            'id': self.id,
            'request_id': self.request_id,
            'affiliate_id': self.affiliate_id,
            'affiliate_code': self.affiliate.affiliate_code if self.affiliate else None,
            'amount': float(self.amount) if self.amount else 0,
            'currency': self.currency,
            'payment_method': self.payment_method,
            'status': self.status,
            'requires_approval': self.requires_approval,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'rejection_reason': self.rejection_reason,
            'payment_proof': self.payment_proof,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<WithdrawalRequest {self.request_id} - {self.amount} {self.currency}>'
