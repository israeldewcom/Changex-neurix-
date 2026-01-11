# app/utils/notifications.py - ENHANCED NOTIFICATION SYSTEM
from flask import current_app, render_template
from flask_mail import Message
from app import mail, db
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def send_email_notification(recipient, subject, template, **kwargs):
    """Send email notification with template"""
    try:
        # Render template
        html_body = render_template(f'emails/{template}.html', **kwargs)
        text_body = render_template(f'emails/{template}.txt', **kwargs)
        
        # Create message
        msg = Message(
            subject=subject,
            recipients=[recipient],
            html=html_body,
            body=text_body,
            sender=current_app.config['MAIL_DEFAULT_SENDER']
        )
        
        # Send email
        mail.send(msg)
        
        logger.info(f"Email sent to {recipient}", subject=subject)
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email to {recipient}", error=str(e), exc_info=True)
        return False

def send_premium_approval_notification(user):
    """Send premium approval notification"""
    subject = f"🎉 Premium Access Approved - ChangeX Neurix"
    template = 'premium_approval'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_premium_rejection_notification(user, reason):
    """Send premium rejection notification"""
    subject = f"⚠️ Premium Access Request Update - ChangeX Neurix"
    template = 'premium_rejection'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        reason=reason,
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_transaction_approval_notification(transaction):
    """Send transaction approval notification"""
    user = transaction.user
    subject = f"✅ Payment Approved - ChangeX Neurix"
    template = 'transaction_approval'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        transaction=transaction,
        amount=f"{transaction.amount} {transaction.currency}",
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_transaction_rejection_notification(transaction, reason):
    """Send transaction rejection notification"""
    user = transaction.user
    subject = f"❌ Payment Declined - ChangeX Neurix"
    template = 'transaction_rejection'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        transaction=transaction,
        reason=reason,
        amount=f"{transaction.amount} {transaction.currency}",
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_withdrawal_approval_notification(withdrawal):
    """Send withdrawal approval notification"""
    user = withdrawal.affiliate.user
    subject = f"✅ Withdrawal Approved - ChangeX Neurix"
    template = 'withdrawal_approval'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        withdrawal=withdrawal,
        amount=f"{withdrawal.amount} {withdrawal.currency}",
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_withdrawal_rejection_notification(withdrawal, reason):
    """Send withdrawal rejection notification"""
    user = withdrawal.affiliate.user
    subject = f"❌ Withdrawal Declined - ChangeX Neurix"
    template = 'withdrawal_rejection'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        withdrawal=withdrawal,
        reason=reason,
        amount=f"{withdrawal.amount} {withdrawal.currency}",
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_subscription_approval_notification(user):
    """Send subscription approval notification"""
    subject = f"🎉 Subscription Activated - ChangeX Neurix"
    template = 'subscription_approval'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        tier=user.subscription_tier,
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_subscription_rejection_notification(user, reason):
    """Send subscription rejection notification"""
    subject = f"⚠️ Subscription Declined - ChangeX Neurix"
    template = 'subscription_rejection'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        reason=reason,
        tier=user.subscription_tier,
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_admin_approval_notification(admin_user, action_type, target_type, target_id):
    """Send notification to admin about pending approval"""
    subject = f"📋 Pending Approval Required - ChangeX Neurix"
    template = 'admin_approval_required'
    
    return send_email_notification(
        recipient=admin_user.email,
        subject=subject,
        template=template,
        admin=admin_user,
        action_type=action_type,
        target_type=target_type,
        target_id=target_id,
        date=datetime.utcnow().strftime('%B %d, %Y'),
        time=datetime.utcnow().strftime('%H:%M UTC')
    )

def send_welcome_email(user):
    """Send welcome email to new user"""
    subject = f"👋 Welcome to ChangeX Neurix!"
    template = 'welcome'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_password_reset_email(user, reset_token):
    """Send password reset email"""
    subject = f"🔑 Password Reset Request - ChangeX Neurix"
    template = 'password_reset'
    
    reset_link = f"{current_app.config['APP_URL']}/auth/reset-password/{reset_token}"
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        reset_link=reset_link,
        expiry_hours=24,
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_usage_limit_warning(user, resource_type, usage_percentage):
    """Send usage limit warning email"""
    subject = f"⚠️ Usage Limit Warning - ChangeX Neurix"
    template = 'usage_limit_warning'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        resource_type=resource_type,
        usage_percentage=usage_percentage,
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

def send_monthly_report(user, report_data):
    """Send monthly usage report"""
    subject = f"📊 Your Monthly Report - ChangeX Neurix"
    template = 'monthly_report'
    
    return send_email_notification(
        recipient=user.email,
        subject=subject,
        template=template,
        user=user,
        report=report_data,
        month=datetime.utcnow().strftime('%B %Y'),
        date=datetime.utcnow().strftime('%B %d, %Y')
    )

# In-app notifications system
class NotificationSystem:
    """In-app notification system"""
    
    @staticmethod
    def create_notification(user_id, notification_type, title, message, data=None, priority='normal'):
        """Create in-app notification"""
        from app.models.notification import Notification
        
        notification = Notification(
            user_id=user_id,
            notification_type=notification_type,
            title=title,
            message=message,
            data=data or {},
            priority=priority,
            is_read=False
        )
        
        db.session.add(notification)
        db.session.commit()
        
        # Emit WebSocket event
        from app import socketio
        socketio.emit('new_notification', {
            'notification': notification.to_dict(),
            'user_id': user_id
        }, room=f'user_{user_id}')
        
        return notification
    
    @staticmethod
    def send_premium_approved_notification(user):
        """Send premium approved notification"""
        return NotificationSystem.create_notification(
            user_id=user.id,
            notification_type='premium_approved',
            title='🎉 Premium Access Approved',
            message='Your premium access has been approved. You now have access to all premium features.',
            priority='high'
        )
    
    @staticmethod
    def send_transaction_approved_notification(user, transaction):
        """Send transaction approved notification"""
        return NotificationSystem.create_notification(
            user_id=user.id,
            notification_type='transaction_approved',
            title='✅ Payment Approved',
            message=f'Your payment of {transaction.amount} {transaction.currency} has been approved.',
            data={'transaction_id': transaction.id},
            priority='high'
        )
    
    @staticmethod
    def send_withdrawal_approved_notification(user, withdrawal):
        """Send withdrawal approved notification"""
        return NotificationSystem.create_notification(
            user_id=user.id,
            notification_type='withdrawal_approved',
            title='✅ Withdrawal Approved',
            message=f'Your withdrawal request of {withdrawal.amount} {withdrawal.currency} has been approved.',
            data={'withdrawal_id': withdrawal.id},
            priority='high'
        )
    
    @staticmethod
    def send_admin_approval_required_notification(admin_users, action_type, target_type, target_id):
        """Send admin approval required notifications"""
        notifications = []
        for admin in admin_users:
            if admin.is_administrator:
                notification = NotificationSystem.create_notification(
                    user_id=admin.id,
                    notification_type='admin_approval_required',
                    title='📋 Approval Required',
                    message=f'{action_type.replace("_", " ").title()} requires your approval',
                    data={
                        'action_type': action_type,
                        'target_type': target_type,
                        'target_id': target_id
                    },
                    priority='urgent'
                )
                notifications.append(notification)
        
        return notifications
