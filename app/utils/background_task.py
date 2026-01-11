# app/utils/background_tasks.py - ENHANCED BACKGROUND TASKS
import logging
from datetime import datetime, timedelta
from app import db, redis_client, task_queue, scheduler
from app.models.user import User, AdminAction
from app.models.payments import Transaction
from app.models.affiliate import WithdrawalRequest, Commission
from app.models.media import GeneratedImage, GeneratedVideo, AudioFile
from app.utils.notifications import NotificationSystem
import os
import shutil
import json

logger = logging.getLogger(__name__)

def cleanup_old_sessions():
    """Clean up old user sessions and temporary data"""
    try:
        # Clean old Flask sessions from Redis
        cutoff = datetime.utcnow() - timedelta(days=7)
        session_pattern = "session:*"
        
        # This would require Redis SCAN in production
        logger.info("Cleaning up old sessions")
        
        # Clean old temporary files
        temp_dirs = ['temp_media', 'temp_audio', 'temp_videos']
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    filepath = os.path.join(temp_dir, filename)
                    file_age = datetime.utcnow() - datetime.fromtimestamp(os.path.getctime(filepath))
                    if file_age > timedelta(hours=24):
                        try:
                            os.remove(filepath)
                            logger.debug(f"Removed old temp file: {filepath}")
                        except:
                            pass
        
        logger.info("Session cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}", exc_info=True)
        return False

def update_analytics():
    """Update analytics and generate reports"""
    try:
        # Update user statistics
        total_users = User.query.count()
        active_users = User.query.filter(
            User.last_seen >= datetime.utcnow() - timedelta(days=1)
        ).count()
        
        # Update transaction statistics
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_transactions = Transaction.query.filter(
            Transaction.created_at >= today_start,
            Transaction.status == 'completed'
        ).count()
        
        today_revenue = db.session.query(db.func.sum(Transaction.amount)).filter(
            Transaction.created_at >= today_start,
            Transaction.status == 'completed',
            Transaction.transaction_type.in_(['subscription', 'payment'])
        ).scalar() or 0
        
        # Store in Redis for real-time dashboard
        analytics_data = {
            'total_users': total_users,
            'active_users': active_users,
            'today_transactions': today_transactions,
            'today_revenue': float(today_revenue),
            'updated_at': datetime.utcnow().isoformat()
        }
        
        redis_client.set('analytics:daily', json.dumps(analytics_data), ex=3600)
        
        logger.info("Analytics updated", data=analytics_data)
        return True
        
    except Exception as e:
        logger.error(f"Analytics update failed: {e}", exc_info=True)
        return False

def process_pending_commissions():
    """Process pending affiliate commissions"""
    try:
        pending_commissions = Commission.query.filter_by(status='pending').all()
        
        for commission in pending_commissions:
            try:
                # Check if referral purchase is confirmed (30-day window)
                if commission.created_at < datetime.utcnow() - timedelta(days=30):
                    commission.status = 'confirmed'
                    commission.paid_at = datetime.utcnow()
                    
                    # Add to affiliate balance
                    affiliate = commission.affiliate
                    affiliate.balance += commission.amount
                    affiliate.total_earned += commission.amount
                    
                    db.session.commit()
                    
                    logger.info(f"Commission confirmed", 
                               commission_id=commission.id,
                               amount=commission.amount,
                               affiliate_id=affiliate.id)
                    
            except Exception as e:
                logger.error(f"Failed to process commission {commission.id}", error=str(e))
                db.session.rollback()
                continue
        
        logger.info(f"Processed {len(pending_commissions)} pending commissions")
        return True
        
    except Exception as e:
        logger.error(f"Commission processing failed: {e}", exc_info=True)
        return False

def cleanup_temp_files():
    """Clean up temporary files older than 24 hours"""
    try:
        temp_dirs = [
            'temp_media',
            'temp_audio',
            'temp_videos',
            'uploads/temp'
        ]
        
        files_removed = 0
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        try:
                            file_age = datetime.utcnow() - datetime.fromtimestamp(os.path.getmtime(filepath))
                            if file_age > timedelta(hours=24):
                                os.remove(filepath)
                                files_removed += 1
                        except:
                            pass
        
        logger.info(f"Cleaned up {files_removed} temporary files")
        return True
        
    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}", exc_info=True)
        return False

def optimize_models():
    """Optimize AI models and clear cache"""
    try:
        from app import ai_models
        
        # Clear model cache if too large
        cache_size = 0
        for model_name, model in ai_models.items():
            if hasattr(model, 'cache_info'):
                cache_info = model.cache_info()
                cache_size += cache_info.hits + cache_info.misses
        
        # Clear cache if too large
        if cache_size > 10000:
            for model_name, model in ai_models.items():
                if hasattr(model, 'clear_cache'):
                    model.clear_cache()
            
            logger.info("Model cache cleared", cache_size=cache_size)
        
        # Optimize model loading
        for model_name in ['image_generator', 'speech_to_text', 'text_to_speech']:
            if ai_models.get(model_name):
                # Move to GPU if available
                import torch
                if torch.cuda.is_available():
                    try:
                        ai_models[model_name].to('cuda')
                        logger.debug(f"Moved {model_name} to GPU")
                    except:
                        pass
        
        logger.info("Model optimization completed")
        return True
        
    except Exception as e:
        logger.error(f"Model optimization failed: {e}", exc_info=True)
        return False

def check_iot_devices():
    """Check IoT device status and send alerts"""
    try:
        from app.models.media import IoTDevice
        from app.utils.notifications import NotificationSystem
        
        # Check for offline devices
        offline_devices = IoTDevice.query.filter_by(status='offline').all()
        
        for device in offline_devices:
            # Check if device has been offline for more than 5 minutes
            if device.last_seen and device.last_seen < datetime.utcnow() - timedelta(minutes=5):
                # Send notification to user
                NotificationSystem.create_notification(
                    user_id=device.user_id,
                    notification_type='device_offline',
                    title='⚠️ Device Offline',
                    message=f'Your device "{device.name}" has been offline for more than 5 minutes.',
                    data={'device_id': device.id},
                    priority='medium'
                )
                
                logger.warning(f"Device offline alert", 
                              device_id=device.id,
                              device_name=device.name,
                              user_id=device.user_id)
        
        logger.info(f"Checked {len(offline_devices)} IoT devices")
        return True
        
    except Exception as e:
        logger.error(f"IoT device check failed: {e}", exc_info=True)
        return False

def send_daily_reports():
    """Send daily reports to administrators"""
    try:
        from app.utils.notifications import send_email_notification
        from flask import current_app
        
        # Get admin users
        admin_users = User.query.filter_by(is_administrator=True).all()
        
        # Generate daily report
        yesterday = datetime.utcnow() - timedelta(days=1)
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Get yesterday's statistics
        new_users = User.query.filter(
            User.created_at.between(yesterday_start, yesterday_end)
        ).count()
        
        completed_transactions = Transaction.query.filter(
            Transaction.created_at.between(yesterday_start, yesterday_end),
            Transaction.status == 'completed'
        ).count()
        
        total_revenue = db.session.query(db.func.sum(Transaction.amount)).filter(
            Transaction.created_at.between(yesterday_start, yesterday_end),
            Transaction.status == 'completed',
            Transaction.transaction_type.in_(['subscription', 'payment'])
        ).scalar() or 0
        
        pending_approvals = {
            'premium': User.query.filter_by(has_premium_access=False, premium_approved=False).count(),
            'transactions': Transaction.query.filter_by(status='requires_approval').count(),
            'withdrawals': WithdrawalRequest.query.filter_by(status='pending').count()
        }
        
        # Send report to each admin
        for admin in admin_users:
            send_email_notification(
                recipient=admin.email,
                subject=f"📊 Daily Report - {yesterday.strftime('%B %d, %Y')}",
                template='admin_daily_report',
                admin=admin,
                date=yesterday.strftime('%B %d, %Y'),
                new_users=new_users,
                completed_transactions=completed_transactions,
                total_revenue=float(total_revenue),
                pending_approvals=pending_approvals
            )
        
        logger.info(f"Sent daily reports to {len(admin_users)} admins")
        return True
        
    except Exception as e:
        logger.error(f"Daily report sending failed: {e}", exc_info=True)
        return False

def backup_database():
    """Create database backup"""
    try:
        from sqlalchemy import create_engine
        import subprocess
        import gzip
        
        backup_dir = 'backups'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Get database URL from config
        from config import Config
        db_url = Config.SQLALCHEMY_DATABASE_URI
        
        # Create backup filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(backup_dir, f'backup_{timestamp}.sql.gz')
        
        # Backup PostgreSQL database
        if db_url.startswith('postgresql://'):
            # Extract connection parameters
            import re
            match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', db_url)
            if match:
                username, password, host, port, database = match.groups()
                
                # Create backup using pg_dump
                env = os.environ.copy()
                env['PGPASSWORD'] = password
                
                cmd = [
                    'pg_dump',
                    '-h', host,
                    '-p', port,
                    '-U', username,
                    '-d', database,
                    '--no-password'
                ]
                
                with gzip.open(backup_file, 'wb') as f:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)
                    f.write(process.communicate()[0])
        
        # Backup SQLite database
        elif db_url.startswith('sqlite:///'):
            db_path = db_url.replace('sqlite:///', '')
            if os.path.exists(db_path):
                with open(db_path, 'rb') as f_in, gzip.open(backup_file, 'wb') as f_out:
                    f_out.writelines(f_in)
        
        # Keep only last 7 days of backups
        for filename in os.listdir(backup_dir):
            filepath = os.path.join(backup_dir, filename)
            file_age = datetime.utcnow() - datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_age > timedelta(days=7):
                os.remove(filepath)
        
        logger.info(f"Database backup created: {backup_file}")
        return True
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}", exc_info=True)
        return False

def update_currency_rates():
    """Update currency exchange rates"""
    try:
        import requests
        
        # Get latest exchange rates
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=10)
        
        if response.status_code == 200:
            rates = response.json()['rates']
            
            # Store in Redis with 24-hour expiry
            redis_client.set('currency_rates', json.dumps(rates), ex=86400)
            
            logger.info("Currency rates updated")
            return True
        else:
            logger.warning("Failed to fetch currency rates")
            return False
            
    except Exception as e:
        logger.error(f"Currency rate update failed: {e}")
        return False

def check_subscription_renewals():
    """Check and process subscription renewals"""
    try:
        from app.models.user import Subscription
        
        # Find subscriptions expiring in next 3 days
        renew_date = datetime.utcnow() + timedelta(days=3)
        expiring_subscriptions = Subscription.query.filter(
            Subscription.end_date <= renew_date,
            Subscription.status == 'active',
            Subscription.auto_renew == True
        ).all()
        
        for subscription in expiring_subscriptions:
            try:
                # Attempt auto-renewal
                if subscription.payment_method == 'stripe':
                    # Process Stripe renewal
                    success = process_stripe_renewal(subscription)
                elif subscription.payment_method == 'flutterwave':
                    # Process Flutterwave renewal
                    success = process_flutterwave_renewal(subscription)
                else:
                    # Manual renewal required
                    success = False
                
                if success:
                    subscription.end_date = datetime.utcnow() + timedelta(days=30)  # Extend by 30 days
                    subscription.updated_at = datetime.utcnow()
                    
                    # Send renewal confirmation
                    from app.utils.notifications import send_email_notification
                    send_email_notification(
                        recipient=subscription.user.email,
                        subject='✅ Subscription Renewed - ChangeX Neurix',
                        template='subscription_renewed',
                        user=subscription.user,
                        subscription=subscription
                    )
                    
                    logger.info(f"Subscription renewed", 
                               subscription_id=subscription.id,
                               user_id=subscription.user_id)
                else:
                    # Send renewal reminder
                    from app.utils.notifications import send_email_notification
                    send_email_notification(
                        recipient=subscription.user.email,
                        subject='🔔 Subscription Renewal Required - ChangeX Neurix',
                        template='subscription_renewal_reminder',
                        user=subscription.user,
                        subscription=subscription,
                        days_remaining=3
                    )
                    
                    logger.warning(f"Subscription renewal failed", 
                                  subscription_id=subscription.id,
                                  user_id=subscription.user_id)
                
                db.session.commit()
                
            except Exception as e:
                logger.error(f"Failed to process subscription renewal {subscription.id}", error=str(e))
                db.session.rollback()
                continue
        
        logger.info(f"Processed {len(expiring_subscriptions)} subscription renewals")
        return True
        
    except Exception as e:
        logger.error(f"Subscription renewal check failed: {e}", exc_info=True)
        return False

def process_stripe_renewal(subscription):
    """Process Stripe subscription renewal"""
    try:
        import stripe
        
        # Get customer from Stripe
        customer = stripe.Customer.retrieve(subscription.payment_id)
        
        # Charge the customer
        charge = stripe.Charge.create(
            amount=int(subscription.amount * 100),  # Convert to cents
            currency=subscription.currency.lower(),
            customer=customer.id,
            description=f"Subscription renewal - {subscription.tier} tier"
        )
        
        if charge.status == 'succeeded':
            # Create transaction record
            transaction = Transaction(
                user_id=subscription.user_id,
                amount=subscription.amount,
                currency=subscription.currency,
                payment_method='stripe',
                payment_gateway='stripe',
                gateway_transaction_id=charge.id,
                transaction_type='subscription',
                subscription_id=subscription.id,
                status='completed',
                description=f"Auto-renewal for {subscription.tier} subscription"
            )
            
            db.session.add(transaction)
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Stripe renewal failed: {e}")
        return False

def process_flutterwave_renewal(subscription):
    """Process Flutterwave subscription renewal"""
    try:
        from app import rave
        
        if not rave:
            return False
        
        # Get customer payment token
        # This would require storing Flutterwave tokens securely
        
        # For now, we'll just return False
        return False
        
    except Exception as e:
        logger.error(f"Flutterwave renewal failed: {e}")
        return False

class BackgroundTaskManager:
    """Manager for background tasks"""
    
    def __init__(self):
        self.tasks = {}
        self.is_running = False
    
    def start_all_tasks(self):
        """Start all background tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start task monitor
        self.start_task_monitor()
        
        # Start health check
        self.start_health_check()
        
        logger.info("Background task manager started")
    
    def start_task_monitor(self):
        """Monitor background tasks"""
        def monitor():
            while self.is_running:
                try:
                    # Check for failed jobs
                    failed_jobs = task_queue.failed_job_registry.get_job_ids()
                    if failed_jobs:
                        logger.warning(f"Found {len(failed_jobs)} failed jobs")
                        
                        # Retry failed jobs
                        for job_id in failed_jobs:
                            try:
                                job = task_queue.fetch_job(job_id)
                                if job:
                                    job.requeue()
                                    logger.info(f"Requeued failed job: {job_id}")
                            except:
                                pass
                    
                    # Clean old jobs
                    self.clean_old_jobs()
                    
                    time.sleep(300)  # Check every 5 minutes
                    
                except Exception as e:
                    logger.error(f"Task monitor error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def start_health_check(self):
        """Periodic health check"""
        def health_check():
            while self.is_running:
                try:
                    # Check Redis connection
                    if not redis_client.ping():
                        logger.error("Redis connection lost")
                    
                    # Check database connection
                    if not db.session.execute('SELECT 1').scalar():
                        logger.error("Database connection lost")
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    time.sleep(30)
        
        health_thread = threading.Thread(target=health_check, daemon=True)
        health_thread.start()
    
    def clean_old_jobs(self):
        """Clean old completed jobs"""
        try:
            # Clean jobs older than 7 days
            cutoff = datetime.utcnow() - timedelta(days=7)
            
            # This would require accessing RQ internals
            # In production, use RQ's built-in cleanup
            
            return True
            
        except Exception as e:
            logger.error(f"Job cleanup failed: {e}")
            return False
    
    def stop_all_tasks(self):
        """Stop all background tasks"""
        self.is_running = False
        logger.info("Background task manager stopped")
