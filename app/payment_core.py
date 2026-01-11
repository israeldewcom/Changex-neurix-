# app/payments/core.py - ENHANCED PAYMENT SYSTEM WITH APPROVAL
import stripe
import logging
from datetime import datetime
from flask import current_app, request
from app import db
from app.models.user import User, AdminAction
from app.models.payments import Transaction
from app.utils.notifications import NotificationSystem

logger = logging.getLogger(__name__)

class PaymentSystem:
    """Enhanced payment system with approval workflow"""
    
    def __init__(self):
        self.stripe_api_key = current_app.config.get('STRIPE_SECRET_KEY')
        self.flutterwave_public_key = current_app.config.get('FLUTTERWAVE_PUBLIC_KEY')
        self.flutterwave_secret_key = current_app.config.get('FLUTTERWAVE_SECRET_KEY')
        
        if self.stripe_api_key:
            stripe.api_key = self.stripe_api_key
    
    def initialize(self):
        """Initialize payment system"""
        logger.info("Payment system initialized")
        return True
    
    def create_stripe_payment_intent(self, amount, currency='usd', metadata=None):
        """Create Stripe payment intent"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency,
                metadata=metadata or {},
                automatic_payment_methods={
                    'enabled': True,
                }
            )
            
            return {
                'success': True,
                'client_secret': intent.client_secret,
                'payment_intent_id': intent.id,
                'amount': amount,
                'currency': currency
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe payment intent creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_stripe_webhook(self, payload, sig_header):
        """Process Stripe webhook"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, current_app.config['STRIPE_WEBHOOK_SECRET']
            )
            
            # Handle different event types
            if event['type'] == 'payment_intent.succeeded':
                self._handle_stripe_payment_success(event['data']['object'])
            elif event['type'] == 'payment_intent.payment_failed':
                self._handle_stripe_payment_failure(event['data']['object'])
            elif event['type'] == 'customer.subscription.created':
                self._handle_stripe_subscription_created(event['data']['object'])
            elif event['type'] == 'customer.subscription.deleted':
                self._handle_stripe_subscription_deleted(event['data']['object'])
            
            return {'success': True}
            
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid Stripe webhook signature: {e}")
            return {'success': False, 'error': 'Invalid signature'}
        except Exception as e:
            logger.error(f"Stripe webhook processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_stripe_payment_success(self, payment_intent):
        """Handle successful Stripe payment"""
        try:
            # Create transaction record
            transaction = Transaction(
                transaction_id=payment_intent['id'],
                amount=payment_intent['amount'] / 100,  # Convert from cents
                currency=payment_intent['currency'],
                payment_method='stripe',
                payment_gateway='stripe',
                gateway_transaction_id=payment_intent['id'],
                gateway_response=payment_intent,
                status='completed',
                metadata=payment_intent.get('metadata', {})
            )
            
            # Set user if available in metadata
            if 'user_id' in payment_intent.get('metadata', {}):
                transaction.user_id = payment_intent['metadata']['user_id']
            
            db.session.add(transaction)
            db.session.commit()
            
            # Update user subscription if applicable
            if transaction.user_id and 'subscription_tier' in payment_intent.get('metadata', {}):
                self._update_user_subscription(
                    transaction.user_id,
                    payment_intent['metadata']['subscription_tier'],
                    'stripe',
                    payment_intent['id']
                )
            
            logger.info(f"Stripe payment processed: {payment_intent['id']}")
            
        except Exception as e:
            logger.error(f"Failed to handle Stripe payment success: {e}")
            db.session.rollback()
    
    def process_manual_payment(self, user_id, amount, currency, proof_url, details):
        """Process manual payment (bank transfer, etc.)"""
        try:
            # Create transaction requiring approval
            transaction = Transaction(
                user_id=user_id,
                amount=amount,
                currency=currency,
                payment_method='manual',
                payment_gateway='manual',
                transaction_type='payment',
                status='requires_approval',
                is_manual_payment=True,
                manual_payment_proof=proof_url,
                manual_payment_details=details,
                description=f"Manual payment: {details.get('description', 'No description')}"
            )
            
            db.session.add(transaction)
            db.session.commit()
            
            # Notify admins about pending approval
            self._notify_admins_pending_approval('transaction', transaction.id)
            
            # Send confirmation to user
            NotificationSystem.create_notification(
                user_id=user_id,
                notification_type='manual_payment_submitted',
                title='💰 Payment Submitted',
                message='Your manual payment has been submitted and is pending approval.',
                data={'transaction_id': transaction.id},
                priority='medium'
            )
            
            logger.info(f"Manual payment submitted for approval: {transaction.id}")
            
            return {
                'success': True,
                'transaction_id': transaction.id,
                'message': 'Payment submitted for approval'
            }
            
        except Exception as e:
            logger.error(f"Manual payment processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _update_user_subscription(self, user_id, tier, payment_method, payment_id):
        """Update user subscription after payment"""
        try:
            user = User.query.get(user_id)
            if not user:
                return False
            
            # Update user subscription
            user.subscription_tier = tier
            user.has_premium_access = True
            user.premium_approved = True
            user.premium_approved_at = datetime.utcnow()
            
            # Create or update subscription record
            from app.models.user import Subscription
            subscription = Subscription.query.filter_by(user_id=user_id).first()
            
            if not subscription:
                subscription = Subscription(user_id=user_id)
                db.session.add(subscription)
            
            subscription.tier = tier
            subscription.status = 'active'
            subscription.amount = current_app.config['PRICING_TIERS'][tier]['price']
            subscription.payment_method = payment_method
            subscription.payment_id = payment_id
            subscription.payment_status = 'completed'
            subscription.start_date = datetime.utcnow()
            subscription.end_date = datetime.utcnow() + timedelta(days=30)  # 30-day subscription
            
            db.session.commit()
            
            # Send notification
            NotificationSystem.send_premium_approved_notification(user)
            
            logger.info(f"User subscription updated: {user_id} -> {tier}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user subscription: {e}")
            db.session.rollback()
            return False
    
    def _notify_admins_pending_approval(self, target_type, target_id):
        """Notify admins about pending approval"""
        from app.models.user import User
        
        admins = User.query.filter_by(is_administrator=True).all()
        
        for admin in admins:
            NotificationSystem.create_notification(
                user_id=admin.id,
                notification_type='admin_approval_required',
                title='📋 Approval Required',
                message=f'{target_type.replace("_", " ").title()} requires your approval',
                data={
                    'target_type': target_type,
                    'target_id': target_id
                },
                priority='high'
            )
    
    def get_payment_methods(self):
        """Get available payment methods"""
        methods = [
            {
                'id': 'stripe',
                'name': 'Credit/Debit Card',
                'icon': 'credit-card',
                'supported_currencies': ['USD', 'EUR', 'GBP'],
                'instant': True
            },
            {
                'id': 'flutterwave',
                'name': 'Flutterwave',
                'icon': 'globe',
                'supported_currencies': ['USD', 'NGN', 'KES', 'GHS'],
                'instant': True
            },
            {
                'id': 'manual',
                'name': 'Bank Transfer',
                'icon': 'bank',
                'supported_currencies': ['USD', 'EUR', 'GBP', 'NGN'],
                'instant': False,
                'requires_approval': True,
                'instructions': current_app.config.get('MANUAL_PAYMENT_INSTRUCTIONS', '')
            }
        ]
        
        return methods
    
    def get_transaction_history(self, user_id, limit=50, offset=0):
        """Get user transaction history"""
        transactions = Transaction.query.filter_by(user_id=user_id)\
            .order_by(Transaction.created_at.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()
        
        return [tx.to_dict() for tx in transactions]
    
    def refund_transaction(self, transaction_id, admin_user, reason=None):
        """Refund transaction"""
        try:
            transaction = Transaction.query.get_or_404(transaction_id)
            
            if transaction.status != 'completed':
                return {'success': False, 'error': 'Transaction not completed'}
            
            # Process refund based on payment method
            if transaction.payment_method == 'stripe':
                # Process Stripe refund
                refund = stripe.Refund.create(
                    payment_intent=transaction.gateway_transaction_id,
                    reason='requested_by_customer'
                )
                
                if refund.status == 'succeeded':
                    transaction.status = 'refunded'
                    
                    # Create refund transaction
                    refund_tx = Transaction(
                        user_id=transaction.user_id,
                        amount=transaction.amount,
                        currency=transaction.currency,
                        payment_method=transaction.payment_method,
                        payment_gateway=transaction.payment_gateway,
                        gateway_transaction_id=refund.id,
                        transaction_type='refund',
                        status='completed',
                        description=f"Refund for transaction {transaction.id}. Reason: {reason or 'No reason provided'}"
                    )
                    
                    db.session.add(refund_tx)
                    
                    # Log admin action
                    action = AdminAction(
                        admin_id=admin_user.id,
                        action_type='refund_processed',
                        target_type='transaction',
                        target_id=transaction.id,
                        details=f"Refund processed. Amount: {transaction.amount} {transaction.currency}. Reason: {reason or 'No reason provided'}",
                        ip_address=request.remote_addr
                    )
                    db.session.add(action)
                    
                    db.session.commit()
                    
                    # Notify user
                    NotificationSystem.create_notification(
                        user_id=transaction.user_id,
                        notification_type='refund_processed',
                        title='💸 Refund Processed',
                        message=f'Your refund of {transaction.amount} {transaction.currency} has been processed.',
                        data={'transaction_id': transaction.id},
                        priority='medium'
                    )
                    
                    return {'success': True, 'refund_id': refund.id}
                
                else:
                    return {'success': False, 'error': 'Refund failed'}
            
            else:
                # Manual refund process
                transaction.status = 'refunded'
                
                # Create refund transaction
                refund_tx = Transaction(
                    user_id=transaction.user_id,
                    amount=transaction.amount,
                    currency=transaction.currency,
                    payment_method=transaction.payment_method,
                    transaction_type='refund',
                    status='completed',
                    description=f"Manual refund for transaction {transaction.id}. Reason: {reason or 'No reason provided'}"
                )
                
                db.session.add(refund_tx)
                
                # Log admin action
                action = AdminAction(
                    admin_id=admin_user.id,
                    action_type='refund_processed',
                    target_type='transaction',
                    target_id=transaction.id,
                    details=f"Manual refund processed. Amount: {transaction.amount} {transaction.currency}. Reason: {reason or 'No reason provided'}",
                    ip_address=request.remote_addr
                )
                db.session.add(action)
                
                db.session.commit()
                
                # Notify user
                NotificationSystem.create_notification(
                    user_id=transaction.user_id,
                    notification_type='refund_processed',
                    title='💸 Refund Processed',
                    message=f'Your manual refund of {transaction.amount} {transaction.currency} has been processed.',
                    data={'transaction_id': transaction.id},
                    priority='medium'
                )
                
                return {'success': True}
                
        except Exception as e:
            logger.error(f"Refund processing failed: {e}")
            return {'success': False, 'error': str(e)}
