# app/admin_management/routes.py - ADMIN APPROVAL SYSTEM
from flask import render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from app.admin_management import bp
from app import db
from app.models.user import User, AdminAction, Subscription
from app.models.payments import Transaction
from app.models.affiliate import WithdrawalRequest, Commission
from app.utils.decorators import admin_required, role_required
import json
from datetime import datetime, timedelta

@bp.route('/dashboard')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard with approval queues"""
    
    # Get pending approvals
    pending_premium_users = User.query.filter_by(
        has_premium_access=False,
        premium_approved=False
    ).filter(
        User.subscription_tier != 'free'
    ).count()
    
    pending_transactions = Transaction.query.filter_by(
        status='requires_approval',
        transaction_type='manual_payment'
    ).count()
    
    pending_withdrawals = WithdrawalRequest.query.filter_by(
        status='pending'
    ).count()
    
    pending_subscriptions = Subscription.query.filter_by(
        status='requires_approval'
    ).count()
    
    # Get recent admin actions
    recent_actions = AdminAction.query.order_by(
        AdminAction.action_timestamp.desc()
    ).limit(50).all()
    
    # Get system stats
    total_users = User.query.count()
    total_premium_users = User.query.filter_by(has_premium_access=True).count()
    total_transactions = Transaction.query.count()
    total_withdrawals = WithdrawalRequest.query.count()
    
    return render_template('admin_management/dashboard.html',
                         title='Admin Dashboard',
                         pending_premium_users=pending_premium_users,
                         pending_transactions=pending_transactions,
                         pending_withdrawals=pending_withdrawals,
                         pending_subscriptions=pending_subscriptions,
                         total_users=total_users,
                         total_premium_users=total_premium_users,
                         total_transactions=total_transactions,
                         total_withdrawals=total_withdrawals,
                         recent_actions=recent_actions)

@bp.route('/premium-approvals')
@login_required
@role_required('admin')
def premium_approval_queue():
    """Get list of users requiring premium approval"""
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Get users with premium subscription but not approved
    users = User.query.filter(
        User.subscription_tier.in_(['premium', 'pro', 'enterprise', 'custom']),
        User.has_premium_access == False,
        User.premium_approved == False
    ).order_by(User.created_at.desc())
    
    paginated_users = users.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'users': [user.to_dict(include_sensitive=True) for user in paginated_users.items],
        'total': paginated_users.total,
        'pages': paginated_users.pages,
        'current_page': paginated_users.page,
        'per_page': paginated_users.per_page
    })

@bp.route('/premium-approve/<int:user_id>', methods=['POST'])
@login_required
@role_required('admin')
def approve_premium(user_id):
    """Approve premium access for user"""
    
    data = request.get_json()
    notes = data.get('notes', '')
    
    user = User.query.get_or_404(user_id)
    
    if user.has_premium_access:
        return jsonify({'error': 'User already has premium access'}), 400
    
    # Approve premium access
    success = user.approve_premium_access(current_user, notes)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Premium access approved successfully',
            'user': user.to_dict()
        })
    
    return jsonify({'error': 'Failed to approve premium access'}), 500

@bp.route('/premium-reject/<int:user_id>', methods=['POST'])
@login_required
@role_required('admin')
def reject_premium(user_id):
    """Reject premium access for user"""
    
    data = request.get_json()
    reason = data.get('reason', '')
    
    if not reason:
        return jsonify({'error': 'Rejection reason is required'}), 400
    
    user = User.query.get_or_404(user_id)
    
    if user.has_premium_access:
        return jsonify({'error': 'User already has premium access'}), 400
    
    # Reject premium access
    success = user.reject_premium_access(current_user, reason)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Premium access rejected',
            'user': user.to_dict()
        })
    
    return jsonify({'error': 'Failed to reject premium access'}), 500

@bp.route('/transaction-approvals')
@login_required
@role_required('admin')
def transaction_approval_queue():
    """Get list of transactions requiring approval"""
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    transactions = Transaction.query.filter_by(
        status='requires_approval',
        transaction_type='manual_payment'
    ).order_by(Transaction.created_at.desc())
    
    paginated_transactions = transactions.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'transactions': [tx.to_dict() for tx in paginated_transactions.items],
        'total': paginated_transactions.total,
        'pages': paginated_transactions.pages,
        'current_page': paginated_transactions.page,
        'per_page': paginated_transactions.per_page
    })

@bp.route('/transaction-approve/<int:transaction_id>', methods=['POST'])
@login_required
@role_required('admin')
def approve_transaction(transaction_id):
    """Approve manual payment transaction"""
    
    data = request.get_json()
    notes = data.get('notes', '')
    
    transaction = Transaction.query.get_or_404(transaction_id)
    
    if transaction.status != 'requires_approval':
        return jsonify({'error': 'Transaction does not require approval'}), 400
    
    # Approve transaction
    transaction.status = 'completed'
    transaction.processed_at = datetime.utcnow()
    transaction.processed_by = current_user.id
    transaction.notes = notes
    
    # Update user subscription if applicable
    if transaction.user_id and transaction.amount > 0:
        user = User.query.get(transaction.user_id)
        if user:
            # Activate premium access
            user.has_premium_access = True
            user.premium_approved = True
            user.premium_approved_by = current_user.id
            user.premium_approved_at = datetime.utcnow()
            
            # Update subscription
            subscription = user.subscription
            if subscription:
                subscription.status = 'active'
                subscription.payment_status = 'completed'
                subscription.requires_approval = False
                subscription.approved_by = current_user.id
                subscription.approved_at = datetime.utcnow()
    
    # Log admin action
    action = AdminAction(
        admin_id=current_user.id,
        action_type='transaction_approval',
        target_type='transaction',
        target_id=transaction.id,
        details=f"Manual payment approved. Amount: {transaction.amount} {transaction.currency}. Notes: {notes}",
        ip_address=request.remote_addr
    )
    db.session.add(action)
    db.session.commit()
    
    # Send notification
    from app.utils.notifications import send_transaction_approval_notification
    send_transaction_approval_notification(transaction)
    
    return jsonify({
        'success': True,
        'message': 'Transaction approved successfully',
        'transaction': transaction.to_dict()
    })

@bp.route('/transaction-reject/<int:transaction_id>', methods=['POST'])
@login_required
@role_required('admin')
def reject_transaction(transaction_id):
    """Reject manual payment transaction"""
    
    data = request.get_json()
    reason = data.get('reason', '')
    
    if not reason:
        return jsonify({'error': 'Rejection reason is required'}), 400
    
    transaction = Transaction.query.get_or_404(transaction_id)
    
    if transaction.status != 'requires_approval':
        return jsonify({'error': 'Transaction does not require approval'}), 400
    
    # Reject transaction
    transaction.status = 'rejected'
    transaction.rejection_reason = reason
    transaction.processed_at = datetime.utcnow()
    transaction.processed_by = current_user.id
    
    # Log admin action
    action = AdminAction(
        admin_id=current_user.id,
        action_type='transaction_rejection',
        target_type='transaction',
        target_id=transaction.id,
        details=f"Manual payment rejected. Reason: {reason}",
        ip_address=request.remote_addr
    )
    db.session.add(action)
    db.session.commit()
    
    # Send notification
    from app.utils.notifications import send_transaction_rejection_notification
    send_transaction_rejection_notification(transaction, reason)
    
    return jsonify({
        'success': True,
        'message': 'Transaction rejected',
        'transaction': transaction.to_dict()
    })

@bp.route('/withdrawal-approvals')
@login_required
@role_required('admin')
def withdrawal_approval_queue():
    """Get list of withdrawal requests requiring approval"""
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    withdrawals = WithdrawalRequest.query.filter_by(
        status='pending'
    ).order_by(WithdrawalRequest.created_at.desc())
    
    paginated_withdrawals = withdrawals.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'withdrawals': [w.to_dict() for w in paginated_withdrawals.items],
        'total': paginated_withdrawals.total,
        'pages': paginated_withdrawals.pages,
        'current_page': paginated_withdrawals.page,
        'per_page': paginated_withdrawals.per_page
    })

@bp.route('/withdrawal-approve/<int:withdrawal_id>', methods=['POST'])
@login_required
@role_required('admin')
def approve_withdrawal(withdrawal_id):
    """Approve affiliate withdrawal request"""
    
    data = request.get_json()
    notes = data.get('notes', '')
    
    withdrawal = WithdrawalRequest.query.get_or_404(withdrawal_id)
    
    if withdrawal.status != 'pending':
        return jsonify({'error': 'Withdrawal is not pending'}), 400
    
    # Check if affiliate has sufficient balance
    affiliate = withdrawal.affiliate
    if affiliate.balance < withdrawal.amount:
        return jsonify({'error': 'Insufficient affiliate balance'}), 400
    
    # Approve withdrawal
    withdrawal.status = 'approved'
    withdrawal.approved_by = current_user.id
    withdrawal.approved_at = datetime.utcnow()
    withdrawal.notes = notes
    
    # Deduct from affiliate balance
    affiliate.balance -= withdrawal.amount
    affiliate.total_withdrawn += withdrawal.amount
    
    # Log admin action
    action = AdminAction(
        admin_id=current_user.id,
        action_type='withdrawal_approval',
        target_type='withdrawal',
        target_id=withdrawal.id,
        details=f"Withdrawal approved. Amount: {withdrawal.amount} {withdrawal.currency}. Notes: {notes}",
        ip_address=request.remote_addr
    )
    db.session.add(action)
    db.session.commit()
    
    # Send notification
    from app.utils.notifications import send_withdrawal_approval_notification
    send_withdrawal_approval_notification(withdrawal)
    
    return jsonify({
        'success': True,
        'message': 'Withdrawal approved successfully',
        'withdrawal': withdrawal.to_dict()
    })

@bp.route('/withdrawal-reject/<int:withdrawal_id>', methods=['POST'])
@login_required
@role_required('admin')
def reject_withdrawal(withdrawal_id):
    """Reject affiliate withdrawal request"""
    
    data = request.get_json()
    reason = data.get('reason', '')
    
    if not reason:
        return jsonify({'error': 'Rejection reason is required'}), 400
    
    withdrawal = WithdrawalRequest.query.get_or_404(withdrawal_id)
    
    if withdrawal.status != 'pending':
        return jsonify({'error': 'Withdrawal is not pending'}), 400
    
    # Reject withdrawal
    withdrawal.status = 'rejected'
    withdrawal.rejection_reason = reason
    withdrawal.processed_at = datetime.utcnow()
    withdrawal.processed_by = current_user.id
    
    # Log admin action
    action = AdminAction(
        admin_id=current_user.id,
        action_type='withdrawal_rejection',
        target_type='withdrawal',
        target_id=withdrawal.id,
        details=f"Withdrawal rejected. Reason: {reason}",
        ip_address=request.remote_addr
    )
    db.session.add(action)
    db.session.commit()
    
    # Send notification
    from app.utils.notifications import send_withdrawal_rejection_notification
    send_withdrawal_rejection_notification(withdrawal, reason)
    
    return jsonify({
        'success': True,
        'message': 'Withdrawal rejected',
        'withdrawal': withdrawal.to_dict()
    })

@bp.route('/subscription-approvals')
@login_required
@role_required('admin')
def subscription_approval_queue():
    """Get list of subscriptions requiring approval"""
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    subscriptions = Subscription.query.filter_by(
        status='requires_approval'
    ).order_by(Subscription.created_at.desc())
    
    paginated_subscriptions = subscriptions.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'subscriptions': [sub.to_dict() for sub in paginated_subscriptions.items],
        'total': paginated_subscriptions.total,
        'pages': paginated_subscriptions.pages,
        'current_page': paginated_subscriptions.page,
        'per_page': paginated_subscriptions.per_page
    })

@bp.route('/subscription-approve/<int:subscription_id>', methods=['POST'])
@login_required
@role_required('admin')
def approve_subscription(subscription_id):
    """Approve subscription"""
    
    data = request.get_json()
    notes = data.get('notes', '')
    
    subscription = Subscription.query.get_or_404(subscription_id)
    
    if subscription.status != 'requires_approval':
        return jsonify({'error': 'Subscription does not require approval'}), 400
    
    # Approve subscription
    success = subscription.approve(current_user, notes)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Subscription approved successfully',
            'subscription': subscription.to_dict()
        })
    
    return jsonify({'error': 'Failed to approve subscription'}), 500

@bp.route('/subscription-reject/<int:subscription_id>', methods=['POST'])
@login_required
@role_required('admin')
def reject_subscription(subscription_id):
    """Reject subscription"""
    
    data = request.get_json()
    reason = data.get('reason', '')
    
    if not reason:
        return jsonify({'error': 'Rejection reason is required'}), 400
    
    subscription = Subscription.query.get_or_404(subscription_id)
    
    if subscription.status != 'requires_approval':
        return jsonify({'error': 'Subscription does not require approval'}), 400
    
    # Reject subscription
    success = subscription.reject(current_user, reason)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Subscription rejected',
            'subscription': subscription.to_dict()
        })
    
    return jsonify({'error': 'Failed to reject subscription'}), 500

@bp.route('/user/<int:user_id>/toggle-premium', methods=['POST'])
@login_required
@role_required('admin')
def toggle_user_premium(user_id):
    """Toggle premium access for user (admin override)"""
    
    data = request.get_json()
    action = data.get('action', 'grant')  # grant or revoke
    reason = data.get('reason', '')
    
    user = User.query.get_or_404(user_id)
    
    if action == 'grant':
        user.has_premium_access = True
        user.premium_approved = True
        user.premium_approved_by = current_user.id
        user.premium_approved_at = datetime.utcnow()
        message = 'Premium access granted'
    else:
        user.has_premium_access = False
        user.premium_approved = False
        user.premium_rejection_reason = reason or 'Access revoked by admin'
        message = 'Premium access revoked'
    
    # Log admin action
    action_type = 'premium_grant' if action == 'grant' else 'premium_revoke'
    admin_action = AdminAction(
        admin_id=current_user.id,
        action_type=action_type,
        target_type='user',
        target_id=user.id,
        details=f"{message}. Reason: {reason or 'No reason provided'}",
        ip_address=request.remote_addr
    )
    db.session.add(admin_action)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': message,
        'user': user.to_dict()
    })

@bp.route('/audit-logs')
@login_required
@role_required('admin')
def get_audit_logs():
    """Get admin audit logs"""
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    action_type = request.args.get('action_type')
    admin_id = request.args.get('admin_id', type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    query = AdminAction.query
    
    # Apply filters
    if action_type:
        query = query.filter_by(action_type=action_type)
    
    if admin_id:
        query = query.filter_by(admin_id=admin_id)
    
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            query = query.filter(AdminAction.action_timestamp >= start)
        except ValueError:
            pass
    
    if end_date:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            query = query.filter(AdminAction.action_timestamp <= end)
        except ValueError:
            pass
    
    # Order and paginate
    logs = query.order_by(AdminAction.action_timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'logs': [log.to_dict() for log in logs.items],
        'total': logs.total,
        'pages': logs.pages,
        'current_page': logs.page,
        'per_page': logs.per_page
    })

@bp.route('/system-stats')
@login_required
@role_required('admin')
def get_system_stats():
    """Get system statistics"""
    
    # User statistics
    total_users = User.query.count()
    active_users_24h = User.query.filter(
        User.last_seen >= datetime.utcnow() - timedelta(hours=24)
    ).count()
    new_users_today = User.query.filter(
        User.created_at >= datetime.utcnow().date()
    ).count()
    premium_users = User.query.filter_by(has_premium_access=True).count()
    
    # Transaction statistics
    total_transactions = Transaction.query.count()
    total_revenue = db.session.query(db.func.sum(Transaction.amount)).filter(
        Transaction.status == 'completed',
        Transaction.transaction_type.in_(['subscription', 'payment'])
    ).scalar() or 0
    
    # Withdrawal statistics
    total_withdrawals = WithdrawalRequest.query.count()
    total_paid_out = db.session.query(db.func.sum(WithdrawalRequest.amount)).filter(
        WithdrawalRequest.status == 'paid'
    ).scalar() or 0
    
    # Media generation statistics
    from app.models.media import GeneratedImage, GeneratedVideo, AudioFile
    total_images = GeneratedImage.query.count()
    total_videos = GeneratedVideo.query.count()
    total_audio = AudioFile.query.count()
    
    # Pending actions
    pending_premium = User.query.filter_by(has_premium_access=False, premium_approved=False).count()
    pending_transactions = Transaction.query.filter_by(status='requires_approval').count()
    pending_withdrawals = WithdrawalRequest.query.filter_by(status='pending').count()
    
    return jsonify({
        'user_stats': {
            'total_users': total_users,
            'active_users_24h': active_users_24h,
            'new_users_today': new_users_today,
            'premium_users': premium_users,
            'free_users': total_users - premium_users
        },
        'financial_stats': {
            'total_transactions': total_transactions,
            'total_revenue': float(total_revenue),
            'total_withdrawals': total_withdrawals,
            'total_paid_out': float(total_paid_out),
            'net_revenue': float(total_revenue - total_paid_out)
        },
        'media_stats': {
            'total_images': total_images,
            'total_videos': total_videos,
            'total_audio': total_audio,
            'total_media': total_images + total_videos + total_audio
        },
        'pending_actions': {
            'premium_approvals': pending_premium,
            'transaction_approvals': pending_transactions,
            'withdrawal_approvals': pending_withdrawals,
            'total_pending': pending_premium + pending_transactions + pending_withdrawals
        },
        'timestamp': datetime.utcnow().isoformat()
    })
