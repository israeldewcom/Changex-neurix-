# deploy.sh - PRODUCTION DEPLOYMENT SCRIPT
#!/bin/bash

# ChangeX Neurix Production Deployment Script

set -e  # Exit on error

echo "🚀 ChangeX Neurix Production Deployment"
echo "======================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Configuration
APP_NAME="changex-neurix"
DOMAIN="${DOMAIN:-changexneurix.com}"
DEPLOY_DIR="/opt/${APP_NAME}"
BACKUP_DIR="${DEPLOY_DIR}/backups"

print_step() {
    echo -e "\n${GREEN}[+]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# 1. System Setup
system_setup() {
    print_step "System setup..."
    
    sudo apt-get update -y
    sudo apt-get upgrade -y
    
    sudo apt-get install -y \
        docker.io \
        docker-compose \
        nginx \
        certbot \
        python3-certbot-nginx \
        postgresql-client \
        redis-tools \
        curl \
        git \
        jq \
        unzip \
        fail2ban \
        ufw
    
    # Configure firewall
    sudo ufw --force enable
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    sudo ufw allow ssh
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw --force reload
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    newgrp docker
    
    # Enable services
    sudo systemctl enable docker
    sudo systemctl start docker
}

# 2. Directory Setup
setup_directories() {
    print_step "Setting up directories..."
    
    sudo mkdir -p $DEPLOY_DIR
    sudo mkdir -p $BACKUP_DIR
    sudo mkdir -p $DEPLOY_DIR/{nginx,postgres,backups}
    sudo mkdir -p $DEPLOY_DIR/nginx/{ssl,conf.d}
    
    sudo chown -R $USER:$USER $DEPLOY_DIR
    
    mkdir -p $DEPLOY_DIR/media/{images,videos,audio,uploads}
    mkdir -p $DEPLOY_DIR/logs/{app,nginx,postgres}
    mkdir -p $DEPLOY_DIR/model_cache
}

# 3. Generate SSL Certificates
generate_ssl() {
    print_step "Generating SSL certificates..."
    
    if [ ! -f "$DEPLOY_DIR/nginx/ssl/dhparam.pem" ]; then
        sudo openssl dhparam -out $DEPLOY_DIR/nginx/ssl/dhparam.pem 2048
    fi
    
    # Generate self-signed certificate for initial setup
    if [ ! -f "$DEPLOY_DIR/nginx/ssl/cert.pem" ]; then
        sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout $DEPLOY_DIR/nginx/ssl/privkey.pem \
            -out $DEPLOY_DIR/nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=ChangeX Neurix/CN=$DOMAIN"
    fi
    
    print_info "SSL certificates generated. For production, run:"
    print_info "sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos"
}

# 4. Generate Environment File
generate_env() {
    print_step "Generating environment file..."
    
    if [ ! -f "$DEPLOY_DIR/.env.production" ]; then
        cat > $DEPLOY_DIR/.env.production << EOF
# ChangeX Neurix Production Environment
# =====================================

# Flask Configuration
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Database
DB_USER=neurix_user
DB_PASSWORD=$(openssl rand -hex 16)
DB_NAME=changex_neurix

# Redis
REDIS_PASSWORD=$(openssl rand -hex 16)

# Email (Configure with your SMTP server)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password

# Payment Gateways (Configure with your keys)
STRIPE_PUBLIC_KEY=pk_test_...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
FLUTTERWAVE_PUBLIC_KEY=FLWPUBK_TEST_...
FLUTTERWAVE_SECRET_KEY=FLWSECK_TEST_...

# Manual Payment Configuration
MANUAL_PAYMENT_PHONE=+1234567890
MANUAL_PAYMENT_NAME=Admin Name
MANUAL_PAYMENT_BANK=Bank Name

# Domain Configuration
APP_URL=https://$DOMAIN
API_URL=https://api.$DOMAIN

# Admin Approval Settings
PREMIUM_APPROVAL_REQUIRED=true
MANUAL_PAYMENT_APPROVAL_REQUIRED=true
WITHDRAWAL_APPROVAL_REQUIRED=true

# Monitoring
SENTRY_DSN=https://your_sentry_dsn@sentry.io/1234567
EOF
        
        print_warning "Generated .env.production file"
        print_warning "Please edit this file to add your actual API keys and configuration"
    fi
}

# 5. Copy Application Files
copy_application() {
    print_step "Copying application files..."
    
    # Copy all files except development-only
    rsync -av \
        --exclude='.git' \
        --exclude='.env*' \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='node_modules' \
        --exclude='venv' \
        --exclude='.venv' \
        --exclude='tmp' \
        --exclude='*.log' \
        ./ $DEPLOY_DIR/
    
    # Make scripts executable
    chmod +x $DEPLOY_DIR/deploy.sh
    chmod +x $DEPLOY_DIR/run.py
}

# 6. Configure Nginx
configure_nginx() {
    print_step "Configuring Nginx..."
    
    cat > $DEPLOY_DIR/nginx/nginx.conf << 'EOF'
user www-data;
worker_processes auto;
pid /run/nginx.pid;

events {
    worker_connections 1024;
    multi_accept on;
}

http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_ecdh_curve secp384r1;
    ssl_session_timeout 10m;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript
               application/json application/javascript application/xml+rss
               application/atom+xml image/svg+xml;

    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # Include site configurations
    include /etc/nginx/conf.d/*.conf;
}
EOF
    
    cat > $DEPLOY_DIR/nginx/conf.d/changex-neurix.conf << EOF
# ChangeX Neurix Nginx Configuration

server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    
    # Redirect to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name $DOMAIN www.$DOMAIN;
    
    # SSL Certificates
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_dhparam /etc/nginx/ssl/dhparam.pem;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Client Settings
    client_max_body_size 1G;
    client_body_timeout 300s;
    
    # Proxy Settings
    proxy_http_version 1.1;
    proxy_set_header Upgrade \$http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host \$host;
    proxy_set_header X-Real-IP \$remote_addr;
    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto \$scheme;
    proxy_cache_bypass \$http_upgrade;
    
    # Static Files
    location /static/ {
        alias /var/www/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
    
    location /media/ {
        alias /var/www/media/;
        expires 7d;
        add_header Cache-Control "public";
    }
    
    # WebSocket Support
    location /socket.io/ {
        proxy_pass http://web:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    
    # API Rate Limiting
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://web:8000;
    }
    
    # Health Check
    location /health {
        proxy_pass http://web:8000;
        access_log off;
    }
    
    # Main Application
    location / {
        proxy_pass http://web:8000;
    }
}
EOF
}

# 7. Create Backup Script
create_backup_script() {
    print_step "Creating backup script..."
    
    cat > $DEPLOY_DIR/backup.sh << 'EOF'
#!/bin/bash

# ChangeX Neurix Backup Script

set -e

BACKUP_DIR="/opt/changex-neurix/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"

echo "🔒 Starting backup: $BACKUP_FILE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Stop services to ensure consistency
cd /opt/changex-neurix
docker-compose stop

# Backup PostgreSQL
echo "Backing up PostgreSQL..."
docker-compose exec -T postgres pg_dumpall -U neurix_user > $BACKUP_DIR/postgres_backup_$TIMESTAMP.sql

# Backup Redis
echo "Backing up Redis..."
docker-compose exec -T redis redis-cli --raw SAVE
cp -r /opt/changex-neurix/redis_data $BACKUP_DIR/redis_$TIMESTAMP 2>/dev/null || true

# Backup media files
echo "Backing up media files..."
tar czf $BACKUP_DIR/media_$TIMESTAMP.tar.gz media/ 2>/dev/null || true

# Backup configuration
echo "Backing up configuration..."
tar czf $BACKUP_DIR/config_$TIMESTAMP.tar.gz \
    .env.production \
    docker-compose.yml \
    nginx/ \
    --exclude=nginx/ssl

# Create single archive
echo "Creating final archive..."
tar czf $BACKUP_FILE \
    $BACKUP_DIR/postgres_backup_$TIMESTAMP.sql \
    $BACKUP_DIR/media_$TIMESTAMP.tar.gz \
    $BACKUP_DIR/config_$TIMESTAMP.tar.gz

# Cleanup temporary files
rm -f $BACKUP_DIR/*_$TIMESTAMP.*

# Start services
docker-compose start

# Remove old backups (keep last 30 days)
find $BACKUP_DIR -name "backup_*.tar.gz" -mtime +30 -delete

echo "✅ Backup completed: $BACKUP_FILE"
echo "Size: $(du -h $BACKUP_FILE | cut -f1)"
EOF
    
    chmod +x $DEPLOY_DIR/backup.sh
}

# 8. Create Systemd Service
create_systemd_service() {
    print_step "Creating systemd service..."
    
    cat > /tmp/changex-neurix.service << EOF
[Unit]
Description=ChangeX Neurix
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$DEPLOY_DIR
ExecStart=/usr/bin/docker-compose -f $DEPLOY_DIR/docker-compose.production.yml up -d
ExecStop=/usr/bin/docker-compose -f $DEPLOY_DIR/docker-compose.production.yml down
User=$USER
Group=$USER

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv /tmp/changex-neurix.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable changex-neurix.service
}

# 9. Initialize Application
initialize_application() {
    print_step "Initializing application..."
    
    cd $DEPLOY_DIR
    
    # Build and start services
    docker-compose -f docker-compose.production.yml build --no-cache
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to start
    sleep 30
    
    # Run database migrations
    print_step "Running database migrations..."
    docker-compose -f docker-compose.production.yml exec web flask db upgrade
    
    # Create admin user
    print_step "Creating admin user..."
    
    read -p "Enter admin email: " ADMIN_EMAIL
    read -s -p "Enter admin password: " ADMIN_PASSWORD
    echo
    
    docker-compose -f docker-compose.production.yml exec web python -c "
from app import create_app, db
from app.models.user import User

app = create_app()
with app.app_context():
    if not User.query.filter_by(email='$ADMIN_EMAIL').first():
        admin = User(
            username='admin',
            email='$ADMIN_EMAIL',
            is_administrator=True,
            is_verified=True
        )
        admin.set_password('$ADMIN_PASSWORD')
        db.session.add(admin)
        db.session.commit()
        print('✅ Admin user created')
    else:
        print('ℹ️ Admin user already exists')
"
}

# 10. Post-Deployment Check
post_deployment_check() {
    print_step "Running post-deployment checks..."
    
    sleep 10
    
    # Check if services are running
    SERVICES_RUNNING=$(docker-compose -f $DEPLOY_DIR/docker-compose.production.yml ps --services --filter "status=running" | wc -l)
    SERVICES_TOTAL=$(docker-compose -f $DEPLOY_DIR/docker-compose.production.yml ps --services | wc -l)
    
    echo "Services running: $SERVICES_RUNNING/$SERVICES_TOTAL"
    
    # Test health endpoint
    HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    
    if [ "$HEALTH_STATUS" = "200" ]; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${RED}❌ Health check failed (Status: $HEALTH_STATUS)${NC}"
    fi
    
    # Get server IP
    SERVER_IP=$(curl -s ifconfig.me)
    
    # Display deployment information
    echo -e "\n${GREEN}=======================================${NC}"
    echo -e "${GREEN}🚀 DEPLOYMENT COMPLETE!${NC}"
    echo -e "${GREEN}=======================================${NC}"
    echo ""
    echo -e "${YELLOW}Important Information:${NC}"
    echo "Server IP: $SERVER_IP"
    echo "Domain: $DOMAIN"
    echo "Deployment Directory: $DEPLOY_DIR"
    echo "Backup Directory: $BACKUP_DIR"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Configure DNS for $DOMAIN to point to $SERVER_IP"
    echo "2. Update .env.production with real API keys"
    echo "3. Run SSL certificate: sudo certbot --nginx"
    echo "4. Configure firewall: sudo ufw allow 80,443,22"
    echo ""
    echo -e "${YELLOW}Management Commands:${NC}"
    echo "Start: sudo systemctl start changex-neurix"
    echo "Stop: sudo systemctl stop changex-neurix"
    echo "Status: sudo systemctl status changex-neurix"
    echo "Logs: docker-compose -f $DEPLOY_DIR/docker-compose.production.yml logs -f"
    echo "Backup: $DEPLOY_DIR/backup.sh"
    echo ""
    echo -e "${GREEN}Access URLs:${NC}"
    echo "Application: https://$DOMAIN"
    echo "Admin Panel: https://$DOMAIN/admin"
    echo "API: https://$DOMAIN/api/v2"
    echo ""
}

# Main deployment function
main() {
    print_step "Starting ChangeX Neurix Deployment"
    
    # Ask for domain
    read -p "Enter your domain name (e.g., changexneurix.com): " DOMAIN
    
    echo ""
    echo "Deployment Configuration:"
    echo "Domain: $DOMAIN"
    echo "Deployment Directory: $DEPLOY_DIR"
    echo ""
    
    read -p "Continue with deployment? (y/N): " CONFIRM
    
    if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
    
    # Execute deployment steps
    system_setup
    setup_directories
    generate_ssl
    generate_env
    copy_application
    configure_nginx
    create_backup_script
    create_systemd_service
    initialize_application
    post_deployment_check
    
    echo -e "${GREEN}✨ Deployment completed successfully!${NC}"
}

# Run deployment
main "$@"
