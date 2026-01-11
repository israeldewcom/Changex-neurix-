# config.py - ENHANCED PRODUCTION CONFIGURATION
import os
from datetime import timedelta
from dotenv import load_dotenv
import secrets

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    # Basic Configuration
    APP_NAME = "ChangeX Neurix"
    APP_VERSION = os.environ.get('APP_VERSION', '2.0.0')
    ENV = os.environ.get('FLASK_ENV', 'production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_urlsafe(64)
    SERVER_NAME = os.environ.get('SERVER_NAME', None)
    APPLICATION_ROOT = os.environ.get('APPLICATION_ROOT', '/')
    PREFERRED_URL_SCHEME = os.environ.get('PREFERRED_URL_SCHEME', 'https')
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://' + os.path.join(basedir, 'data', 'changex_neurix.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 300,
        'pool_pre_ping': True,
        'pool_size': 50,
        'max_overflow': 100,
        'pool_timeout': 30,
        'echo': False,
        'isolation_level': 'READ COMMITTED'
    }
    
    # Redis Configuration
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)
    REDIS_URL = f"redis://{REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}/0"
    REDIS_MAX_CONNECTIONS = 100
    
    # Security Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or secrets.token_urlsafe(64)
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    BCRYPT_LOG_ROUNDS = 13
    BCRYPT_HANDLE_LONG_PASSWORDS = True
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600
    WTF_CSRF_SSL_STRICT = True
    PERMANENT_SESSION_LIFETIME = timedelta(days=31)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    REMEMBER_COOKIE_SECURE = True
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_DURATION = timedelta(days=31)
    
    # Rate Limiting Configuration
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_STRATEGY = 'fixed-window-elastic-expiry'
    RATELIMIT_DEFAULT = ['500 per hour', '100 per minute']
    RATELIMIT_APPLICATION = '5000 per hour'
    RATELIMIT_HEADERS_ENABLED = True
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    ALLOWED_EXTENSIONS = {
        'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'json', 'py',
        'js', 'html', 'css', 'md', 'xml', 'yaml', 'yml', 'sql', 'ipynb',
        # Media formats
        'mp4', 'avi', 'mov', 'mkv', 'webm',  # Video
        'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac',  # Audio
        'svg', 'bmp', 'tiff', 'webp'  # Additional image formats
    }
    
    # Cache Configuration
    CACHE_TYPE = 'RedisCache'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_KEY_PREFIX = 'changex_neurix_'
    
    # Email Configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USE_SSL = os.environ.get('MAIL_USE_SSL', 'False').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER') or 'noreply@changexneurix.com'
    MAIL_MAX_EMAILS = None
    MAIL_ASCII_ATTACHMENTS = False
    ADMINS = os.environ.get('ADMINS', 'admin@changexneurix.com').split(',')
    
    # Monitoring & Observability
    SENTRY_DSN = os.environ.get('SENTRY_DSN', '')
    SENTRY_TRACES_SAMPLE_RATE = float(os.environ.get('SENTRY_TRACES_SAMPLE_RATE', '1.0'))
    SENTRY_PROFILES_SAMPLE_RATE = float(os.environ.get('SENTRY_PROFILES_SAMPLE_RATE', '1.0'))
    ENABLE_METRICS = True
    METRICS_PORT = 9090
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # AI Model Configuration
    MODEL_CACHE_SIZE = 50000
    TRAINING_BATCH_SIZE = 32
    MAX_SEQUENCE_LENGTH = 4096
    EMBEDDING_DIMENSION = 768
    ENABLE_MODEL_CACHING = True
    MODEL_CACHE_DIR = os.path.join(basedir, 'model_cache')
    
    # Text Models
    TEXT_MODEL = "microsoft/DialoGPT-medium"
    SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
    TRANSLATION_MODEL = "t5-base"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Code Models
    CODE_MODEL = "microsoft/codebert-base"
    CODE_GENERATION_MODEL = "Salesforce/codegen-350M-mono"
    
    # Vision Models
    IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"
    OBJECT_DETECTION_MODEL = "facebook/detr-resnet-50"
    
    # Audio Models
    SPEECH_TO_TEXT_MODEL = "openai/whisper-base"
    TEXT_TO_SPEECH_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
    
    # Video Models
    VIDEO_GENERATION_ENABLED = True
    MAX_VIDEO_DURATION = 300  # 5 minutes
    VIDEO_FPS = 30
    
    # Automation Settings
    MAX_CONCURRENT_JOBS = 200
    JOB_TIMEOUT = 28800  # 8 hours
    ENABLE_BACKGROUND_TASKS = True
    BACKGROUND_TASK_INTERVAL = 30  # 30 seconds
    
    # Performance Settings
    COMPRESS_LEVEL = 6
    COMPRESS_MIN_SIZE = 500
    THREAD_POOL_SIZE = mp.cpu_count() * 4
    PROCESS_POOL_SIZE = mp.cpu_count() * 2
    GPU_WORKERS = 4
    
    # Payment Gateway Configuration
    STRIPE_PUBLIC_KEY = os.environ.get('STRIPE_PUBLIC_KEY', '')
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
    STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')
    FLUTTERWAVE_PUBLIC_KEY = os.environ.get('FLUTTERWAVE_PUBLIC_KEY', '')
    FLUTTERWAVE_SECRET_KEY = os.environ.get('FLUTTERWAVE_SECRET_KEY', '')
    FLUTTERWAVE_ENCRYPTION_KEY = os.environ.get('FLUTTERWAVE_ENCRYPTION_KEY', '')
    
    # Manual Payment Configuration
    MANUAL_PAYMENT_PHONE = os.environ.get('MANUAL_PAYMENT_PHONE', '+9161806424')
    MANUAL_PAYMENT_NAME = os.environ.get('MANUAL_PAYMENT_NAME', 'Israel Wycliffe')
    MANUAL_PAYMENT_BANK = os.environ.get('MANUAL_PAYMENT_BANK', 'OPay')
    MANUAL_PAYMENT_REQUIRES_APPROVAL = True  # NEW: All manual payments require admin approval
    
    # Affiliate Configuration
    AFFILIATE_COMMISSION_RATE = float(os.environ.get('AFFILIATE_COMMISSION_RATE', '0.15'))  # 15%
    AFFILIATE_COOKIE_DURATION = int(os.environ.get('AFFILIATE_COOKIE_DURATION', '30'))  # 30 days
    MIN_PAYOUT_AMOUNT = float(os.environ.get('MIN_PAYOUT_AMOUNT', '50.00'))
    AFFILIATE_WITHDRAWAL_REQUIRES_APPROVAL = True  # NEW: Withdrawals require admin approval
    
    # Pricing Tiers
    PRICING_TIERS = {
        'free': {
            'price': 0,
            'features': ['basic_ai', '1000_requests', '3_models', '1gb_storage', 'email_support'],
            'limits': {'requests': 1000, 'models': 3, 'storage': 1, 'media_generations': 10},
            'requires_approval': False
        },
        'premium': {
            'price': 49,
            'features': ['advanced_ai', '5000_requests', '8_models', '5gb_storage', 'priority_support', 'image_generation'],
            'limits': {'requests': 5000, 'models': 8, 'storage': 5, 'media_generations': 50},
            'requires_approval': True  # NEW: Requires admin approval
        },
        'pro': {
            'price': 99,
            'features': ['advanced_ai', '10000_requests', '15_models', '10gb_storage', 'priority_support', 'api_access', 'image_generation', 'audio_processing'],
            'limits': {'requests': 10000, 'models': 15, 'storage': 10, 'media_generations': 100},
            'requires_approval': True
        },
        'enterprise': {
            'price': 999,
            'features': ['all_ai', 'unlimited_requests', 'unlimited_models', '100gb_storage', '24_7_support', 'custom_models', 'on_premise', 'video_generation', 'iot_control', 'self_learning'],
            'limits': {'requests': 100000, 'models': 100, 'storage': 100, 'media_generations': 1000},
            'requires_approval': True
        },
        'custom': {
            'price': 0,
            'features': ['fully_custom', 'contact_sales'],
            'limits': {'requests': 0, 'models': 0, 'storage': 0, 'media_generations': 0},
            'requires_approval': True
        }
    }
    
    # Media Processing Configuration
    MEDIA_TEMP_DIR = os.path.join(basedir, 'temp_media')
    MAX_IMAGE_SIZE = (4096, 4096)
    MAX_AUDIO_DURATION = 3600  # 1 hour
    MEDIA_COMPRESSION_QUALITY = 85
    ENABLE_MEDIA_CACHING = True
    MEDIA_CACHE_TTL = 86400  # 24 hours
    
    # IoT Configuration
    IOT_BROKER = os.environ.get('IOT_BROKER', 'localhost')
    IOT_PORT = int(os.environ.get('IOT_PORT', 1883))
    IOT_USERNAME = os.environ.get('IOT_USERNAME', None)
    IOT_PASSWORD = os.environ.get('IOT_PASSWORD', None)
    IOT_TOPIC_PREFIX = 'changex_neurix/iot/'
    IOT_ENABLED = True
    
    # Self-Learning Configuration
    SELF_LEARNING_ENABLED = True
    LEARNING_CYCLE_INTERVAL = 3600  # 1 hour
    FEEDBACK_BUFFER_SIZE = 1000
    MODEL_RETRAINING_THRESHOLD = 0.7  # Retrain if performance drops below 70%
    
    # Advanced Features
    ENABLE_NEUROMORPHIC_COMPUTING = os.environ.get('ENABLE_NEUROMORPHIC_COMPUTING', 'False').lower() == 'true'
    ENABLE_FEDERATED_LEARNING = os.environ.get('ENABLE_FEDERATED_LEARNING', 'True').lower() == 'true'
    ENABLE_EDGE_COMPUTING = os.environ.get('ENABLE_EDGE_COMPUTING', 'True').lower() == 'true'
    ENABLE_DISTRIBUTED_TRAINING = True
    DISTRIBUTED_WORKERS = int(os.environ.get('DISTRIBUTED_WORKERS', 8))
    
    # Advanced Security
    ENABLE_ADVANCED_ENCRYPTION = True
    ENABLE_BIOMETRIC_AUTH = os.environ.get('ENABLE_BIOMETRIC_AUTH', 'False').lower() == 'true'
    ENABLE_2FA = True
    SESSION_ENCRYPTION_KEY = os.environ.get('SESSION_ENCRYPTION_KEY') or secrets.token_urlsafe(32)
    
    # Enterprise Features
    ENABLE_MULTI_TENANCY = os.environ.get('ENABLE_MULTI_TENANCY', 'True').lower() == 'true'
    ENABLE_AUDIT_LOGGING = True
    ENABLE_COMPLIANCE_FRAMEWORK = True
    GDPR_COMPLIANT = os.environ.get('GDPR_COMPLIANT', 'True').lower() == 'true'
    
    # Business Intelligence
    ENABLE_ANALYTICS = True
    ENABLE_BI_REPORTING = True
    ANALYTICS_RETENTION_DAYS = 365
    
    # Workflow Automation
    ENABLE_WORKFLOW_ENGINE = True
    MAX_WORKFLOW_STEPS = 100
    WORKFLOW_TIMEOUT = 3600  # 1 hour
    
    # API Configuration
    API_VERSION = 'v2'
    API_TITLE = 'ChangeX Neurix API v2'
    API_DESCRIPTION = 'Universal AI Platform for Enterprise with Advanced Media Generation'
    API_PREFIX = '/api/v2'
    API_RATE_LIMIT = os.environ.get('API_RATE_LIMIT', '2000 per hour')
    
    # Intelligence Features
    ENABLE_ADVANCED_REASONING = True
    ENABLE_PREDICTIVE_ANALYTICS = True
    ENABLE_REAL_TIME_LEARNING = True
    KNOWLEDGE_GRAPH_ENABLED = True
    
    # WebSocket Configuration
    SOCKETIO_MESSAGE_QUEUE = REDIS_URL
    SOCKETIO_ASYNC_MODE = 'gevent'
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.environ.get('SOCKETIO_CORS_ALLOWED_ORIGINS', '*')
    SOCKETIO_LOGGER = True
    SOCKETIO_ENGINEIO_LOGGER = True
    
    # CDN Configuration
    CDN_DOMAIN = os.environ.get('CDN_DOMAIN', '')
    STATIC_FOLDER = 'static'
    
    # Streaming Configuration
    ENABLE_STREAMING = True
    STREAM_CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    STREAM_TIMEOUT = 300  # 5 minutes
    
    # Timezone
    TIMEZONE = 'UTC'
    
    # Admin Approval Settings
    PREMIUM_APPROVAL_REQUIRED = True
    MANUAL_PAYMENT_APPROVAL_REQUIRED = True
    WITHDRAWAL_APPROVAL_REQUIRED = True
    ADMIN_APPROVAL_NOTIFICATION_EMAIL = True

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    PROPAGATE_EXCEPTIONS = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    
    # Production security
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    
    # Production database
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 300,
        'pool_pre_ping': True,
        'pool_size': 50,
        'max_overflow': 100,
        'pool_timeout': 30,
        'echo': False,
        'isolation_level': 'READ COMMITTED'
    }
    
    # Production media settings
    MEDIA_COMPRESSION_QUALITY = 90
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024  # 5GB for production
    
    # Production rate limiting
    RATELIMIT_DEFAULT = ['200 per hour', '50 per minute']
    RATELIMIT_APPLICATION = '2000 per hour'

class DevelopmentConfig(Config):
    DEBUG = True
    DEVELOPMENT = True
    TESTING = False
    
    # Development settings
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False
    WTF_CSRF_ENABLED = True
    
    # Debug toolbar
    DEBUG_TB_ENABLED = True
    DEBUG_TB_INTERCEPT_REDIRECTS = False
    
    # Development database
    SQLALCHEMY_DATABASE_URI = 'postgresql://' + os.path.join(Config.basedir, 'data', 'changex_neurix_dev.db')
    
    # Development media settings
    MEDIA_COMPRESSION_QUALITY = 70
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB for development
    
    # Development rate limiting
    RATELIMIT_ENABLED = False

class TestingConfig(Config):
    TESTING = True
    DEBUG = False
    
    # Testing database
    SQLALCHEMY_DATABASE_URI = 'postgresql://:memory:'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    # Disable rate limiting for testing
    RATELIMIT_ENABLED = False
    
    # Disable background tasks for testing
    ENABLE_BACKGROUND_TASKS = False
    
    # Disable media processing for testing
    VIDEO_GENERATION_ENABLED = False
    IOT_ENABLED = False
    
    # Disable admin approvals for testing
    PREMIUM_APPROVAL_REQUIRED = False
    MANUAL_PAYMENT_APPROVAL_REQUIRED = False
    WITHDRAWAL_APPROVAL_REQUIRED = False

# Configuration dictionary
config = {
    'production': ProductionConfig,
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'default': ProductionConfig
}
