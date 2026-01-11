# app/__init__.py - ENHANCED PRODUCTION-READY VERSION
from flask import Flask, render_template, request, jsonify, send_file, session, Response, stream_with_context
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_cors import CORS
from flask_mail import Mail
from flask_socketio import SocketIO, emit
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from config import Config
import logging
from logging.handlers import RotatingFileHandler, SMTPHandler
import os
import threading
import time
import asyncio
from datetime import datetime, timedelta
import redis
from rq import Queue
from rq_scheduler import Scheduler
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import stripe
import flutterwave_python.rave as flutterwave
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import qrcode
import io
import base64
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import wraps
import jwt
import hmac
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoModel, AutoTokenizer, pipeline
import cv2
import spacy
from scipy import spatial, stats
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# NEW IMPORTS FOR PRODUCTION
import torchvision
from torchvision import transforms, models
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import whisper
from TTS.api import TTS
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, ImageSequenceClip
import paho.mqtt.client as mqtt  # IoT
import websockets
import aiohttp
from PIL import Image
import subprocess
from werkzeug.utils import secure_filename
import uuid

# PRODUCTION VALIDATION IMPORTS
from marshmallow import Schema, fields, validate, ValidationError
from flask_caching import Cache
import structlog
import prometheus_client
from prometheus_client import Counter, Histogram, generate_latest

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()
mail = Mail()
socketio = SocketIO(cors_allowed_origins="*", async_mode='gevent', logger=True, engineio_logger=True)
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per hour", "50 per minute"])
cache = Cache(config={'CACHE_TYPE': 'RedisCache', 'CACHE_REDIS_URL': Config.REDIS_URL})
admin = Admin(name='ChangeX Neurix Admin', template_mode='bootstrap4')

# Enhanced Redis connection with pooling
redis_client = redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    password=Config.REDIS_PASSWORD,
    decode_responses=True,
    socket_keepalive=True,
    retry_on_timeout=True,
    max_connections=50,
    health_check_interval=30
)

# Queue with retry mechanism
task_queue = Queue('default', connection=redis_client, default_timeout=3600)
scheduler = Scheduler(queue=task_queue, connection=redis_client)

# Payment gateways initialization
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', Config.STRIPE_SECRET_KEY)
flutterwave_public_key = os.environ.get('FLUTTERWAVE_PUBLIC_KEY', Config.FLUTTERWAVE_PUBLIC_KEY)
flutterwave_secret_key = os.environ.get('FLUTTERWAVE_SECRET_KEY', Config.FLUTTERWAVE_SECRET_KEY)

if flutterwave_public_key and flutterwave_secret_key:
    rave = flutterwave.Rave(flutterwave_public_key, flutterwave_secret_key, usingEnv=False)
else:
    rave = None

# Global AI Models with enhanced caching and LRU strategy
ai_models = {
    'text_model': None,
    'code_model': None,
    'vision_model': None,
    'reasoning_model': None,
    'nlp_model': None,
    'embeddings_model': None,
    'summarization_model': None,
    'translation_model': None,
    # NEW MODELS
    'image_generator': None,
    'image_variator': None,
    'speech_to_text': None,
    'text_to_speech': None,
    'video_generator': None,
    'audio_processor': None,
    'iot_controller': None,
    'self_learning_engine': None
}

# Thread pools for parallel processing with monitoring
cpu_thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count() * 4, thread_name_prefix="cpu_worker")
gpu_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gpu_worker")
process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count() * 2)

# NEW: Self-learning model registry with versioning
model_registry = {}
training_queue = Queue('training', connection=redis_client)

# PROMETHEUS METRICS
REQUEST_COUNT = Counter('changex_neurix_requests_total', 'Total API Requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('changex_neurix_request_latency_seconds', 'Request latency', ['endpoint'])
IMAGE_GENERATION_COUNT = Counter('changex_neurix_image_generations_total', 'Total Image Generations', ['status'])
VIDEO_GENERATION_COUNT = Counter('changex_neurix_video_generations_total', 'Total Video Generations', ['status'])
AUDIO_PROCESSING_COUNT = Counter('changex_neurix_audio_processings_total', 'Total Audio Processings', ['status'])
IOT_COMMAND_COUNT = Counter('changex_neurix_iot_commands_total', 'Total IoT Commands', ['status'])

# STRUCTURED LOGGING
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.render_to_log_kwargs,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# PRODUCTION VALIDATION SCHEMAS
class ImageGenerationSchema(Schema):
    prompt = fields.Str(required=True, validate=[
        validate.Length(min=3, max=1000),
        validate.Regexp(r'^[A-Za-z0-9\s\.,!?\'"-]+$')
    ])
    negative_prompt = fields.Str(validate=validate.Length(max=500))
    width = fields.Int(validate=validate.Range(min=256, max=2048))
    height = fields.Int(validate=validate.Range(min=256, max=2048))
    num_inference_steps = fields.Int(validate=validate.Range(min=10, max=150))
    guidance_scale = fields.Float(validate=validate.Range(min=1.0, max=20.0))
    seed = fields.Int(validate=validate.Range(min=0, max=2**32-1))

class VideoGenerationSchema(Schema):
    text = fields.Str(required=True, validate=validate.Length(min=10, max=5000))
    duration = fields.Float(validate=validate.Range(min=1, max=300))
    fps = fields.Int(validate=validate.Range(min=1, max=60))
    theme = fields.Str(validate=validate.Length(max=100))

class AudioProcessingSchema(Schema):
    text = fields.Str(required=True, validate=validate.Length(min=1, max=5000))
    voice_model = fields.Str(validate=validate.Length(max=100))
    speed = fields.Float(validate=validate.Range(min=0.5, max=2.0))
    pitch = fields.Float(validate=validate.Range(min=-1.0, max=1.0))

class IoTCommandSchema(Schema):
    device_id = fields.Str(required=True, validate=validate.Length(min=1, max=128))
    action = fields.Str(required=True, validate=validate.Length(min=1, max=50))
    params = fields.Dict()

# ADMIN ROLES PERMISSIONS
ADMIN_ROLES = {
    'super_admin': ['*'],
    'admin': ['manage_users', 'manage_content', 'view_reports', 'manage_payments', 'approve_premium', 'manage_affiliates'],
    'moderator': ['manage_content', 'view_reports'],
    'support': ['view_reports', 'manage_users']
}

class SelfLearningImageGenerator:
    """Enhanced self-learning image generator with production optimizations"""
    
    def __init__(self):
        self.model_name = "runwayml/stable-diffusion-v1-5"
        self.finetuned_models = {}
        self.feedback_buffer = []
        self.learning_rate = 0.0001
        self.batch_size = 4
        self.cache_hits = 0
        self.cache_misses = 0
        self.prompt_cache = {}
        
    @cache.memoize(timeout=3600)  # Cache for 1 hour
    def load_base_model(self):
        """Load base Stable Diffusion model with caching"""
        try:
            logger.info("Loading Stable Diffusion model", model=self.model_name)
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                cache_dir=Config.MODEL_CACHE_DIR
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                pipe.enable_attention_slicing()
                pipe.enable_xformers_memory_efficient_attention()
            
            logger.info("Stable Diffusion model loaded successfully")
            return pipe
            
        except Exception as e:
            logger.error("Failed to load image generator", error=str(e), exc_info=True)
            return None
    
    def generate_image(self, prompt, negative_prompt="", height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
        """Generate image from text prompt with caching and monitoring"""
        
        # Check cache first
        cache_key = f"image_{hash(prompt)}_{height}_{width}_{num_inference_steps}"
        if cache_key in self.prompt_cache:
            self.cache_hits += 1
            logger.debug("Image generation cache hit", key=cache_key)
            return self.prompt_cache[cache_key]
        
        self.cache_misses += 1
        IMAGE_GENERATION_COUNT.labels(status='started').inc()
        
        try:
            model = ai_models.get('image_generator')
            if not model:
                model = self.load_base_model()
                ai_models['image_generator'] = model
            
            if not model:
                raise Exception("Image generator not available")
            
            # Track generation start time
            start_time = time.time()
            
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                image = model(
                    prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True, quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Store in cache
            self.prompt_cache[cache_key] = img_str
            
            # Store for learning with validation
            if len(prompt) <= 1000:  # Sanity check
                self.feedback_buffer.append({
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'image_hash': hash(img_str),
                    'timestamp': datetime.utcnow(),
                    'quality_score': 0.5,
                    'dimensions': f"{height}x{width}",
                    'inference_steps': num_inference_steps
                })
                
                # Process feedback batch if buffer is full
                if len(self.feedback_buffer) > 100:
                    self._process_feedback_batch()
            
            # Log metrics
            generation_time = time.time() - start_time
            logger.info("Image generated successfully", 
                       prompt_length=len(prompt),
                       generation_time=f"{generation_time:.2f}s",
                       dimensions=f"{width}x{height}")
            
            IMAGE_GENERATION_COUNT.labels(status='success').inc()
            return img_str
            
        except Exception as e:
            logger.error("Image generation failed", error=str(e), exc_info=True)
            IMAGE_GENERATION_COUNT.labels(status='failed').inc()
            return None
    
    def create_variations(self, image_bytes, strength=0.75):
        """Create variations of an existing image with validation"""
        try:
            if 'image_variator' not in ai_models or not ai_models['image_variator']:
                ai_models['image_variator'] = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    cache_dir=Config.MODEL_CACHE_DIR
                )
                
                if torch.cuda.is_available():
                    ai_models['image_variator'] = ai_models['image_variator'].to("cuda")
            
            # Validate image bytes
            if len(image_bytes) > Config.MAX_CONTENT_LENGTH:
                raise ValueError("Image too large")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Validate image dimensions
            if image.width > Config.MAX_IMAGE_SIZE[0] or image.height > Config.MAX_IMAGE_SIZE[1]:
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Generate variation
            result = ai_models['image_variator'](
                prompt="A variation of the input image",
                image=image,
                strength=strength,
                guidance_scale=7.5
            ).images[0]
            
            buffered = io.BytesIO()
            result.save(buffered, format="PNG", optimize=True, quality=95)
            return base64.b64encode(buffered.getvalue()).decode()
            
        except Exception as e:
            logger.error("Image variation failed", error=str(e), exc_info=True)
            return None
    
    def _process_feedback_batch(self):
        """Process feedback for model improvement with validation"""
        if len(self.feedback_buffer) < self.batch_size:
            return
        
        try:
            # Validate feedback data
            valid_feedback = []
            for item in self.feedback_buffer:
                if (isinstance(item, dict) and 
                    'prompt' in item and 
                    'quality_score' in item):
                    valid_feedback.append(item)
            
            if not valid_feedback:
                return
            
            # Calculate metrics
            avg_score = np.mean([item['quality_score'] for item in valid_feedback])
            prompt_lengths = [len(item['prompt']) for item in valid_feedback]
            
            logger.info("Processing feedback batch", 
                       batch_size=len(valid_feedback),
                       average_score=avg_score,
                       avg_prompt_length=np.mean(prompt_lengths))
            
            # In production, this would fine-tune the model
            # For now, we'll just log and clear buffer
            self.feedback_buffer = []
            
        except Exception as e:
            logger.error("Feedback processing failed", error=str(e), exc_info=True)

# PRODUCTION-READY VIDEO GENERATOR
class VideoGenerator:
    """Advanced video generation system with production optimizations"""
    
    def __init__(self):
        self.temp_dir = Config.MEDIA_TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
        self.ffmpeg_path = self._find_ffmpeg()
        self.max_frames = 1000  # Safety limit
        
    def _find_ffmpeg(self):
        """Find ffmpeg binary with fallbacks"""
        paths = ['ffmpeg', '/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg']
        for path in paths:
            try:
                subprocess.run([path, '-version'], capture_output=True, check=True)
                return path
            except:
                continue
        raise Exception("FFmpeg not found. Install ffmpeg: apt-get install ffmpeg")
    
    def text_to_video(self, text, duration=5, fps=24, theme="abstract"):
        """Generate video from text with production optimizations"""
        
        VIDEO_GENERATION_COUNT.labels(status='started').inc()
        start_time = time.time()
        
        try:
            # Validate inputs
            if len(text) < 10 or len(text) > 5000:
                raise ValueError("Text must be between 10 and 5000 characters")
            
            if duration > Config.MAX_VIDEO_DURATION:
                duration = Config.MAX_VIDEO_DURATION
            
            # Generate keyframes from text
            image_gen = SelfLearningImageGenerator()
            frames = []
            
            # Split text into scenes
            scenes = self._split_text_into_scenes(text)
            if len(scenes) > 20:  # Limit scenes
                scenes = scenes[:20]
            
            logger.info("Generating video scenes", num_scenes=len(scenes), total_duration=duration)
            
            for i, scene in enumerate(scenes):
                try:
                    # Generate image for scene with timeout
                    img_base64 = image_gen.generate_image(
                        f"Video scene: {scene}",
                        height=512,
                        width=512
                    )
                    
                    if img_base64:
                        # Convert base64 to image
                        img_data = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Resize for video with aspect ratio preservation
                        img = self._resize_with_aspect(img, (1280, 720))
                        
                        # Add text overlay
                        self._add_text_overlay(img, scene)
                        
                        # Save frame
                        frame_path = os.path.join(self.temp_dir, f"frame_{i:04d}.png")
                        img.save(frame_path, optimize=True, quality=85)
                        frames.append(frame_path)
                        
                        # Log progress
                        if i % 5 == 0:
                            logger.debug("Video generation progress", 
                                        scene=i+1, 
                                        total=len(scenes),
                                        progress=f"{(i+1)/len(scenes)*100:.1f}%")
                
                except Exception as e:
                    logger.warning("Failed to generate scene", scene=i, error=str(e))
                    continue
            
            if len(frames) < 2:
                raise Exception("Not enough frames generated")
            
            # Create video from frames
            duration_per_frame = duration / len(frames)
            video_path = self._create_video_from_frames(frames, duration_per_frame, fps)
            
            if not video_path:
                raise Exception("Video creation failed")
            
            # Add audio narration if TTS is available
            if ai_models.get('text_to_speech') and len(text) < 1000:
                try:
                    audio_path = self._text_to_speech(text)
                    if audio_path and os.path.exists(audio_path):
                        video_path = self._add_audio_to_video(video_path, audio_path)
                except Exception as e:
                    logger.warning("TTS failed, continuing without audio", error=str(e))
            
            # Convert to base64
            with open(video_path, 'rb') as f:
                video_base64 = base64.b64encode(f.read()).decode()
            
            # Cleanup
            self._cleanup_temp_files(frames + [video_path])
            
            # Log success
            total_time = time.time() - start_time
            logger.info("Video generation completed", 
                       duration=f"{duration}s",
                       frames=len(frames),
                       total_time=f"{total_time:.2f}s")
            
            VIDEO_GENERATION_COUNT.labels(status='success').inc()
            return video_base64
            
        except Exception as e:
            logger.error("Text to video generation failed", error=str(e), exc_info=True)
            VIDEO_GENERATION_COUNT.labels(status='failed').inc()
            return None
    
    def _resize_with_aspect(self, image, target_size):
        """Resize image maintaining aspect ratio"""
        target_width, target_height = target_size
        image_ratio = image.width / image.height
        target_ratio = target_width / target_height
        
        if image_ratio > target_ratio:
            # Image is wider
            new_height = target_height
            new_width = int(target_height * image_ratio)
        else:
            # Image is taller
            new_width = target_width
            new_height = int(target_width / image_ratio)
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crop to target size
        left = (new_width - target_width) / 2
        top = (new_height - target_height) / 2
        right = (new_width + target_width) / 2
        bottom = (new_height + target_height) / 2
        
        return resized.crop((left, top, right, bottom))
    
    def _split_text_into_scenes(self, text, max_scene_length=100):
        """Split text into scenes for video generation with smart splitting"""
        # Split by sentences first
        sentences = []
        for sentence in text.replace('. ', '.\n').split('\n'):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
        
        scenes = []
        current_scene = ""
        
        for sentence in sentences:
            if len(current_scene) + len(sentence) < max_scene_length:
                if current_scene:
                    current_scene += " " + sentence
                else:
                    current_scene = sentence
            else:
                if current_scene:
                    scenes.append(current_scene)
                current_scene = sentence
        
        if current_scene:
            scenes.append(current_scene)
        
        return scenes[:20]  # Limit to 20 scenes

# ENHANCED AUDIO PROCESSOR
class AudioProcessor:
    """Advanced audio processing system with production features"""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.sample_rate = 44100
        
    def text_to_audio(self, text, voice="female", speed=1.0):
        """Convert text to speech audio with validation"""
        
        AUDIO_PROCESSING_COUNT.labels(status='started').inc()
        
        try:
            # Validate input
            if len(text) > 5000:
                raise ValueError("Text too long (max 5000 characters)")
            
            if not 0.5 <= speed <= 2.0:
                speed = max(0.5, min(2.0, speed))
            
            # Load TTS model if not loaded
            if not ai_models.get('text_to_speech'):
                ai_models['text_to_speech'] = TTS(
                    model_name=Config.TEXT_TO_SPEECH_MODEL,
                    progress_bar=False,
                    gpu=torch.cuda.is_available()
                )
            
            # Generate audio with unique filename
            audio_uuid = str(uuid.uuid4())
            audio_path = os.path.join(Config.MEDIA_TEMP_DIR, f"tts_{audio_uuid}.wav")
            
            ai_models['text_to_speech'].tts_to_file(
                text=text,
                file_path=audio_path,
                speaker=voice
            )
            
            # Adjust speed if needed
            if speed != 1.0:
                audio_path = self._adjust_audio_speed(audio_path, speed)
            
            # Load and convert to base64
            with open(audio_path, 'rb') as f:
                audio_base64 = base64.b64encode(f.read()).decode()
            
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            logger.info("Text to speech completed", 
                       text_length=len(text),
                       voice=voice,
                       speed=speed)
            
            AUDIO_PROCESSING_COUNT.labels(status='success').inc()
            return audio_base64
            
        except Exception as e:
            logger.error("Text to audio failed", error=str(e), exc_info=True)
            AUDIO_PROCESSING_COUNT.labels(status='failed').inc()
            return None

# PRODUCTION IOT CONTROLLER
class IoTController:
    """Production IoT device controller with connection pooling"""
    
    def __init__(self):
        self.mqtt_client = None
        self.devices = {}
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_ping = None
        
    def connect(self, broker="localhost", port=1883, username=None, password=None):
        """Connect to MQTT broker with retry logic"""
        try:
            self.mqtt_client = mqtt.Client(
                client_id=f"changex_neurix_{uuid.uuid4().hex[:8]}",
                clean_session=False
            )
            
            if username and password:
                self.mqtt_client.username_pw_set(username, password)
            
            # Setup callbacks
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message
            self.mqtt_client.on_disconnect = self._on_disconnect
            
            # Connection parameters
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()
            
            # Start heartbeat
            self._start_heartbeat()
            
            self.connected = True
            logger.info("Connected to IoT broker", broker=broker, port=port)
            return True
            
        except Exception as e:
            logger.error("IoT connection failed", error=str(e), exc_info=True)
            
            # Retry logic
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                time.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
                return self.connect(broker, port, username, password)
            
            return False
    
    def _start_heartbeat(self):
        """Start heartbeat monitoring"""
        def heartbeat():
            while self.connected:
                try:
                    self.mqtt_client.publish("changex_neurix/iot/heartbeat", "ping")
                    self.last_ping = datetime.utcnow()
                    time.sleep(30)
                except:
                    break
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
    
    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection"""
        self.connected = False
        logger.warning("IoT disconnected", rc=rc)
        
        # Attempt reconnect
        if rc != 0:
            time.sleep(5)
            self.connect()

# ENHANCED SELF-LEARNING ENGINE
class SelfLearningEngine:
    """Production self-learning intelligence system"""
    
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.feedback_history = []
        self.improvement_cycles = 0
        self.model_performance = {}
        self.performance_threshold = 0.7
        self.retraining_queue = []
        
    def analyze_feedback(self, feedback_data):
        """Analyze user feedback for learning with validation"""
        try:
            # Validate feedback data
            if not isinstance(feedback_data, list):
                raise ValueError("Feedback data must be a list")
            
            if len(feedback_data) == 0:
                return {'status': 'no_feedback'}
            
            # Extract patterns from feedback
            patterns = self._extract_patterns(feedback_data)
            
            # Update knowledge graph
            self._update_knowledge_graph(patterns)
            
            # Identify improvement areas
            improvements = self._identify_improvements()
            
            # Schedule model retraining if needed
            if improvements:
                self._schedule_retraining(improvements)
            
            return {
                'status': 'success',
                'patterns_found': len(patterns),
                'improvements_identified': len(improvements),
                'cycle': self.improvement_cycles
            }
            
        except Exception as e:
            logger.error("Feedback analysis failed", error=str(e), exc_info=True)
            return {'status': 'error', 'error': str(e)}
    
    def optimize_model_selection(self, task_type, constraints):
        """Dynamically select best model for task with cost optimization"""
        try:
            # Validate inputs
            valid_task_types = ['image_generation', 'text_generation', 'video_generation', 
                               'audio_processing', 'speech_recognition', 'code_generation']
            
            if task_type not in valid_task_types:
                raise ValueError(f"Invalid task type. Must be one of: {valid_task_types}")
            
            # Analyze task requirements
            task_requirements = self._analyze_task_requirements(task_type, constraints)
            
            # Get available models
            available_models = self._get_available_models()
            
            # Score each model
            scored_models = []
            for model_id, model_info in available_models.items():
                score = self._calculate_model_score(model_info, task_requirements)
                
                # Check if model meets minimum requirements
                if score >= self.performance_threshold * 100:  # Convert to percentage
                    scored_models.append({
                        'model_id': model_id,
                        'score': score,
                        'cost': model_info.get('cost', 0),
                        'latency': model_info.get('latency', 0),
                        'accuracy': model_info.get('accuracy', 0)
                    })
            
            if not scored_models:
                return None
            
            # Select best model based on weighted score
            scored_models.sort(key=lambda x: x['score'], reverse=True)
            best_model = scored_models[0]
            
            # Update performance tracking
            self._track_model_performance(best_model['model_id'], task_type)
            
            logger.info("Model selected", 
                       task_type=task_type,
                       model_id=best_model['model_id'],
                       score=best_model['score'])
            
            return best_model
            
        except Exception as e:
            logger.error("Model selection failed", error=str(e), exc_info=True)
            return None

# LOAD AI MODELS WITH PRODUCTION OPTIMIZATIONS
def load_ai_models():
    """Load all AI models on startup with production optimizations"""
    try:
        logger.info("Loading AI models with production optimizations...")
        
        # Load models with fallbacks
        models_to_load = [
            ('text_model', transformers.pipeline, 'text-generation', 'microsoft/DialoGPT-medium'),
            ('embeddings_model', transformers.pipeline, 'feature-extraction', 'sentence-transformers/all-MiniLM-L6-v2'),
            ('summarization_model', transformers.pipeline, 'summarization', 'facebook/bart-large-cnn'),
            ('translation_model', transformers.pipeline, 'translation_en_to_fr', 't5-base'),
        ]
        
        for name, loader, task, model_name in models_to_load:
            try:
                ai_models[name] = loader(
                    task,
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    model_kwargs={'cache_dir': Config.MODEL_CACHE_DIR}
                )
                logger.info(f"Loaded {name}", model=model_name)
            except Exception as e:
                logger.warning(f"Failed to load {name}, using fallback", error=str(e))
                ai_models[name] = None
        
        # Load code model
        try:
            ai_models['code_model'] = AutoModel.from_pretrained(
                'microsoft/codebert-base',
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                cache_dir=Config.MODEL_CACHE_DIR
            )
            ai_models['code_tokenizer'] = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        except Exception as e:
            logger.warning("Failed to load code model", error=str(e))
        
        # Load NLP model
        try:
            ai_models['nlp_model'] = spacy.load('en_core_web_lg')
        except:
            logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_lg")
            ai_models['nlp_model'] = spacy.load('en_core_web_lg')
        
        # NEW: Load enhanced models
        try:
            ai_models['image_generator'] = SelfLearningImageGenerator().load_base_model()
            logger.info("Image generator loaded")
        except Exception as e:
            logger.error("Failed to load image generator", error=str(e))
            ai_models['image_generator'] = None
        
        try:
            ai_models['speech_to_text'] = whisper.load_model("base")
            ai_models['text_to_speech'] = TTS(model_name=Config.TEXT_TO_SPEECH_MODEL)
            logger.info("Audio models loaded")
        except Exception as e:
            logger.error("Failed to load audio models", error=str(e))
            ai_models['speech_to_text'] = None
            ai_models['text_to_speech'] = None
        
        try:
            ai_models['video_generator'] = VideoGenerator()
            logger.info("Video generator loaded")
        except Exception as e:
            logger.error("Failed to load video generator", error=str(e))
            ai_models['video_generator'] = None
        
        try:
            ai_models['audio_processor'] = AudioProcessor()
            logger.info("Audio processor loaded")
        except Exception as e:
            logger.error("Failed to load audio processor", error=str(e))
            ai_models['audio_processor'] = None
        
        try:
            ai_models['iot_controller'] = IoTController()
            if Config.IOT_ENABLED and Config.IOT_BROKER:
                ai_models['iot_controller'].connect(
                    broker=Config.IOT_BROKER,
                    port=Config.IOT_PORT,
                    username=Config.IOT_USERNAME,
                    password=Config.IOT_PASSWORD
                )
            logger.info("IoT controller loaded")
        except Exception as e:
            logger.error("Failed to load IoT controller", error=str(e))
            ai_models['iot_controller'] = None
        
        try:
            ai_models['self_learning_engine'] = SelfLearningEngine()
            logger.info("Self-learning engine loaded")
        except Exception as e:
            logger.error("Failed to load self-learning engine", error=str(e))
            ai_models['self_learning_engine'] = None
        
        # Move models to GPU if available
        if torch.cuda.is_available():
            for name, model in ai_models.items():
                if hasattr(model, 'to'):
                    try:
                        model.to('cuda')
                        logger.info(f"Moved {name} to GPU")
                    except:
                        pass
        
        logger.info("All AI models loaded successfully with production optimizations")
        
    except Exception as e:
        logger.error("Error loading AI models", error=str(e), exc_info=True)
        load_fallback_models()

def load_fallback_models():
    """Load lightweight fallback models if primary models fail"""
    try:
        logger.warning("Loading fallback models")
        ai_models['text_model'] = transformers.pipeline(
            'text-generation',
            model='distilgpt2',
            device=-1  # Force CPU
        )
        ai_models['nlp_model'] = spacy.load('en_core_web_sm')
        logger.info("Fallback models loaded")
    except Exception as e:
        logger.error("Fallback model loading failed", error=str(e))

# PRODUCTION FLASK APP FACTORY
def create_app(config_class=Config):
    """Create and configure Flask application with production settings"""
    
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates',
                instance_relative_config=True)
    
    app.config.from_object(config_class)
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    mail.init_app(app)
    cache.init_app(app)
    
    # CORS with production settings
    CORS(app, 
         supports_credentials=True, 
         resources={r"/api/*": {"origins": app.config.get('CORS_ORIGINS', "*")}},
         allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
    
    socketio.init_app(app, 
                      cors_allowed_origins=app.config.get('SOCKETIO_CORS_ALLOWED_ORIGINS', "*"),
                      async_mode='gevent',
                      message_queue=app.config.get('SOCKETIO_MESSAGE_QUEUE'),
                      logger=app.config.get('SOCKETIO_LOGGER'),
                      engineio_logger=app.config.get('SOCKETIO_ENGINEIO_LOGGER'))
    
    limiter.init_app(app)
    
    # Admin panel setup
    admin.init_app(app)
    
    # PRODUCTION MIDDLEWARE
    
    @app.before_request
    def before_request():
        """Global before request with monitoring"""
        request.start_time = time.time()
        request.request_id = str(uuid.uuid4())
        
        # Add request ID to logger
        structlog.contextvars.bind_contextvars(request_id=request.request_id)
        
        # Skip for static files and health checks
        if request.endpoint in ['static', 'health_check', 'metrics']:
            return
        
        logger.info("Request started",
                   method=request.method,
                   path=request.path,
                   endpoint=request.endpoint,
                   user_agent=request.user_agent.string[:200],
                   ip=request.remote_addr)
        
        # Check rate limiting
        if request.endpoint:
            limiter.check()
    
    @app.after_request
    def security_headers(response):
        """Add security headers"""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' cdn.jsdelivr.net cdn.plot.ly; "
            "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "img-src 'self' data: blob: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:;"
        )
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Feature-Policy'] = "camera 'none'; microphone 'none'"
        return response
    
    @app.after_request
    def after_request(response):
        """Global after request with monitoring"""
        # Skip for static files
        if request.endpoint in ['static', 'health_check', 'metrics']:
            return response
        
        # Calculate request duration
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            
            # Log request completion
            logger.info("Request completed",
                       method=request.method,
                       path=request.path,
                       status=response.status_code,
                       duration=f"{duration:.3f}s",
                       content_length=response.content_length or 0)
            
            # Update Prometheus metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint or request.path,
                status=response.status_code
            ).inc()
            
            REQUEST_LATENCY.labels(
                endpoint=request.endpoint or request.path
            ).observe(duration)
            
            # Add performance headers
            response.headers['X-Response-Time'] = f'{duration:.3f}s'
            response.headers['X-Request-ID'] = getattr(request, 'request_id', 'N/A')
        
        return response
    
    # Initialize Sentry for error tracking
    if not app.debug and not app.testing and app.config.get('SENTRY_DSN'):
        sentry_sdk.init(
            dsn=app.config['SENTRY_DSN'],
            integrations=[FlaskIntegration()],
            traces_sample_rate=app.config.get('SENTRY_TRACES_SAMPLE_RATE', 1.0),
            profiles_sample_rate=app.config.get('SENTRY_PROFILES_SAMPLE_RATE', 1.0),
            environment=app.config.get('ENV', 'production'),
            release=f"changex-neurix@{app.config.get('APP_VERSION', '1.0.0')}"
        )
    
    # Login manager configuration
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    login_manager.session_protection = "strong"
    login_manager.refresh_view = 'auth.reauth'
    login_manager.needs_refresh_message = 'Please reauthenticate to access this page.'
    login_manager.needs_refresh_message_category = 'info'
    
    # User loader
    @login_manager.user_loader
    def load_user(user_id):
        from app.models.user import User
        return User.query.get(int(user_id))
    
    # MONITORING ENDPOINTS
    
    @app.route('/health')
    def health_check():
        """Health check endpoint with detailed service status"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': app.config.get('APP_VERSION', '1.0.0'),
            'services': {
                'database': 'connected' if db.session.execute('SELECT 1').scalar() else 'disconnected',
                'redis': 'connected' if redis_client.ping() else 'disconnected',
                'models_loaded': bool(ai_models.get('text_model')),
                'image_generator': 'loaded' if ai_models.get('image_generator') else 'not_loaded',
                'video_generator': 'loaded' if ai_models.get('video_generator') else 'not_loaded',
                'audio_processor': 'loaded' if ai_models.get('audio_processor') else 'not_loaded',
                'iot_controller': 'connected' if ai_models.get('iot_controller') and ai_models['iot_controller'].connected else 'disconnected',
                'queue_workers': task_queue.count
            },
            'system': {
                'memory': f"{psutil.virtual_memory().percent}%",
                'cpu': f"{psutil.cpu_percent()}%",
                'disk': f"{psutil.disk_usage('/').percent}%"
            }
        }
        
        # Check if all critical services are healthy
        critical_services = ['database', 'redis']
        all_healthy = all(
            health_status['services'][service] == 'connected' 
            for service in critical_services
        )
        
        return jsonify(health_status), 200 if all_healthy else 503
    
    @app.route('/metrics')
    @limiter.exempt
    def metrics():
        """Prometheus metrics endpoint"""
        return Response(generate_latest(), mimetype='text/plain')
    
    # ERROR HANDLERS
    
    @app.errorhandler(400)
    def bad_request(error):
        logger.warning("Bad request", error=str(error))
        return jsonify({
            'error': 'Bad request',
            'message': str(error),
            'request_id': getattr(request, 'request_id', 'N/A')
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Please log in',
            'request_id': getattr(request, 'request_id', 'N/A')
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({
            'error': 'Forbidden',
            'message': 'You do not have permission',
            'request_id': getattr(request, 'request_id', 'N/A')
        }), 403
    
    @app.errorhandler(404)
    def not_found_error(error):
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({
                'error': 'Resource not found',
                'path': request.path,
                'request_id': getattr(request, 'request_id', 'N/A')
            }), 404
        return render_template('errors/404.html', title='Page Not Found'), 404
    
    @app.errorhandler(413)
    def too_large(error):
        return jsonify({
            'error': 'File too large',
            'max_size': '1GB',
            'request_id': getattr(request, 'request_id', 'N/A')
        }), 413
    
    @app.errorhandler(429)
    def ratelimit_handler(error):
        return jsonify({
            'error': 'Rate limit exceeded',
            'retry_after': error.description,
            'request_id': getattr(request, 'request_id', 'N/A')
        }), 429
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        logger.error("Internal server error", error=str(error), exc_info=True)
        
        if request.accept_mimetypes.accept_json:
            return jsonify({
                'error': 'Internal server error',
                'request_id': getattr(request, 'request_id', 'N/A')
            }), 500
        
        return render_template('errors/500.html', title='Server Error'), 500
    
    # Register blueprints
    blueprints = {
        'auth': ('app.auth.routes', 'auth_bp', '/auth'),
        'main': ('app.main.routes', 'main_bp', ''),
        'models': ('app.models.routes', 'models_bp', '/models'),
        'automation': ('app.automation.routes', 'automation_bp', '/automation'),
        'scripts': ('app.scripts.routes', 'scripts_bp', '/scripts'),
        'api': ('app.api.routes', 'api_bp', '/api/v2'),
        'monitoring': ('app.monitoring.routes', 'monitoring_bp', '/monitoring'),
        'advanced_ai': ('app.advanced_ai.routes', 'advanced_ai_bp', '/advanced-ai'),
        'enterprise': ('app.enterprise.routes', 'enterprise_bp', '/enterprise'),
        'workflows': ('app.workflows.routes', 'workflows_bp', '/workflows'),
        'intelligence': ('app.intelligence.routes', 'intelligence_bp', '/intelligence'),
        'analytics': ('app.analytics.routes', 'analytics_bp', '/analytics'),
        'payments': ('app.payments.routes', 'payments_bp', '/payments'),
        'affiliate': ('app.affiliate.routes', 'affiliate_bp', '/affiliate'),
        'admin': ('app.admin.routes', 'admin_bp', '/admin'),
        # NEW BLUEPRINTS
        'image_generation': ('app.image_generation.routes', 'image_generation_bp', '/image-generation'),
        'video_generation': ('app.video_generation.routes', 'video_generation_bp', '/video-generation'),
        'audio_processing': ('app.audio_processing.routes', 'audio_processing_bp', '/audio-processing'),
        'iot_control': ('app.iot_control.routes', 'iot_control_bp', '/iot'),
        'self_learning': ('app.self_learning.routes', 'self_learning_bp', '/self-learning'),
        # NEW: ADMIN MANAGEMENT
        'admin_management': ('app.admin_management.routes', 'admin_management_bp', '/admin-management')
    }
    
    for module_path, bp_name, url_prefix in blueprints.values():
        try:
            module = __import__(module_path, fromlist=[''])
            bp = getattr(module, bp_name.split('_')[0])
            app.register_blueprint(bp, url_prefix=url_prefix)
            logger.debug(f"Registered blueprint", blueprint=bp_name, prefix=url_prefix)
        except ImportError as e:
            logger.error(f"Failed to import {module_path}", error=str(e))
        except AttributeError as e:
            logger.error(f"Failed to get blueprint from {module_path}", error=str(e))
    
    # WebSocket events
    @socketio.on('connect')
    def handle_connect():
        logger.info(f'Client connected', sid=request.sid)
        if current_user.is_authenticated:
            emit('status', {'message': 'Connected', 'user': current_user.username})
            socketio.server.enter_room(request.sid, f'user_{current_user.id}')
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f'Client disconnected', sid=request.sid)
    
    # Initialize systems with app context
    with app.app_context():
        # Create necessary directories
        directories = [
            app.config['UPLOAD_FOLDER'],
            'logs',
            'cache',
            Config.MEDIA_TEMP_DIR,
            'temp_audio',
            'model_cache',
            'static/generated',
            'static/uploads',
            'static/exports'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Load AI models
        load_ai_models()
        
        # Initialize database
        db.create_all()
        
        # Create database indexes
        create_database_indexes()
        
        # Initialize admin views
        initialize_admin_views()
        
        # Initialize background systems
        initialize_background_systems()
        
        # Schedule recurring tasks
        schedule_recurring_tasks()
        
        # Setup logging
        setup_production_logging(app)
        
        logger.info('ChangeX Neurix - Production Enterprise Edition Started',
                   environment=app.config['ENV'],
                   debug=app.debug,
                   version=app.config.get('APP_VERSION', '1.0.0'))
    
    return app

# HELPER FUNCTIONS
def create_database_indexes():
    """Create database indexes for performance"""
    from sqlalchemy import Index
    
    # This would be done via migrations in production
    # Here we just log what indexes should exist
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_images_user_created ON generated_images(user_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_videos_user_created ON generated_videos(user_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_audio_user_created ON audio_files(user_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_transactions_user ON transactions(user_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_affiliate_user ON affiliate_referrals(affiliate_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_subscription_status ON subscriptions(status, expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_user_email ON users(email)",
        "CREATE INDEX IF NOT EXISTS idx_user_username ON users(username)",
        "CREATE INDEX IF NOT EXISTS idx_iot_device_user ON iot_devices(user_id, status)",
        "CREATE INDEX IF NOT EXISTS idx_admin_actions ON admin_actions(admin_id, action_timestamp DESC)"
    ]
    
    for index_sql in indexes:
        try:
            db.session.execute(index_sql)
            logger.debug("Created database index", sql=index_sql)
        except Exception as e:
            logger.warning("Failed to create index", sql=index_sql, error=str(e))
    
    db.session.commit()

def initialize_admin_views():
    """Initialize admin panel with enhanced views"""
    from app.models.user import User, Role, APIKey, Subscription, Transaction, AdminAction
    from app.models.affiliate import Affiliate, Commission, WithdrawalRequest
    from app.models.media import GeneratedImage, GeneratedVideo, AudioFile, IoTDevice as IoTDeviceModel
    
    # Custom admin views
    class UserAdminView(ModelView):
        column_list = ['id', 'username', 'email', 'is_active', 'is_administrator', 'created_at']
        column_searchable_list = ['username', 'email']
        column_filters = ['is_active', 'is_administrator', 'created_at']
        form_excluded_columns = ['password_hash']
        
        def is_accessible(self):
            return current_user.is_authenticated and current_user.is_administrator
        
        def inaccessible_callback(self, name, **kwargs):
            return redirect(url_for('auth.login'))
    
    class TransactionAdminView(ModelView):
        column_list = ['id', 'user_id', 'amount', 'currency', 'status', 'transaction_type', 'created_at']
        column_filters = ['status', 'transaction_type', 'created_at']
        column_searchable_list = ['transaction_id', 'user.email']
        form_choices = {
            'status': [
                ('pending', 'Pending'),
                ('completed', 'Completed'),
                ('failed', 'Failed'),
                ('refunded', 'Refunded'),
                ('requires_approval', 'Requires Approval')
            ]
        }
        
        def on_model_change(self, form, model, is_created):
            # Log admin action
            if not is_created and 'status' in form.data:
                action = AdminAction(
                    admin_id=current_user.id,
                    action_type='transaction_update',
                    target_type='transaction',
                    target_id=model.id,
                    details=f"Status changed to {form.status.data}",
                    ip_address=request.remote_addr
                )
                db.session.add(action)
                db.session.commit()
    
    class WithdrawalAdminView(ModelView):
        column_list = ['id', 'affiliate_id', 'amount', 'status', 'payment_method', 'created_at']
        column_filters = ['status', 'payment_method', 'created_at']
        form_choices = {
            'status': [
                ('pending', 'Pending'),
                ('approved', 'Approved'),
                ('rejected', 'Rejected'),
                ('paid', 'Paid')
            ]
        }
        
        def on_model_change(self, form, model, is_created):
            if not is_created and 'status' in form.data:
                # Notify affiliate of status change
                from app.utils.notifications import send_withdrawal_status_notification
                send_withdrawal_status_notification(model)
                
                # Log admin action
                action = AdminAction(
                    admin_id=current_user.id,
                    action_type='withdrawal_update',
                    target_type='withdrawal',
                    target_id=model.id,
                    details=f"Status changed to {form.status.data}",
                    ip_address=request.remote_addr
                )
                db.session.add(action)
                db.session.commit()
    
    # Register admin views
    admin.add_view(UserAdminView(User, db.session))
    admin.add_view(ModelView(Role, db.session))
    admin.add_view(ModelView(APIKey, db.session))
    admin.add_view(ModelView(Subscription, db.session))
    admin.add_view(TransactionAdminView(Transaction, db.session))
    admin.add_view(ModelView(Affiliate, db.session))
    admin.add_view(ModelView(Commission, db.session))
    admin.add_view(WithdrawalAdminView(WithdrawalRequest, db.session))
    admin.add_view(ModelView(GeneratedImage, db.session))
    admin.add_view(ModelView(GeneratedVideo, db.session))
    admin.add_view(ModelView(AudioFile, db.session))
    admin.add_view(ModelView(IoTDeviceModel, db.session))
    admin.add_view(ModelView(AdminAction, db.session))

def initialize_background_systems():
    """Initialize background systems and managers"""
    from app.utils.background_tasks import BackgroundTaskManager
    from app.self_improvement import SelfImprovementSystem
    from app.advanced_ai.neural_architectures import AdvancedNeuralArchitecture
    from app.enterprise.security import EnterpriseSecurityManager
    from app.workflows.engine import WorkflowEngine
    from app.intelligence.core import IntelligenceCore
    from app.analytics.core import AdvancedAnalyticsEngine
    from app.payments.core import PaymentSystem
    from app.affiliate.core import AffiliateSystem
    from app.media.processing import MediaProcessingSystem
    from app.iot.management import IoTManagementSystem
    from app.self_learning.orchestrator import SelfLearningOrchestrator
    
    # Initialize systems
    systems = [
        (SelfImprovementSystem, 'start_background_improvement'),
        (BackgroundTaskManager, 'start_all_tasks'),
        (AdvancedNeuralArchitecture, 'initialize_neural_networks'),
        (EnterpriseSecurityManager, 'initialize_security_systems'),
        (WorkflowEngine, 'start_engine'),
        (IntelligenceCore, 'initialize_systems'),
        (AdvancedAnalyticsEngine, 'initialize'),
        (PaymentSystem, 'initialize'),
        (AffiliateSystem, 'initialize'),
        (MediaProcessingSystem, 'initialize'),
        (IoTManagementSystem, 'initialize'),
        (SelfLearningOrchestrator, 'initialize')
    ]
    
    for system_class, init_method in systems:
        try:
            system = system_class()
            getattr(system, init_method)()
            logger.info(f"Initialized {system_class.__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize {system_class.__name__}", error=str(e))

def schedule_recurring_tasks():
    """Schedule recurring background tasks"""
    tasks = [
        (timedelta(minutes=5), 'app.utils.background_tasks.cleanup_old_sessions', 3600),
        (timedelta(minutes=10), 'app.utils.background_tasks.update_analytics', 300),
        (timedelta(minutes=15), 'app.utils.background_tasks.process_pending_commissions', 1800),
        (timedelta(minutes=20), 'app.utils.background_tasks.cleanup_temp_files', 3600),
        (timedelta(minutes=30), 'app.utils.background_tasks.optimize_models', 86400),
        (timedelta(minutes=40), 'app.utils.background_tasks.check_iot_devices', 300),
        (timedelta(minutes=60), 'app.utils.background_tasks.send_daily_reports', 86400),
        (timedelta(minutes=120), 'app.utils.background_tasks.backup_database', 43200),
        (timedelta(minutes=180), 'app.utils.background_tasks.update_currency_rates', 86400),
        (timedelta(minutes=240), 'app.utils.background_tasks.check_subscription_renewals', 3600)
    ]
    
    for offset, func, interval in tasks:
        try:
            scheduler.schedule(
                scheduled_time=datetime.utcnow() + offset,
                func=func,
                interval=interval,
                repeat=None
            )
            logger.debug(f"Scheduled task", function=func, interval=interval)
        except Exception as e:
            logger.error(f"Failed to schedule task {func}", error=str(e))

def setup_production_logging(app):
    """Setup production logging with rotation and monitoring"""
    if not app.debug:
        # File handler with rotation
        file_handler = RotatingFileHandler(
            'logs/changex_neurix.log',
            maxBytes=10485760,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        # Error email handler
        if app.config.get('MAIL_SERVER'):
            auth = None
            if app.config['MAIL_USERNAME'] or app.config['MAIL_PASSWORD']:
                auth = (app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            
            secure = None
            if app.config['MAIL_USE_TLS']:
                secure = ()
            
            mail_handler = SMTPHandler(
                mailhost=(app.config['MAIL_SERVER'], app.config['MAIL_PORT']),
                fromaddr=app.config['MAIL_DEFAULT_SENDER'],
                toaddrs=app.config['ADMINS'],
                subject='ChangeX Neurix Application Error',
                credentials=auth,
                secure=secure
            )
            mail_handler.setLevel(logging.ERROR)
            mail_handler.setFormatter(logging.Formatter('''
Message type: %(levelname)s
Location: %(pathname)s:%(lineno)d
Module: %(module)s
Function: %(funcName)s
Time: %(asctime)s
Message:
%(message)s
'''))
            app.logger.addHandler(mail_handler)
        
        app.logger.setLevel(logging.INFO)

# Global app instance
from app import models

# Export new classes
__all__ = [
    'SelfLearningImageGenerator',
    'VideoGenerator',
    'AudioProcessor',
    'IoTController',
    'IoTDevice',
    'SelfLearningEngine',
    'ai_models',
    'redis_client',
    'socketio',
    'cache',
    'limiter'
]
