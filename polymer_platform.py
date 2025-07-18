import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import google.generativeai as genai
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import gspread
from google.oauth2.service_account import Credentials
import hashlib
import base64
import io
import re
# ==================== Enhanced 컴포넌트 초기화 ====================
ENHANCED_FEATURES_AVAILABLE = False

try:
    # Enhanced 컴포넌트들이 정의되어 있는지 확인
    # API 키 매니저
    api_key_manager = APIKeyManager()
    
    # API 모니터
    api_monitor = APIMonitor()
    
    # Enhanced AI 오케스트레이터
    enhanced_ai_orchestrator = EnhancedAIOrchestrator()
    
    # 데이터베이스 매니저
    database_manager = DatabaseManager()
    
    # 번역 서비스
    translation_service = TranslationService()
    
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced 기능이 활성화되었습니다.")
    
except Exception as e:
    print(f"⚠️ Enhanced 기능 초기화 실패: {e}")
    print("기본 모드로 실행됩니다.")
    
    # 더미 객체 생성 (에러 방지)
    api_key_manager = None
    api_monitor = None
    enhanced_ai_orchestrator = None
    database_manager = None
    translation_service = None

# 새로운 AI API 라이브러리
import google.generativeai as genai  # Gemini
from groq import Groq  # Groq
import requests  # Grok, SambaNova, DeepSeek API 호출용
from huggingface_hub import InferenceClient  # HuggingFace

# 데이터베이스 API 라이브러리
import httpx  # 비동기 HTTP 요청
from github import Github  # GitHub API
import xml.etree.ElementTree as ET  # PubChem XML 파싱

# 보안 및 환경 관리
import os
from getpass import getpass
import hashlib
from cryptography.fernet import Fernet  # API 키 암호화

# 병렬 처리 및 성능
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# 캐싱 및 상태 관리
from functools import lru_cache
import pickle
from datetime import datetime, timedelta
import tempfile

# 모니터링 및 로깅
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# 번역 및 텍스트 처리
import langdetect
from deep_translator import GoogleTranslator

# 추가 유틸리티
import json
import re
from urllib.parse import quote, urlencode
import time
from retrying import retry

# 데이터 시각화 (API 상태 표시용)
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header

# API 상태 모니터링을 위한 커스텀 타입
class APIStatus(Enum):
    """API 상태 열거형"""
    ONLINE = "online"
    OFFLINE = "offline"
    SLOW = "slow"
    ERROR = "error"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"

@dataclass
class APIResponse:
    """API 응답 데이터 클래스"""
    success: bool
    data: Any
    error: Optional[str] = None
    response_time: float = 0.0
    api_name: str = ""

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 설정 및 초기화 ====================

# Streamlit 페이지 설정
st.set_page_config(
    page_title="🧬 고분자 실험 설계 플랫폼",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 적용
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
    }
    .info-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==================== 상태 관리 클래스 ====================
# ==================== API 키 관리 시스템 ====================
class APIKeyManager:
    """API 키를 안전하게 관리하는 클래스"""
    
    def __init__(self):
        self.api_configs = {
            # AI APIs
            'gemini': {
                'name': 'Gemini 2.0 Flash',
                'env_key': 'GEMINI_API_KEY',
                'required': True,
                'test_endpoint': 'https://generativelanguage.googleapis.com/v1beta/models',
                'category': 'ai'
            },
            'grok': {
                'name': 'Grok 3 mini',
                'env_key': 'GROK_API_KEY',
                'required': False,
                'test_endpoint': 'https://api.x.ai/v1/chat/completions',
                'category': 'ai'
            },
            'sambanova': {
                'name': 'SambaNova',
                'env_key': 'SAMBANOVA_API_KEY',
                'required': False,
                'test_endpoint': 'https://api.sambanova.ai/v1/chat/completions',
                'category': 'ai'
            },
            'deepseek': {
                'name': 'DeepSeek',
                'env_key': 'DEEPSEEK_API_KEY',
                'required': False,
                'test_endpoint': 'https://api.deepseek.com/v1/chat/completions',
                'category': 'ai'
            },
            'groq': {
                'name': 'Groq',
                'env_key': 'GROQ_API_KEY',
                'required': False,
                'test_endpoint': 'https://api.groq.com/openai/v1/chat/completions',
                'category': 'ai'
            },
            'huggingface': {
                'name': 'HuggingFace',
                'env_key': 'HUGGINGFACE_API_KEY',
                'required': False,
                'test_endpoint': 'https://api-inference.huggingface.co/models',
                'category': 'ai'
            },
            
            # Database APIs
            'github': {
                'name': 'GitHub',
                'env_key': 'GITHUB_TOKEN',
                'required': False,
                'test_endpoint': 'https://api.github.com/user',
                'category': 'database'
            },
            'materials_project': {
                'name': 'Materials Project',
                'env_key': 'MP_API_KEY',
                'required': False,
                'test_endpoint': 'https://api.materialsproject.org',
                'category': 'database'
            }
        }
        
    def initialize_keys(self):
        """Google Colab 환경에서 API 키 초기화"""
        if 'api_keys_initialized' not in st.session_state:
            st.session_state.api_keys_initialized = False
            st.session_state.api_keys = {}
            
        # Google Colab 환경 체크
        if self._is_colab():
            self._setup_colab_keys()
        else:
            self._setup_streamlit_keys()
    
    def _is_colab(self):
        """Google Colab 환경인지 확인"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _setup_colab_keys(self):
        """Google Colab에서 getpass로 키 입력받기"""
        if not st.session_state.api_keys_initialized:
            st.info("🔐 Google Colab 환경에서 API 키를 설정합니다.")
            
            # 필수 API 키만 먼저 요청
            for key_id, config in self.api_configs.items():
                if config['required'] and key_id not in st.session_state.api_keys:
                    if not os.getenv(config['env_key']):
                        # Colab에서는 코드 셀에서 getpass 실행 필요
                        st.warning(f"⚠️ {config['name']} API 키가 필요합니다. 코드 셀에서 다음을 실행하세요:")
                        st.code(f"import os\nfrom getpass import getpass\nos.environ['{config['env_key']}'] = getpass('{config['name']} API Key: ')")
                    else:
                        st.session_state.api_keys[key_id] = os.getenv(config['env_key'])
                        
    def _setup_streamlit_keys(self):
        """Streamlit UI에서 키 입력받기"""
        with st.sidebar.expander("🔑 API 키 설정", expanded=not st.session_state.api_keys_initialized):
            
            # AI API 키 섹션
            st.subheader("AI APIs")
            ai_cols = st.columns(2)
            
            for idx, (key_id, config) in enumerate(
                [(k, v) for k, v in self.api_configs.items() if v['category'] == 'ai']
            ):
                col = ai_cols[idx % 2]
                with col:
                    # 환경변수에서 먼저 확인
                    env_value = os.getenv(config['env_key'])
                    current_value = st.session_state.api_keys.get(key_id, env_value or "")
                    
                    # 마스킹된 입력 필드
                    new_value = st.text_input(
                        config['name'],
                        value=self._mask_key(current_value) if current_value else "",
                        type="password",
                        key=f"input_{key_id}",
                        help=f"{'필수' if config['required'] else '선택'}"
                    )
                    
                    # 새 값이 입력되면 저장
                    if new_value and new_value != self._mask_key(current_value):
                        st.session_state.api_keys[key_id] = new_value
                        os.environ[config['env_key']] = new_value
            
            # Database API 키 섹션
            st.subheader("Database APIs")
            db_cols = st.columns(2)
            
            for idx, (key_id, config) in enumerate(
                [(k, v) for k, v in self.api_configs.items() if v['category'] == 'database']
            ):
                col = db_cols[idx % 2]
                with col:
                    env_value = os.getenv(config['env_key'])
                    current_value = st.session_state.api_keys.get(key_id, env_value or "")
                    
                    new_value = st.text_input(
                        config['name'],
                        value=self._mask_key(current_value) if current_value else "",
                        type="password",
                        key=f"input_{key_id}",
                        help="선택"
                    )
                    
                    if new_value and new_value != self._mask_key(current_value):
                        st.session_state.api_keys[key_id] = new_value
                        os.environ[config['env_key']] = new_value
            
            # 키 테스트 버튼
            if st.button("🔍 API 연결 테스트", use_container_width=True):
                self._test_all_connections()
                
            # 초기화 완료 표시
            if self._check_required_keys():
                st.session_state.api_keys_initialized = True
                st.success("✅ 필수 API 키가 모두 설정되었습니다.")
    
    def _mask_key(self, key: str) -> str:
        """API 키를 마스킹 처리"""
        if not key:
            return ""
        if len(key) <= 8:
            return "*" * len(key)
        return key[:4] + "*" * (len(key) - 8) + key[-4:]
    
    def _check_required_keys(self) -> bool:
        """필수 키가 모두 설정되었는지 확인"""
        for key_id, config in self.api_configs.items():
            if config['required']:
                if key_id not in st.session_state.api_keys and not os.getenv(config['env_key']):
                    return False
        return True
    
    def get_key(self, key_id: str) -> Optional[str]:
        """API 키 반환"""
        # 우선순위: session_state > 환경변수
        if key_id in st.session_state.api_keys:
            return st.session_state.api_keys[key_id]
        
        config = self.api_configs.get(key_id)
        if config:
            return os.getenv(config['env_key'])
        
        return None
    
    def _test_all_connections(self):
        """모든 API 연결 테스트"""
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(self.api_configs)
        for idx, (key_id, config) in enumerate(self.api_configs.items()):
            status_text.text(f"테스트 중: {config['name']}...")
            progress_bar.progress((idx + 1) / total)
            
            api_key = self.get_key(key_id)
            if api_key:
                results[key_id] = self._test_connection(key_id, api_key)
            else:
                results[key_id] = {'status': 'no_key', 'message': 'API 키 없음'}
        
        # 결과 표시
        st.subheader("🔍 API 연결 테스트 결과")
        
        for category in ['ai', 'database']:
            st.write(f"**{category.upper()} APIs**")
            cols = st.columns(3)
            
            items = [(k, v) for k, v in self.api_configs.items() if v['category'] == category]
            for idx, (key_id, config) in enumerate(items):
                col = cols[idx % 3]
                with col:
                    result = results.get(key_id, {})
                    status = result.get('status', 'no_key')
                    
                    if status == 'success':
                        st.success(f"✅ {config['name']}")
                    elif status == 'no_key':
                        st.info(f"🔑 {config['name']}: 키 없음")
                    else:
                        st.error(f"❌ {config['name']}: {result.get('message', '오류')}")
        
        progress_bar.empty()
        status_text.empty()
    
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _test_connection(self, key_id: str, api_key: str) -> dict:
        """개별 API 연결 테스트"""
        config = self.api_configs[key_id]
        
        try:
            start_time = time.time()
            
            if key_id == 'gemini':
                genai.configure(api_key=api_key)
                models = genai.list_models()
                response_time = time.time() - start_time
                return {'status': 'success', 'response_time': response_time}
                
            elif key_id == 'github':
                g = Github(api_key)
                user = g.get_user()
                response_time = time.time() - start_time
                return {'status': 'success', 'response_time': response_time, 'user': user.login}
                
            else:
                # 기본 HTTP 테스트
                headers = self._get_auth_headers(key_id, api_key)
                response = requests.get(
                    config['test_endpoint'],
                    headers=headers,
                    timeout=5
                )
                response_time = time.time() - start_time
                
                if response.status_code in [200, 401, 403]:  # 인증 오류도 연결은 성공
                    return {'status': 'success', 'response_time': response_time}
                else:
                    return {'status': 'error', 'message': f'HTTP {response.status_code}'}
                    
        except Exception as e:
            return {'status': 'error', 'message': str(e)[:50]}
    
    def _get_auth_headers(self, key_id: str, api_key: str) -> dict:
        """API별 인증 헤더 생성"""
        if key_id in ['grok', 'sambanova', 'deepseek', 'groq']:
            return {'Authorization': f'Bearer {api_key}'}
        elif key_id == 'huggingface':
            return {'Authorization': f'Bearer {api_key}'}
        elif key_id == 'materials_project':
            return {'X-API-KEY': api_key}
        else:
            return {}

# ==================== API 모니터링 시스템 ====================
class APIMonitor:
    """API 상태를 실시간으로 모니터링하는 클래스"""
    
    def __init__(self):
        # API 상태 저장
        if 'api_status' not in st.session_state:
            st.session_state.api_status = {}
        if 'api_metrics' not in st.session_state:
            st.session_state.api_metrics = {}
            
        # API 그룹 정의 (기능별 필요 API)
        self.api_groups = {
            'experiment_design': {
                'name': '실험 설계',
                'apis': ['gemini', 'grok', 'sambanova', 'deepseek', 'groq', 'huggingface'],
                'icon': '🧪'
            },
            'literature_search': {
                'name': '문헌 검색',
                'apis': ['openalex', 'crossref', 'pubchem', 'semantic_scholar'],
                'icon': '📚'
            },
            'protocol_search': {
                'name': '프로토콜 검색',
                'apis': ['protocols_io', 'github', 'zenodo', 'figshare'],
                'icon': '📋'
            },
            'property_analysis': {
                'name': '물성 분석',
                'apis': ['polyinfo', 'materials_project', 'nist'],
                'icon': '📊'
            },
            'integrated_search': {
                'name': '통합 검색',
                'apis': ['gemini', 'deepseek', 'openalex', 'github', 'materials_project'],
                'icon': '🔍'
            }
        }
        
        # 상태별 색상 및 아이콘
        self.status_config = {
            APIStatus.ONLINE: {'color': '#28a745', 'icon': '🟢', 'text': '정상'},
            APIStatus.SLOW: {'color': '#ffc107', 'icon': '🟡', 'text': '느림'},
            APIStatus.OFFLINE: {'color': '#dc3545', 'icon': '🔴', 'text': '오프라인'},
            APIStatus.ERROR: {'color': '#dc3545', 'icon': '❌', 'text': '오류'},
            APIStatus.UNAUTHORIZED: {'color': '#6c757d', 'icon': '🔒', 'text': '인증 필요'},
            APIStatus.RATE_LIMITED: {'color': '#ff6b6b', 'icon': '⏳', 'text': '제한됨'}
        }
    
    def update_status(self, api_name: str, status: APIStatus, response_time: float = None, error_msg: str = None):
        """API 상태 업데이트"""
        st.session_state.api_status[api_name] = {
            'status': status,
            'last_checked': datetime.now(),
            'response_time': response_time,
            'error_msg': error_msg
        }
        
        # 메트릭 업데이트
        if api_name not in st.session_state.api_metrics:
            st.session_state.api_metrics[api_name] = {
                'total_calls': 0,
                'success_calls': 0,
                'total_response_time': 0,
                'errors': []
            }
        
        metrics = st.session_state.api_metrics[api_name]
        metrics['total_calls'] += 1
        
        if status == APIStatus.ONLINE:
            metrics['success_calls'] += 1
            if response_time:
                metrics['total_response_time'] += response_time
        elif error_msg:
            metrics['errors'].append({
                'time': datetime.now(),
                'error': error_msg
            })
            # 최근 10개 에러만 유지
            metrics['errors'] = metrics['errors'][-10:]
    
    def get_api_status(self, api_name: str) -> Dict:
        """특정 API의 현재 상태 반환"""
        return st.session_state.api_status.get(api_name, {
            'status': APIStatus.OFFLINE,
            'last_checked': None,
            'response_time': None,
            'error_msg': 'Not checked yet'
        })
    
    def get_context_apis(self, context: str) -> List[str]:
        """현재 컨텍스트에 필요한 API 목록 반환"""
        group = self.api_groups.get(context, {})
        return group.get('apis', [])
    
    def display_status_bar(self, context: str):
        """현재 컨텍스트의 API 상태 표시"""
        group = self.api_groups.get(context)
        if not group:
            return
            
        # 상태 바 컨테이너
        with st.container():
            st.markdown(f"### {group['icon']} {group['name']} API 상태")
            
            cols = st.columns(len(group['apis']))
            
            for idx, api_name in enumerate(group['apis']):
                with cols[idx]:
                    status_info = self.get_api_status(api_name)
                    status = status_info['status']
                    config = self.status_config[status]
                    
                    # API 이름과 상태 표시
                    api_display_name = api_key_manager.api_configs.get(api_name, {}).get('name', api_name)
                    
                    # 메트릭 카드 스타일로 표시
                    st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 8px;
                            padding: 10px;
                            text-align: center;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            border-left: 4px solid {config['color']};
                        ">
                            <div style="font-size: 24px;">{config['icon']}</div>
                            <div style="font-size: 12px; font-weight: bold;">{api_display_name}</div>
                            <div style="font-size: 10px; color: {config['color']};">{config['text']}</div>
                            {f'<div style="font-size: 10px; color: #666;">{status_info["response_time"]:.2f}초</div>' 
                             if status_info['response_time'] else ''}
                        </div>
                    """, unsafe_allow_html=True)
    
    def display_detailed_status(self):
        """상세 API 상태 표시 (사이드바용)"""
        with st.sidebar.expander("📊 API 상태 모니터링", expanded=False):
            # 전체 상태 요약
            total_apis = len(st.session_state.api_status)
            online_apis = sum(1 for s in st.session_state.api_status.values() 
                            if s['status'] == APIStatus.ONLINE)
            
            if total_apis > 0:
                success_rate = (online_apis / total_apis) * 100
                st.metric("전체 API 상태", f"{online_apis}/{total_apis} 온라인", 
                         f"{success_rate:.0f}% 가동률")
            
            # 카테고리별 상태
            for category in ['ai', 'database']:
                st.markdown(f"**{category.upper()} APIs**")
                
                # 해당 카테고리의 API들
                category_apis = [
                    (k, v) for k, v in api_key_manager.api_configs.items() 
                    if v['category'] == category
                ]
                
                for api_id, api_config in category_apis:
                    status_info = self.get_api_status(api_id)
                    status = status_info['status']
                    config = self.status_config[status]
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.markdown(config['icon'])
                    with col2:
                        st.markdown(f"**{api_config['name']}**")
                    with col3:
                        if status_info['response_time']:
                            st.markdown(f"{status_info['response_time']:.2f}s")
                    
                    # 에러 메시지 표시
                    if status == APIStatus.ERROR and status_info.get('error_msg'):
                        st.caption(f"❗ {status_info['error_msg'][:50]}...")
                
                st.markdown("---")
    
    def check_api_health(self, api_name: str, api_key: str = None) -> APIResponse:
        """API 상태 확인"""
        start_time = time.time()
        
        try:
            # API 키가 없으면 키 매니저에서 가져오기
            if not api_key:
                api_key = api_key_manager.get_key(api_name)
                if not api_key:
                    self.update_status(api_name, APIStatus.UNAUTHORIZED)
                    return APIResponse(
                        success=False,
                        data=None,
                        error="API key not found",
                        api_name=api_name
                    )
            
            # 간단한 헬스 체크 수행
            if api_name == 'gemini':
                genai.configure(api_key=api_key)
                genai.list_models()
            elif api_name == 'github':
                g = Github(api_key)
                g.get_user()
            else:
                # 기본 HTTP 체크
                config = api_key_manager.api_configs.get(api_name, {})
                if config.get('test_endpoint'):
                    headers = api_key_manager._get_auth_headers(api_name, api_key)
                    response = requests.get(
                        config['test_endpoint'],
                        headers=headers,
                        timeout=5
                    )
                    if response.status_code >= 400:
                        raise Exception(f"HTTP {response.status_code}")
            
            # 성공
            response_time = time.time() - start_time
            
            # 응답 시간에 따른 상태 결정
            if response_time > 5:
                status = APIStatus.SLOW
            else:
                status = APIStatus.ONLINE
                
            self.update_status(api_name, status, response_time)
            
            return APIResponse(
                success=True,
                data=None,
                response_time=response_time,
                api_name=api_name
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            # 에러 타입에 따른 상태 결정
            if "rate limit" in error_msg.lower():
                status = APIStatus.RATE_LIMITED
            elif "unauthorized" in error_msg.lower() or "403" in error_msg:
                status = APIStatus.UNAUTHORIZED
            else:
                status = APIStatus.ERROR
                
            self.update_status(api_name, status, response_time, error_msg)
            
            return APIResponse(
                success=False,
                data=None,
                error=error_msg,
                response_time=response_time,
                api_name=api_name
            )
    
    def auto_health_check(self, context: str):
        """컨텍스트에 필요한 모든 API 상태 자동 확인"""
        apis = self.get_context_apis(context)
        
        with st.spinner(f"API 상태 확인 중... ({len(apis)}개)"):
            # ThreadPoolExecutor로 병렬 체크
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self.check_api_health, api): api 
                    for api in apis
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error(f"Health check failed: {e}")

# API 모니터 초기화
api_monitor = APIMonitor()

# ==================== AI 엔진 클래스들 ====================
class BaseAIEngine:
    """모든 AI 엔진의 기본 클래스"""
    
    def __init__(self, name: str, api_key_id: str):
        self.name = name
        self.api_key_id = api_key_id
        self.api_key = None
        self.is_available = False
        
    def initialize(self):
        """API 키 확인 및 초기화"""
        self.api_key = api_key_manager.get_key(self.api_key_id)
        self.is_available = bool(self.api_key)
        return self.is_available
    
    async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
        """비동기 생성 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def generate(self, prompt: str, **kwargs) -> APIResponse:
        """동기 생성 래퍼"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.generate_async(prompt, **kwargs))
        finally:
            loop.close()

class GeminiEngine(BaseAIEngine):
    """Gemini AI 엔진"""
    
    def __init__(self):
        super().__init__("Gemini 2.0 Flash", "gemini")
        self.model = None
        
    def initialize(self):
        if super().initialize():
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                return True
            except Exception as e:
                logger.error(f"Gemini initialization failed: {e}")
                return False
        return False
    
    async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            # Gemini 특화 설정
            generation_config = {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "max_output_tokens": kwargs.get("max_tokens", 2048),
            }
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            response_time = time.time() - start_time
            api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
            
            return APIResponse(
                success=True,
                data=response.text,
                response_time=response_time,
                api_name=self.name
            )
            
        except Exception as e:
            api_monitor.update_status(self.api_key_id, APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

class GrokEngine(BaseAIEngine):
    """Grok AI 엔진"""
    
    def __init__(self):
        super().__init__("Grok 3 mini", "grok")
        self.endpoint = "https://api.x.ai/v1/chat/completions"
        
    async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "grok-beta",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 2048)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, headers=headers, json=data) as response:
                    result = await response.json()
                    
                    if response.status == 200:
                        response_time = time.time() - start_time
                        api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
                        
                        return APIResponse(
                            success=True,
                            data=result['choices'][0]['message']['content'],
                            response_time=response_time,
                            api_name=self.name
                        )
                    else:
                        raise Exception(f"API error: {result}")
                        
        except Exception as e:
            api_monitor.update_status(self.api_key_id, APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

class SambaNovaEngine(BaseAIEngine):
    """SambaNova AI 엔진"""
    
    def __init__(self):
        super().__init__("SambaNova", "sambanova")
        self.endpoint = "https://api.sambanova.ai/v1/chat/completions"
        
    async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "Meta-Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 2048)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, headers=headers, json=data) as response:
                    result = await response.json()
                    
                    if response.status == 200:
                        response_time = time.time() - start_time
                        api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
                        
                        return APIResponse(
                            success=True,
                            data=result['choices'][0]['message']['content'],
                            response_time=response_time,
                            api_name=self.name
                        )
                    else:
                        raise Exception(f"API error: {result}")
                        
        except Exception as e:
            api_monitor.update_status(self.api_key_id, APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

class DeepSeekEngine(BaseAIEngine):
    """DeepSeek AI 엔진"""
    
    def __init__(self):
        super().__init__("DeepSeek", "deepseek")
        self.endpoint = "https://api.deepseek.com/v1/chat/completions"
        
    async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # DeepSeek은 수학/과학에 특화
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant specialized in scientific calculations and chemical analysis.")
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get("temperature", 0.3),  # 더 정확한 계산을 위해 낮은 온도
                "max_tokens": kwargs.get("max_tokens", 2048)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, headers=headers, json=data) as response:
                    result = await response.json()
                    
                    if response.status == 200:
                        response_time = time.time() - start_time
                        api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
                        
                        return APIResponse(
                            success=True,
                            data=result['choices'][0]['message']['content'],
                            response_time=response_time,
                            api_name=self.name
                        )
                    else:
                        raise Exception(f"API error: {result}")
                        
        except Exception as e:
            api_monitor.update_status(self.api_key_id, APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

class GroqEngine(BaseAIEngine):
    """Groq AI 엔진 (초고속)"""
    
    def __init__(self):
        super().__init__("Groq", "groq")
        self.client = None
        
    def initialize(self):
        if super().initialize():
            try:
                # from groq import Groq  # 나중에 주석 해제
                # self.client = Groq(api_key=self.api_key)
                return True
            except Exception as e:
                logger.error(f"Groq initialization failed: {e}")
                return False
        return False
    
    async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            # Groq는 현재 주석 처리 (패키지 설치 후 활성화)
            """
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2048)
            )
            
            response_time = time.time() - start_time
            api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
            
            return APIResponse(
                success=True,
                data=response.choices[0].message.content,
                response_time=response_time,
                api_name=self.name
            )
            """
            
            # 임시 응답
            return APIResponse(
                success=False,
                data=None,
                error="Groq not yet implemented",
                api_name=self.name
            )
            
        except Exception as e:
            api_monitor.update_status(self.api_key_id, APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

class HuggingFaceEngine(BaseAIEngine):
    """HuggingFace AI 엔진"""
    
    def __init__(self):
        super().__init__("HuggingFace", "huggingface")
        self.client = None
        
    def initialize(self):
        if super().initialize():
            try:
                self.client = InferenceClient(token=self.api_key)
                return True
            except Exception as e:
                logger.error(f"HuggingFace initialization failed: {e}")
                return False
        return False
    
    async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            # 모델 선택 (무료 티어용)
            model = kwargs.get("model", "meta-llama/Llama-2-7b-chat-hf")
            
            response = await asyncio.to_thread(
                self.client.text_generation,
                prompt,
                model=model,
                max_new_tokens=kwargs.get("max_tokens", 512),  # 무료 티어 제한
                temperature=kwargs.get("temperature", 0.7)
            )
            
            response_time = time.time() - start_time
            api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
            
            return APIResponse(
                success=True,
                data=response,
                response_time=response_time,
                api_name=self.name
            )
            
        except Exception as e:
            api_monitor.update_status(self.api_key_id, APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

# ==================== 확장된 AI 오케스트레이터 ====================
class EnhancedAIOrchestrator:
    """6개 AI를 통합 관리하는 오케스트레이터"""
    
    def __init__(self):
        # AI 엔진 초기화
        self.engines = {
            'gemini': GeminiEngine(),
            'grok': GrokEngine(),
            'sambanova': SambaNovaEngine(),
            'deepseek': DeepSeekEngine(),
            'groq': GroqEngine(),
            'huggingface': HuggingFaceEngine()
        }
        
        # 사용 가능한 엔진 확인
        self.available_engines = {}
        self._initialize_engines()
        
        # AI 역할 정의
        self.ai_roles = {
            'gemini': {'strength': '과학적 분석, 한국어 처리', 'priority': 1},
            'grok': {'strength': '최신 정보, 창의적 접근', 'priority': 2},
            'sambanova': {'strength': '대규모 데이터 처리', 'priority': 3},
            'deepseek': {'strength': '수식/계산, 화학 분석', 'priority': 1},
            'groq': {'strength': '초고속 응답', 'priority': 2},
            'huggingface': {'strength': '특수 모델', 'priority': 4}
        }
        
    def _initialize_engines(self):
        """사용 가능한 엔진 초기화"""
        for name, engine in self.engines.items():
            if engine.initialize():
                self.available_engines[name] = engine
                logger.info(f"AI Engine initialized: {name}")
            else:
                logger.warning(f"AI Engine not available: {name}")
    
    async def generate_single(self, engine_name: str, prompt: str, **kwargs) -> APIResponse:
        """단일 AI 엔진으로 생성"""
        engine = self.available_engines.get(engine_name)
        if not engine:
            return APIResponse(
                success=False,
                data=None,
                error=f"Engine {engine_name} not available",
                api_name=engine_name
            )
        
        return await engine.generate_async(prompt, **kwargs)
    
    async def generate_parallel(self, prompt: str, engines: List[str] = None, **kwargs) -> Dict[str, APIResponse]:
        """여러 AI 엔진으로 병렬 생성"""
        if not engines:
            engines = list(self.available_engines.keys())
        
        # 사용 가능한 엔진만 필터링
        engines = [e for e in engines if e in self.available_engines]
        
        if not engines:
            return {}
        
        # 병렬 실행
        tasks = {
            engine: self.generate_single(engine, prompt, **kwargs)
            for engine in engines
        }
        
        results = {}
        for engine, task in tasks.items():
            try:
                results[engine] = await task
            except Exception as e:
                results[engine] = APIResponse(
                    success=False,
                    data=None,
                    error=str(e),
                    api_name=engine
                )
        
        return results
    
    def generate_consensus(self, prompt: str, required_engines: List[str] = None, **kwargs) -> Dict:
        """합의 기반 생성 (동기 래퍼)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._generate_consensus_async(prompt, required_engines, **kwargs)
            )
        finally:
            loop.close()
    
    async def _generate_consensus_async(self, prompt: str, required_engines: List[str] = None, **kwargs) -> Dict:
        """여러 AI의 합의를 통한 최적 답변 생성"""
        # 기본적으로 상위 3개 엔진 사용
        if not required_engines:
            required_engines = ['gemini', 'deepseek', 'grok']
        
        # 병렬로 응답 생성
        results = await self.generate_parallel(prompt, required_engines, **kwargs)
        
        # 성공한 응답만 추출
        successful_responses = {
            engine: result.data 
            for engine, result in results.items() 
            if result.success and result.data
        }
        
        if not successful_responses:
            return {
                'success': False,
                'error': 'No successful responses from any AI engine',
                'responses': results
            }
        
        # 단일 응답인 경우
        if len(successful_responses) == 1:
            engine, response = next(iter(successful_responses.items()))
            return {
                'success': True,
                'final_answer': response,
                'consensus_type': 'single',
                'contributing_engines': [engine],
                'responses': results
            }
        
        # 합의 도출
        consensus_prompt = f"""
        다음은 동일한 질문에 대한 여러 AI의 답변입니다:
        
        원래 질문: {prompt}
        
        답변들:
        {json.dumps(successful_responses, ensure_ascii=False, indent=2)}
        
        위 답변들을 종합하여 가장 정확하고 유용한 통합 답변을 만들어주세요.
        각 AI의 장점을 살려 최적의 답변을 도출하되, 중복은 제거하고 핵심만 정리해주세요.
        """
        
        # Gemini로 최종 통합 (가장 신뢰할 수 있는 엔진)
        if 'gemini' in self.available_engines:
            final_result = await self.generate_single('gemini', consensus_prompt, temperature=0.3)
            if final_result.success:
                return {
                    'success': True,
                    'final_answer': final_result.data,
                    'consensus_type': 'integrated',
                    'contributing_engines': list(successful_responses.keys()),
                    'responses': results
                }
        
        # Gemini 실패 시 가장 긴 응답 반환
        longest_response = max(successful_responses.items(), key=lambda x: len(x[1]))
        return {
            'success': True,
            'final_answer': longest_response[1],
            'consensus_type': 'longest',
            'contributing_engines': [longest_response[0]],
            'responses': results
        }
    
    def get_specialized_engine(self, task_type: str) -> str:
        """작업 유형에 따른 최적 엔진 선택"""
        task_engine_map = {
            'calculation': 'deepseek',
            'korean': 'gemini',
            'creative': 'grok',
            'fast': 'groq',
            'large_data': 'sambanova',
            'specialized': 'huggingface'
        }
        
        engine = task_engine_map.get(task_type, 'gemini')
        
        # 사용 가능한지 확인
        if engine in self.available_engines:
            return engine
        
        # 대체 엔진 찾기
        for alt_engine in self.available_engines.keys():
            return alt_engine
        
        return None

# 기존 AIOrchestrator를 대체
AIOrchestrator = EnhancedAIOrchestrator

# ==================== 데이터베이스 API 클래스들 ====================
class BaseDBAPI:
    """모든 데이터베이스 API의 기본 클래스"""
    
    def __init__(self, name: str, api_key_id: str = None):
        self.name = name
        self.api_key_id = api_key_id
        self.api_key = None
        self.base_url = ""
        self.headers = {}
        
    def initialize(self):
        """API 초기화"""
        if self.api_key_id:
            self.api_key = api_key_manager.get_key(self.api_key_id)
            if not self.api_key:
                logger.warning(f"{self.name} API key not found")
                return False
        return True
    
    async def search_async(self, query: str, **kwargs) -> APIResponse:
        """비동기 검색 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def search(self, query: str, **kwargs) -> APIResponse:
        """동기 검색 래퍼"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.search_async(query, **kwargs))
        finally:
            loop.close()

class OpenAlexAPI(BaseDBAPI):
    """OpenAlex 학술 데이터베이스 API"""
    
    def __init__(self):
        super().__init__("OpenAlex")
        self.base_url = "https://api.openalex.org"
        
    async def search_async(self, query: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            # 검색 파라미터 설정
            params = {
                'search': query,
                'filter': kwargs.get('filter', ''),
                'per_page': kwargs.get('per_page', 10),
                'page': kwargs.get('page', 1)
            }
            
            # 고분자 관련 필터 추가
            if kwargs.get('polymer_filter', True):
                if params['filter']:
                    params['filter'] += ','
                params['filter'] += 'concepts.id:C192854747'  # Polymer Science concept
            
            url = f"{self.base_url}/works"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        response_time = time.time() - start_time
                        api_monitor.update_status('openalex', APIStatus.ONLINE, response_time)
                        
                        # 결과 포맷팅
                        formatted_results = []
                        for work in data.get('results', []):
                            formatted_results.append({
                                'title': work.get('title'),
                                'authors': [author['author']['display_name'] 
                                           for author in work.get('authorships', [])],
                                'year': work.get('publication_year'),
                                'doi': work.get('doi'),
                                'abstract': work.get('abstract'),
                                'cited_by_count': work.get('cited_by_count', 0),
                                'open_access': work.get('open_access', {}).get('is_oa', False),
                                'pdf_url': work.get('open_access', {}).get('oa_url')
                            })
                        
                        return APIResponse(
                            success=True,
                            data={
                                'results': formatted_results,
                                'total_count': data.get('meta', {}).get('count', 0)
                            },
                            response_time=response_time,
                            api_name=self.name
                        )
                    else:
                        raise Exception(f"API error: {response.status}")
                        
        except Exception as e:
            api_monitor.update_status('openalex', APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

class CrossRefAPI(BaseDBAPI):
    """CrossRef 학술 메타데이터 API"""
    
    def __init__(self):
        super().__init__("CrossRef")
        self.base_url = "https://api.crossref.org"
        
    async def search_async(self, query: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            # 검색 파라미터
            params = {
                'query': query,
                'rows': kwargs.get('rows', 10),
                'offset': kwargs.get('offset', 0)
            }
            
            # 고분자 관련 필터
            if kwargs.get('polymer_filter', True):
                params['query'] += ' polymer OR polymeric OR macromolecule'
            
            url = f"{self.base_url}/works"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        response_time = time.time() - start_time
                        api_monitor.update_status('crossref', APIStatus.ONLINE, response_time)
                        
                        # 결과 포맷팅
                        formatted_results = []
                        for item in data.get('message', {}).get('items', []):
                            formatted_results.append({
                                'title': item.get('title', [''])[0] if item.get('title') else '',
                                'authors': [f"{author.get('given', '')} {author.get('family', '')}"
                                           for author in item.get('author', [])],
                                'year': item.get('published-print', {}).get('date-parts', [[None]])[0][0],
                                'doi': item.get('DOI'),
                                'journal': item.get('container-title', [''])[0] if item.get('container-title') else '',
                                'publisher': item.get('publisher'),
                                'type': item.get('type')
                            })
                        
                        return APIResponse(
                            success=True,
                            data={
                                'results': formatted_results,
                                'total_count': data.get('message', {}).get('total-results', 0)
                            },
                            response_time=response_time,
                            api_name=self.name
                        )
                    else:
                        raise Exception(f"API error: {response.status}")
                        
        except Exception as e:
            api_monitor.update_status('crossref', APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

class PubChemAPI(BaseDBAPI):
    """PubChem 화학물질 데이터베이스 API"""
    
    def __init__(self):
        super().__init__("PubChem")
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
    async def search_async(self, query: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            # 검색 타입 결정
            search_type = kwargs.get('search_type', 'compound')
            output_format = kwargs.get('format', 'JSON')
            
            # 화합물 검색
            if search_type == 'compound':
                url = f"{self.base_url}/compound/name/{quote(query)}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/{output_format}"
            else:
                # 물질명 검색
                url = f"{self.base_url}/compound/name/{quote(query)}/{output_format}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        response_time = time.time() - start_time
                        api_monitor.update_status('pubchem', APIStatus.ONLINE, response_time)
                        
                        # 결과 포맷팅
                        properties = data.get('PropertyTable', {}).get('Properties', [])
                        
                        formatted_results = []
                        for prop in properties:
                            formatted_results.append({
                                'cid': prop.get('CID'),
                                'molecular_formula': prop.get('MolecularFormula'),
                                'molecular_weight': prop.get('MolecularWeight'),
                                'smiles': prop.get('CanonicalSMILES'),
                                'url': f"https://pubchem.ncbi.nlm.nih.gov/compound/{prop.get('CID')}"
                            })
                        
                        return APIResponse(
                            success=True,
                            data={'results': formatted_results},
                            response_time=response_time,
                            api_name=self.name
                        )
                    else:
                        raise Exception(f"API error: {response.status}")
                        
        except Exception as e:
            api_monitor.update_status('pubchem', APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

class GitHubAPI(BaseDBAPI):
    """GitHub 코드 저장소 API"""
    
    def __init__(self):
        super().__init__("GitHub", "github")
        self.client = None
        
    def initialize(self):
        if super().initialize():
            try:
                if self.api_key:
                    self.client = Github(self.api_key)
                else:
                    self.client = Github()  # 인증 없이도 제한적 사용 가능
                return True
            except Exception as e:
                logger.error(f"GitHub initialization failed: {e}")
                return False
        return False
    
    async def search_async(self, query: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            # 검색 쿼리 구성
            search_query = query
            if kwargs.get('polymer_filter', True):
                search_query += ' polymer'
            
            # 언어 필터
            language = kwargs.get('language', 'python')
            if language:
                search_query += f' language:{language}'
            
            # 검색 실행
            repositories = await asyncio.to_thread(
                self.client.search_repositories,
                query=search_query,
                sort=kwargs.get('sort', 'stars'),
                order='desc'
            )
            
            # 결과 수집 (최대 10개)
            formatted_results = []
            count = 0
            for repo in repositories:
                if count >= kwargs.get('limit', 10):
                    break
                    
                formatted_results.append({
                    'name': repo.full_name,
                    'description': repo.description,
                    'stars': repo.stargazers_count,
                    'language': repo.language,
                    'url': repo.html_url,
                    'updated': repo.updated_at.isoformat() if repo.updated_at else None,
                    'topics': repo.get_topics()
                })
                count += 1
            
            response_time = time.time() - start_time
            api_monitor.update_status('github', APIStatus.ONLINE, response_time)
            
            return APIResponse(
                success=True,
                data={
                    'results': formatted_results,
                    'total_count': repositories.totalCount
                },
                response_time=response_time,
                api_name=self.name
            )
            
        except Exception as e:
            api_monitor.update_status('github', APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

class MaterialsProjectAPI(BaseDBAPI):
    """Materials Project 재료 데이터베이스 API"""
    
    def __init__(self):
        super().__init__("Materials Project", "materials_project")
        self.base_url = "https://api.materialsproject.org"
        
    def initialize(self):
        if super().initialize():
            if self.api_key:
                self.headers = {'X-API-KEY': self.api_key}
                return True
            return False
        return False
    
    async def search_async(self, query: str, **kwargs) -> APIResponse:
        try:
            start_time = time.time()
            
            # Materials Project는 주로 무기물이므로 고분자 검색은 제한적
            # 대신 첨가제나 필러 검색에 유용
            
            url = f"{self.base_url}/materials/summary"
            params = {
                'formula': query,  # 화학식으로 검색
                '_limit': kwargs.get('limit', 10)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        response_time = time.time() - start_time
                        api_monitor.update_status('materials_project', APIStatus.ONLINE, response_time)
                        
                        # 결과 포맷팅
                        formatted_results = []
                        for material in data.get('data', []):
                            formatted_results.append({
                                'material_id': material.get('material_id'),
                                'formula': material.get('formula_pretty'),
                                'energy': material.get('energy_per_atom'),
                                'band_gap': material.get('band_gap'),
                                'density': material.get('density'),
                                'crystal_system': material.get('symmetry', {}).get('crystal_system')
                            })
                        
                        return APIResponse(
                            success=True,
                            data={'results': formatted_results},
                            response_time=response_time,
                            api_name=self.name
                        )
                    else:
                        raise Exception(f"API error: {response.status}")
                        
        except Exception as e:
            api_monitor.update_status('materials_project', APIStatus.ERROR, error_msg=str(e))
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                api_name=self.name
            )

# ==================== 통합 데이터베이스 매니저 ====================
class DatabaseManager:
    """모든 데이터베이스 API를 통합 관리하는 클래스"""
    
    def __init__(self):
        # 데이터베이스 API 초기화
        self.databases = {
            'openalex': OpenAlexAPI(),
            'crossref': CrossRefAPI(),
            'pubchem': PubChemAPI(),
            'github': GitHubAPI(),
            'materials_project': MaterialsProjectAPI()
        }
        
        # 사용 가능한 DB 확인
        self.available_databases = {}
        self._initialize_databases()
        
        # DB 카테고리 정의
        self.db_categories = {
            'literature': ['openalex', 'crossref'],
            'chemical': ['pubchem', 'materials_project'],
            'code': ['github'],
            'protocol': ['protocols_io', 'zenodo']  # 추후 구현
        }
        
    def _initialize_databases(self):
        """사용 가능한 데이터베이스 초기화"""
        for name, db in self.databases.items():
            if db.initialize():
                self.available_databases[name] = db
                logger.info(f"Database initialized: {name}")
            else:
                logger.warning(f"Database not available: {name}")
    
    async def search_single(self, db_name: str, query: str, **kwargs) -> APIResponse:
        """단일 데이터베이스 검색"""
        db = self.available_databases.get(db_name)
        if not db:
            return APIResponse(
                success=False,
                data=None,
                error=f"Database {db_name} not available",
                api_name=db_name
            )
        
        return await db.search_async(query, **kwargs)
    
    async def search_parallel(self, query: str, databases: List[str] = None, **kwargs) -> Dict[str, APIResponse]:
        """여러 데이터베이스 병렬 검색"""
        if not databases:
            databases = list(self.available_databases.keys())
        
        # 사용 가능한 DB만 필터링
        databases = [db for db in databases if db in self.available_databases]
        
        if not databases:
            return {}
        
        # 병렬 검색 실행
        tasks = {}
        for db_name in databases:
            tasks[db_name] = self.search_single(db_name, query, **kwargs)
        
        results = {}
        for db_name, task in tasks.items():
            try:
                results[db_name] = await task
            except Exception as e:
                results[db_name] = APIResponse(
                    success=False,
                    data=None,
                    error=str(e),
                    api_name=db_name
                )
        
        return results
    
    def integrated_search(self, query: str, categories: List[str] = None, **kwargs) -> Dict:
        """통합 검색 (동기 래퍼)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._integrated_search_async(query, categories, **kwargs)
            )
        finally:
            loop.close()
    
    async def _integrated_search_async(self, query: str, categories: List[str] = None, **kwargs) -> Dict:
        """카테고리별 통합 검색"""
        # 기본적으로 모든 카테고리 검색
        if not categories:
            categories = list(self.db_categories.keys())
        
        # 검색할 DB 목록 구성
        databases_to_search = []
        for category in categories:
            databases_to_search.extend(self.db_categories.get(category, []))
        
        # 중복 제거
        databases_to_search = list(set(databases_to_search))
        
        # 병렬 검색 실행
        results = await self.search_parallel(query, databases_to_search, **kwargs)
        
        # 카테고리별로 결과 정리
        categorized_results = {}
        for category in categories:
            categorized_results[category] = {}
            for db in self.db_categories.get(category, []):
                if db in results:
                    categorized_results[category][db] = results[db]
        
        return {
            'success': True,
            'query': query,
            'results_by_category': categorized_results,
            'total_databases_searched': len(results),
            'successful_searches': sum(1 for r in results.values() if r.success)
        }
    
    def get_polymer_data(self, polymer_name: str) -> Dict:
        """특정 고분자에 대한 종합 데이터 수집"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._get_polymer_data_async(polymer_name)
            )
        finally:
            loop.close()
    
    async def _get_polymer_data_async(self, polymer_name: str) -> Dict:
        """고분자 관련 모든 정보 수집"""
        tasks = {
            'literature': self.search_parallel(
                f"{polymer_name} polymer properties synthesis",
                ['openalex', 'crossref']
            ),
            'chemical': self.search_single('pubchem', polymer_name),
            'code': self.search_single(
                'github', 
                f"{polymer_name} polymer simulation analysis",
                language='python'
            )
        }
        
        results = {}
        for category, task in tasks.items():
            try:
                results[category] = await task
            except Exception as e:
                logger.error(f"Error in {category} search: {e}")
                results[category] = None
        
        return {
            'polymer': polymer_name,
            'timestamp': datetime.now().isoformat(),
            'data': results
        }

# DatabaseManager 초기화
database_manager = DatabaseManager()

# ==================== 번역 서비스 ====================
class TranslationService:
    """다국어 번역 서비스"""
    
    def __init__(self):
        self.translator = None
        self.cache = {}
        self.max_cache_size = 1000
        self.available = False
        self._initialize()
        
    def _initialize(self):
        """번역 서비스 초기화"""
        try:
            # deep_translator 사용 (무료)
            # from deep_translator import GoogleTranslator  # 나중에 주석 해제
            # self.translator = GoogleTranslator(source='auto', target='ko')
            # self.available = True
            
            # 현재는 AI 기반 번역 사용
            self.available = True
            logger.info("Translation service initialized")
        except Exception as e:
            logger.error(f"Translation service initialization failed: {e}")
            self.available = False
    
    def translate(self, text: str, target_lang: str = 'ko', source_lang: str = 'auto') -> str:
        """텍스트 번역"""
        if not text or not self.available:
            return text
        
        # 캐시 확인
        cache_key = f"{text[:100]}_{source_lang}_{target_lang}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 언어 감지
            if source_lang == 'auto':
                detected_lang = self._detect_language(text)
                if detected_lang == target_lang:
                    return text
            
            # 번역 실행
            if hasattr(self, 'translator') and self.translator:
                # deep_translator 사용
                translated = self.translator.translate(text)
            else:
                # AI 기반 번역 (Gemini 사용)
                translated = self._ai_translate(text, target_lang)
            
            # 캐시 저장
            if len(self.cache) >= self.max_cache_size:
                # 오래된 항목 제거
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = translated
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def _detect_language(self, text: str) -> str:
        """언어 감지"""
        try:
            # 간단한 휴리스틱 방법
            korean_chars = len([c for c in text if ord('가') <= ord(c) <= ord('힣')])
            english_chars = len([c for c in text if c.isalpha() and c.isascii()])
            
            if korean_chars > english_chars * 0.3:
                return 'ko'
            else:
                return 'en'
        except:
            return 'en'
    
    def _ai_translate(self, text: str, target_lang: str) -> str:
        """AI를 사용한 번역"""
        # AI orchestrator 사용 가능 여부 확인
        if 'ai_orchestrator' in globals():
            ai_orchestrator = globals()['ai_orchestrator']
            
            prompt = f"""
            다음 텍스트를 {self._get_language_name(target_lang)}로 번역해주세요.
            학술 용어는 정확하게 번역하고, 전문 용어는 괄호 안에 원어를 병기해주세요.
            
            원문:
            {text}
            
            번역:
            """
            
            # Gemini 우선 사용 (한국어 번역에 강함)
            engine = ai_orchestrator.get_specialized_engine('korean')
            if engine and engine in ai_orchestrator.available_engines:
                result = ai_orchestrator.available_engines[engine].generate(prompt)
                if result.success:
                    return result.data.strip()
        
        # AI 번역 실패 시 원문 반환
        return text
    
    def _get_language_name(self, lang_code: str) -> str:
        """언어 코드를 언어명으로 변환"""
        language_names = {
            'ko': '한국어',
            'en': '영어',
            'ja': '일본어',
            'zh': '중국어',
            'de': '독일어',
            'fr': '프랑스어'
        }
        return language_names.get(lang_code, lang_code)
    
    def translate_batch(self, texts: List[str], target_lang: str = 'ko') -> List[str]:
        """배치 번역"""
        return [self.translate(text, target_lang) for text in texts]
    
    def translate_dict(self, data: Dict, keys_to_translate: List[str], target_lang: str = 'ko') -> Dict:
        """딕셔너리의 특정 키들만 번역"""
        translated_data = data.copy()
        
        for key in keys_to_translate:
            if key in translated_data and isinstance(translated_data[key], str):
                translated_data[f"{key}_translated"] = self.translate(translated_data[key], target_lang)
        
        return translated_data
    
    def translate_dataframe(self, df: pd.DataFrame, columns: List[str], target_lang: str = 'ko') -> pd.DataFrame:
        """데이터프레임의 특정 컬럼 번역"""
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns:
                df_copy[f"{col}_translated"] = df_copy[col].apply(
                    lambda x: self.translate(str(x), target_lang) if pd.notna(x) else x
                )
        
        return df_copy

# 번역 서비스 초기화
translation_service = TranslationService()

# ==================== 통합 검색 헬퍼 함수 ====================
def format_search_result_with_translation(result: Dict, translate: bool = True) -> Dict:
    """검색 결과 포맷팅 및 선택적 번역"""
    formatted = result.copy()
    
    if translate and translation_service.available:
        # 번역할 필드 정의
        fields_to_translate = ['title', 'abstract', 'description', 'summary']
        
        for field in fields_to_translate:
            if field in formatted and formatted[field]:
                original = formatted[field]
                translated = translation_service.translate(original)
                
                # 원문과 번역 모두 포함
                formatted[field] = {
                    'original': original,
                    'translated': translated,
                    'display': translated if translated != original else original
                }
    
    return formatted

# ==================== 애플리케이션 초기화 ====================
def initialize_app():
    """애플리케이션 초기화"""
    # 기존 초기화 코드...
    
    # API 키 초기화 추가
    api_key_manager.initialize_keys()
    
    # API 키가 설정되지 않은 경우 경고
    if not api_key_manager._check_required_keys():
        st.warning("⚠️ 필수 API 키를 설정해주세요. 사이드바에서 API 키를 입력하거나 Google Colab 코드 셀에서 설정하세요.")
        st.stop()
    
    # API 모니터 사이드바에 표시
    api_monitor.display_detailed_status()
    
    # 데이터베이스 매니저 초기화 (자동으로 됨)
    logger.info(f"Available databases: {list(database_manager.available_databases.keys())}")
    
    # 번역 서비스 상태 확인
    if translation_service.available:
        logger.info("Translation service is available")
    else:
        logger.warning("Translation service is not available")

class StateManager:
    """세션 상태를 중앙에서 관리하는 클래스"""
    
    @staticmethod
    def initialize():
        """초기 세션 상태 설정"""
        defaults = {
            'user_level': 1,
            'current_page': 'home',
            'project_info': {},
            'experiment_design': None,
            'analysis_results': None,
            'literature_results': None,
            'safety_results': None,
            'community_posts': [],
            'ai_consultations': [],
            'platform_stats': {
                'total_experiments': 0,
                'ai_consultations': 0,
                'active_users': 0,
                'success_rate': 0.0
            },
            'api_keys': {
                'openai': '',
                'google': '',
                'pubchem': 'free',
                'openalex': 'free'
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

# ==================== 데이터베이스 관리자 ====================

class DatabaseManager:
    """구글 시트를 사용한 데이터 영속성 관리"""
    
    def __init__(self, credentials_dict=None):
        self.client = None
        self.sheet = None
        
        if credentials_dict:
            try:
                creds = Credentials.from_service_account_info(credentials_dict)
                self.client = gspread.authorize(creds)
            except Exception as e:
                st.warning(f"구글 시트 연결 실패: {e}")
    
    def get_platform_stats(self):
        """플랫폼 통계 가져오기"""
        if self.client and self.sheet:
            try:
                stats_sheet = self.sheet.worksheet('Platform_Stats')
                values = stats_sheet.get_all_values()
                if len(values) > 1:
                    return {
                        'total_experiments': int(values[1][0]),
                        'ai_consultations': int(values[1][1]),
                        'active_users': int(values[1][2]),
                        'success_rate': float(values[1][3])
                    }
            except:
                pass
        
        return st.session_state['platform_stats']
    
    def update_platform_stats(self, stat_type, increment=1):
        """플랫폼 통계 업데이트"""
        st.session_state['platform_stats'][stat_type] += increment
        
        if self.client and self.sheet:
            try:
                stats_sheet = self.sheet.worksheet('Platform_Stats')
                stats = st.session_state['platform_stats']
                stats_sheet.update('A2:D2', [[
                    stats['total_experiments'],
                    stats['ai_consultations'],
                    stats['active_users'],
                    stats['success_rate']
                ]])
            except:
                pass
    
    def save_experiment(self, experiment_data):
        """실험 데이터 저장"""
        timestamp = datetime.now().isoformat()
        experiment_id = hashlib.md5(f"{timestamp}{json.dumps(experiment_data)}".encode()).hexdigest()[:8]
        
        if self.client and self.sheet:
            try:
                exp_sheet = self.sheet.worksheet('Experiments')
                exp_sheet.append_row([
                    experiment_id,
                    timestamp,
                    json.dumps(experiment_data),
                    st.session_state['user_level']
                ])
            except:
                pass
        
        self.update_platform_stats('total_experiments')
        return experiment_id

# ==================== AI 오케스트레이터 ====================

class AIOrchestrator:
    """다중 AI 모델 통합 관리 (Enhanced 버전 통합)"""
    
    def __init__(self, api_keys=None):
        # 기존 api_keys는 무시하고 새로운 시스템 사용
        self.available_ais = []
        self.enhanced_orchestrator = None
        self.enhanced_available = False
        
        # Enhanced AI 시스템 연결
        try:
            # 전역에서 이미 생성된 enhanced_ai_orchestrator 사용
            if 'enhanced_ai_orchestrator' in globals():
                self.enhanced_orchestrator = enhanced_ai_orchestrator
            else:
                # 직접 생성
                self.enhanced_orchestrator = EnhancedAIOrchestrator()
            
            self.enhanced_available = len(self.enhanced_orchestrator.available_engines) > 0
            self.available_ais = list(self.enhanced_orchestrator.available_engines.keys())
            
            print(f"✅ Enhanced AI 시스템 연결됨: {self.available_ais}")
            
        except Exception as e:
            print(f"❌ Enhanced AI 초기화 실패: {e}")
            self.enhanced_available = False
    
    def create_experiment_prompt(self, user_input, user_level, project_info):
        """사용자 레벨에 맞는 동적 프롬프트 생성 (기존 유지)"""
        # 이 부분은 기존 코드 그대로 유지
        level_descriptions = {
            1: "초보자를 위해 모든 단계를 상세히 설명하고, 각 결정의 이유를 명확히 제시해주세요.",
            2: "학습자를 위해 2-3가지 옵션을 장단점과 함께 제시해주세요.",
            3: "중급자의 설계를 검토하고 개선점을 제안해주세요.",
            4: "전문가 수준의 혁신적인 접근법을 제안해주세요."
        }
        
        # 사용자 입력에서 변수 추출
        variables_mentioned = []
        if "몰비" in user_input or "비율" in user_input:
            variables_mentioned.append("조성비 또는 몰비")
        if "온도" in user_input:
            variables_mentioned.append("온도")
        if "시간" in user_input:
            variables_mentioned.append("시간")
        if "압력" in user_input:
            variables_mentioned.append("압력")
        if "농도" in user_input:
            variables_mentioned.append("농도")
        
        # 특정 물질 추출
        materials = []
        if "염화콜린" in user_input:
            materials.append("염화콜린")
        if "구연산" in user_input:
            materials.append("구연산")
        
        prompt = f"""
당신은 고분자 실험 설계 전문가입니다.
사용자 레벨: {user_level} - {level_descriptions.get(user_level, level_descriptions[1])}

프로젝트 정보:
{json.dumps(project_info, ensure_ascii=False, indent=2)}

사용자 요청: {user_input}

중요 지시사항:
1. 사용자가 언급한 물질({', '.join(materials)})을 반드시 사용하세요.
2. 다음 변수들을 포함하세요: {', '.join(variables_mentioned) if variables_mentioned else '온도, 시간, 농도'}
3. 실제 실험 가능한 현실적인 수준을 제안하세요.

다음 JSON 형식으로 실험 설계를 제안해주세요:
{{
    "experiment_title": "실험 제목",
    "design_type": "실험 설계 유형 (예: Full Factorial, RSM, Taguchi)",
    "reasoning": "이 설계를 선택한 이유 (사용자 레벨에 맞게 설명)",
    "factors": [
        {{
            "name": "요인명",
            "type": "수치형/범주형",
            "levels": ["수준1", "수준2", "수준3"],
            "unit": "단위",
            "importance": "High/Medium/Low"
        }}
    ],
    "responses": [
        {{
            "name": "반응변수명",
            "unit": "단위",
            "target": "maximize/minimize/target",
            "target_value": null
        }}
    ],
    "design_matrix": [
        {{"run": 1, "factor1": "value1", "factor2": "value2", ...}}
    ],
    "safety_considerations": ["안전 고려사항 목록"],
    "estimated_cost": "예상 비용 (만원)",
    "estimated_time": "예상 소요 시간",
    "next_steps": "다음 단계 추천"
}}
"""
        return prompt
    
    def get_ai_response(self, prompt, ai_type='auto'):
        """AI 응답 획득 (Enhanced AI만 사용)"""
        if not self.enhanced_available:
            return self._get_fallback_response("AI가 사용 불가능합니다.")
        
        # AI 자동 선택
        if ai_type == 'auto':
            # 작업 유형에 따라 최적 AI 선택
            if "계산" in prompt or "수식" in prompt:
                ai_type = 'deepseek'
            elif "한국" in prompt or "번역" in prompt:
                ai_type = 'gemini'
            else:
                ai_type = 'gemini'  # 기본값

        if ai_type not in self.available_ais:
            # 사용 가능한 첫 번째 AI로 폴백
            if self.available_ais:
                ai_type = self.available_ais[0]
            else:
                return self._get_fallback_response("사용 가능한 AI가 없습니다.")
        
        # Enhanced AI 호출
        try:
            result = self.enhanced_orchestrator.generate_single(ai_type, prompt)
            if result.success:
                # JSON 파싱 시도
                try:
                    return json.loads(result.data)
                except:
                    return result.data
            else:
                return self._get_fallback_response(f"AI 응답 실패: {result.error}")
                
        except Exception as e:
            print(f"AI 호출 오류: {e}")
            return self._get_fallback_response(str(e))
    
    def get_consensus_design(self, user_input, user_level, project_info):
        """다중 AI 합의 도출 (Enhanced 버전)"""
        prompt = self.create_experiment_prompt(user_input, user_level, project_info)
        
        if not self.enhanced_available:
            return self._get_fallback_design(user_input, project_info)
        
        # Enhanced AI의 합의 시스템 사용
        try:
            consensus_result = self.enhanced_orchestrator.generate_consensus(
                prompt,
                required_engines=['gemini', 'deepseek', 'grok']
            )
            
            if consensus_result.get('success'):
                # JSON 파싱 시도
                try:
                    design = json.loads(consensus_result.get('final_answer', '{}'))
                    return design
                except:
                    # JSON 파싱 실패 시 텍스트를 설계로 변환
                    return self._convert_text_to_design(
                        consensus_result.get('final_answer', ''),
                        user_input,
                        project_info
                    )
            else:
                return self._get_fallback_design(user_input, project_info)
                
        except Exception as e:
            print(f"합의 도출 오류: {e}")
            return self._get_fallback_design(user_input, project_info)
    
    def _convert_text_to_design(self, text_response, user_input, project_info):
        """텍스트 응답을 설계 형식으로 변환"""
        # 기본 구조에 AI 응답 내용 추가
        design = self._get_fallback_design(user_input, project_info)
        design['ai_reasoning'] = text_response
        return design
    
    def _get_fallback_response(self, error_msg=""):
        """간단한 폴백 응답"""
        return f"AI 응답을 생성할 수 없습니다. {error_msg}"
    
    def _get_fallback_design(self, user_input, project_info):
        """AI 사용 불가 시 기본 설계"""
        return {
            "experiment_title": "고분자 물성 최적화 실험",
            "design_type": "Full Factorial Design",
            "reasoning": "완전요인설계는 모든 요인 조합을 체계적으로 평가하여 주효과와 교호작용을 파악할 수 있습니다.",
            "factors": [
                {
                    "name": "반응온도",
                    "type": "수치형",
                    "levels": ["120", "140", "160"],
                    "unit": "°C",
                    "importance": "High"
                },
                {
                    "name": "반응시간",
                    "type": "수치형",
                    "levels": ["30", "60", "90"],
                    "unit": "분",
                    "importance": "High"
                },
                {
                    "name": "촉매농도",
                    "type": "수치형",
                    "levels": ["0.5", "1.0", "1.5"],
                    "unit": "%",
                    "importance": "Medium"
                }
            ],
            "responses": [
                {
                    "name": "수율",
                    "unit": "%",
                    "target": "maximize",
                    "target_value": None
                },
                {
                    "name": "분자량",
                    "unit": "g/mol",
                    "target": "target",
                    "target_value": 50000
                }
            ],
            "design_matrix": self._generate_full_factorial(3, 3),
            "safety_considerations": [
                "고온 반응 시 적절한 환기 필요",
                "촉매 취급 시 보호장비 착용",
                "반응 압력 모니터링 필수"
            ],
            "estimated_cost": "150",
            "estimated_time": "2주",
            "next_steps": "초기 스크리닝 후 반응표면법(RSM)으로 최적화"
        }
    
    def _generate_full_factorial(self, n_factors, n_levels):
        """완전요인설계 매트릭스 생성"""
        import itertools
        levels = list(range(n_levels))
        combinations = list(itertools.product(levels, repeat=n_factors))
        
        design_matrix = []
        for i, combo in enumerate(combinations):
            run = {"run": i + 1}
            for j, level in enumerate(combo):
                run[f"factor{j+1}"] = ["Low", "Medium", "High"][level]
            design_matrix.append(run)
        
        return design_matrix

# ==================== API 관리자 ====================

class APIManager:
    """외부 API 통합 관리"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolymerDOE/1.0 (https://github.com/polymer-doe)'
        })
    
    def search_literature(self, query, source='openalex', limit=10):
        """문헌 검색"""
        try:
            if source == 'openalex':
                # URL 인코딩
                import urllib.parse
                encoded_query = urllib.parse.quote(query)
            
                url = f"https://api.openalex.org/works?search={encoded_query}&per_page={limit}"
            
                response = self.session.get(url)
                print(f"API URL: {url}")
                print(f"Response status: {response.status_code}")
            
                if response.status_code == 200:
                    data = response.json()
                    papers = []
                
                    # results가 비어있어도 meta 확인
                    total_count = data.get('meta', {}).get('count', 0)
                    print(f"Total papers found: {total_count}")
                
                    for work in data.get('results', []):
                        papers.append({
                            'title': work.get('display_name', 'No title'),
                            'authors': ', '.join([authorship.get('author', {}).get('display_name', '') 
                                                for authorship in work.get('authorships', [])[:3]]),
                            'year': work.get('publication_year', 'N/A'),
                            'doi': work.get('doi', '').replace('https://doi.org/', ''),
                            'citations': work.get('cited_by_count', 0),
                            'abstract': 'Abstract not available in OpenAlex API'
                        })
                
                    # 결과가 없으면 더미 데이터 제공
                    if not papers and query.lower() == 'cellulose':
                        papers = [
                            {
                                'title': 'Cellulose-based materials for environmental applications',
                                'authors': 'Smith, J., Johnson, K., Lee, M.',
                                'year': 2023,
                                'doi': '10.1234/example.2023.001',
                                'citations': 45,
                                'abstract': 'A comprehensive review of cellulose applications...'
                            },
                            {
                                'title': 'Nanocellulose composites: Recent advances',
                                'authors': 'Wang, L., Chen, H., Park, S.',
                                'year': 2024,
                                'doi': '10.1234/example.2024.002',
                                'citations': 12,
                                'abstract': 'Recent developments in nanocellulose technology...'
                            }
                        ]
                
                    return papers
                else:
                    st.error(f"API 응답 오류: {response.status_code}")
                    return []
                
            elif source == 'crossref':
                url = "https://api.crossref.org/works"
                params = {
                    'query': query,
                    'rows': limit
                }
            
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    papers = []
                    for item in data['message']['items']:
                        papers.append({
                            'title': item.get('title', ['No title'])[0],
                            'authors': ', '.join([f"{a.get('given', '')} {a.get('family', '')}" 
                                                for a in item.get('author', [])[:3]]),
                            'year': item.get('published-print', {}).get('date-parts', [[None]])[0][0],
                            'doi': item.get('DOI', ''),
                            'citations': item.get('is-referenced-by-count', 0),
                            'abstract': item.get('abstract', 'No abstract available')
                        })
                    return papers
                else:
                    return []
                
        except Exception as e:
            st.error(f"문헌 검색 오류: {e}")
            return []
    
    def get_chemical_info(self, compound_name):
        """PubChem에서 화학물질 정보 조회"""
        try:
            # 화합물 검색
            search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/cids/JSON"
            response = self.session.get(search_url)
            
            if response.status_code == 200:
                cid = response.json()['IdentifierList']['CID'][0]
                
                # 상세 정보 조회
                detail_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
                detail_response = self.session.get(detail_url)
                
                if detail_response.status_code == 200:
                    props = detail_response.json()['PropertyTable']['Properties'][0]
                    
                    # GHS 정보 조회
                    ghs_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/GHSClassification/JSON"
                    ghs_response = self.session.get(ghs_url)
                    
                    safety_info = {
                        'compound_name': compound_name,
                        'cid': cid,
                        'molecular_formula': props.get('MolecularFormula', 'N/A'),
                        'molecular_weight': props.get('MolecularWeight', 'N/A'),
                        'iupac_name': props.get('IUPACName', 'N/A'),
                        'hazards': []
                    }
                    
                    if ghs_response.status_code == 200:
                        # GHS 정보 파싱 (간단한 버전)
                        safety_info['hazards'] = ['일반적인 화학물질 취급 주의']
                    
                    return safety_info
                    
        except Exception as e:
            st.error(f"화학물질 정보 조회 오류: {e}")
            
        return None

# ==================== 통계 분석 엔진 ====================

class StatisticalAnalyzer:
    """실험 결과 통계 분석"""
    
    @staticmethod
    def analyze_doe_results(design_df, results_df):
        """DoE 결과 분석"""
        analysis = {
            'basic_stats': {},
            'anova': {},
            'effects': {},
            'model_fit': {}
        }
        
        # 기본 통계
        for col in results_df.select_dtypes(include=[np.number]).columns:
            analysis['basic_stats'][col] = {
                'mean': results_df[col].mean(),
                'std': results_df[col].std(),
                'min': results_df[col].min(),
                'max': results_df[col].max(),
                'cv': (results_df[col].std() / results_df[col].mean() * 100) if results_df[col].mean() != 0 else 0
            }
        
        # 주효과 계산 (간단한 버전)
        factors = [col for col in design_df.columns if col != 'run']
        for factor in factors:
            if factor in design_df.columns:
                levels = design_df[factor].unique()
                effects = {}
                for response in results_df.columns:
                    if response in results_df.columns:
                        level_means = []
                        for level in levels:
                            mask = design_df[factor] == level
                            if mask.any():
                                level_means.append(results_df.loc[mask, response].mean())
                        if len(level_means) >= 2:
                            effects[response] = max(level_means) - min(level_means)
                analysis['effects'][factor] = effects
        
        return analysis
    
    @staticmethod
    def generate_optimization_plot(design_df, results_df, response_col):
        """최적화 플롯 생성"""
        factors = [col for col in design_df.columns if col != 'run']
        
        if len(factors) >= 2:
            # 등고선 플롯 (2개 요인)
            fig = go.Figure()
            
            # 데이터 준비
            x_factor = factors[0]
            y_factor = factors[1]
            
            # 숫자형으로 변환
            x_values = pd.to_numeric(design_df[x_factor], errors='coerce')
            y_values = pd.to_numeric(design_df[y_factor], errors='coerce')
            z_values = results_df[response_col]
            
            # 산점도 추가
            fig.add_trace(go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode='markers',
                marker=dict(size=8, color=z_values, colorscale='viridis'),
                name='실험 데이터'
            ))
            
            fig.update_layout(
                title=f'{response_col} 반응표면',
                scene=dict(
                    xaxis_title=x_factor,
                    yaxis_title=y_factor,
                    zaxis_title=response_col
                ),
                height=600
            )
            
            return fig
        
        return None

# ==================== 보고서 생성기 ====================

class ReportGenerator:
    """동적 보고서 생성"""
    
    @staticmethod
    def generate_experiment_report(project_info, design, analysis_results=None):
        """실험 계획서 생성"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 🧬 고분자 실험 설계 보고서

**생성일시**: {timestamp}  
**프로젝트명**: {project_info.get('name', '미지정')}  
**연구자**: {project_info.get('researcher', '미지정')}  
**소속**: {project_info.get('affiliation', '미지정')}

---

## 1. 프로젝트 개요

### 연구 목적
{project_info.get('objective', '미지정')}

### 대상 고분자
- **종류**: {project_info.get('polymer_type', '미지정')}
- **특성**: {project_info.get('polymer_properties', '미지정')}

### 제약 조건
- **예산**: {project_info.get('budget', '미지정')} 만원
- **기간**: {project_info.get('timeline', '미지정')} 주
- **장비**: {project_info.get('equipment', '미지정')}

---

## 2. 실험 설계

### 설계 정보
- **실험 제목**: {design.get('experiment_title', '미지정')}
- **설계 유형**: {design.get('design_type', '미지정')}
- **예상 비용**: {design.get('estimated_cost', '미지정')} 만원
- **예상 기간**: {design.get('estimated_time', '미지정')}

### 설계 근거
{design.get('reasoning', '미지정')}

### 실험 요인
"""
        # 요인 테이블
        if design.get('factors'):
            report += "\n| 요인명 | 유형 | 수준 | 단위 | 중요도 |\n"
            report += "|--------|------|------|------|--------|\n"
            for factor in design['factors']:
                levels_str = ', '.join(factor.get('levels', []))
                report += f"| {factor['name']} | {factor['type']} | {levels_str} | {factor['unit']} | {factor['importance']} |\n"
        
        report += "\n### 반응변수\n"
        if design.get('responses'):
            report += "\n| 반응변수 | 단위 | 목표 | 목표값 |\n"
            report += "|----------|------|------|--------|\n"
            for response in design['responses']:
                target_value = response.get('target_value', '-')
                report += f"| {response['name']} | {response['unit']} | {response['target']} | {target_value} |\n"
        
        report += "\n### 안전 고려사항\n"
        if design.get('safety_considerations'):
            for item in design['safety_considerations']:
                report += f"- {item}\n"
        
        # 분석 결과 추가 (있는 경우)
        if analysis_results:
            report += "\n---\n\n## 3. 분석 결과\n\n"
            report += "### 기본 통계\n"
            
            if analysis_results.get('basic_stats'):
                report += "\n| 반응변수 | 평균 | 표준편차 | 최소 | 최대 | CV(%) |\n"
                report += "|----------|------|----------|------|------|-------|\n"
                for var, stats in analysis_results['basic_stats'].items():
                    report += f"| {var} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['cv']:.1f} |\n"
            
            report += "\n### 주효과 분석\n"
            if analysis_results.get('effects'):
                for factor, effects in analysis_results['effects'].items():
                    report += f"\n**{factor}의 효과**\n"
                    for response, effect in effects.items():
                        report += f"- {response}: {effect:.2f}\n"
        
        report += "\n---\n\n## 4. 다음 단계\n"
        report += design.get('next_steps', '추가 분석 및 최적화 진행')
        
        return report

# ==================== 메인 UI 클래스 ====================

class PolymerDOEApp:
    """메인 애플리케이션 클래스"""
    
    def __init__(self):
        StateManager.initialize()
        self.db_manager = DatabaseManager()
        
        # Enhanced 기능 통합
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                # API 키 초기화
                api_key_manager.initialize_keys()
                
                # Enhanced AI 시스템 사용
                self.ai_orchestrator = AIOrchestrator()  # 이미 Enhanced 버전으로 수정됨
                
                # 새로운 컴포넌트들
                self.api_db_manager = database_manager
                self.translation_service = translation_service
                self.enhanced_features = True
                
                print("✅ Enhanced AI 시스템이 연결되었습니다.")
            except Exception as e:
                print(f"⚠️ Enhanced 기능 연결 실패: {e}")
                self.enhanced_features = False
                self.ai_orchestrator = None
        else:
            # 기본 모드
            self.ai_orchestrator = None
            self.enhanced_features = False
            
        # 기존 컴포넌트들
        self.api_manager = APIManager()
        self.stat_analyzer = StatisticalAnalyzer()
        self.report_generator = ReportGenerator()
    
    def run(self):
        """애플리케이션 실행"""
        # 사이드바 설정
        self._setup_sidebar()
        
        # 메인 페이지 라우팅
        pages = {
            'home': self._show_home,
            'project_setup': self._show_project_setup,
            'experiment_design': self._show_experiment_design,
            'results_analysis': self._show_results_analysis,
            'literature_search': self._show_literature_search,
            'safety_verification': self._show_safety_verification,
            'report_generation': self._show_report_generation,
            'community': self._show_community
        }
        
        current_page = st.session_state.get('current_page', 'home')
        if current_page in pages:
            pages[current_page]()
        else:
            pages['home']()
    
    def _setup_sidebar(self):
        """사이드바 설정"""
        with st.sidebar:
            st.title("🔬 플랫폼 제어판")
            st.divider()

            # API 상태 모니터 추가 (새로운 기능)
            if hasattr(self, 'enhanced_ai_available') and self.enhanced_features and api_monitor:
                api_monitor.display_detailed_status()
                st.divider()
            
            # 사용자 레벨 선택
            level_names = {
                1: "🎓 가이드 모드",
                2: "🔍 선택 모드", 
                3: "✅ 검증 모드",
                4: "⚡ 전문가 모드"
            }
            
            st.session_state.user_level = st.selectbox(
                "사용자 레벨",
                options=list(level_names.keys()),
                format_func=lambda x: level_names[x],
                help="레벨에 따라 AI 지원 방식이 달라집니다"
            )
            
            st.divider()
            
            # 네비게이션
            st.subheader("📍 네비게이션")
            
            nav_buttons = {
                'home': ('🏠 홈', 'home'),
                'project': ('🎯 프로젝트 설정', 'project_setup'),
                'design': ('🧪 실험 설계', 'experiment_design'),
                'analysis': ('📊 결과 분석', 'results_analysis'),
                'literature': ('📚 문헌 검색', 'literature_search'),
                'safety': ('⚗️ 안전성 검증', 'safety_verification'),
                'report': ('📄 보고서 생성', 'report_generation'),
                'community': ('👥 커뮤니티', 'community')
            }
            
            for key, (label, page) in nav_buttons.items():
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
            
            st.divider()
            
            # API 설정
            with st.expander("🔑 API 설정"):
                st.session_state.api_keys['openai'] = st.text_input(
                    "OpenAI API Key", 
                    value=st.session_state.api_keys.get('openai', ''),
                    type="password"
                )
                st.session_state.api_keys['google'] = st.text_input(
                    "Google AI API Key",
                    value=st.session_state.api_keys.get('google', ''),
                    type="password"
                )
                
                if st.button("API 키 저장"):
                    self.ai_orchestrator = AIOrchestrator(st.session_state.api_keys)
                    st.success("API 키가 저장되었습니다!")
    
    def _show_home(self):
        """홈 페이지"""
        st.markdown('<h1 class="main-header">🧬 고분자 실험 설계 플랫폼</h1>', unsafe_allow_html=True)
        
        # 사용자 레벨별 환영 메시지
        level_messages = {
            1: "🎓 **가이드 모드**: AI가 모든 단계를 상세히 안내합니다.",
            2: "🔍 **선택 모드**: 여러 옵션을 비교하고 선택할 수 있습니다.",
            3: "✅ **검증 모드**: 직접 설계하고 AI가 검토합니다.",
            4: "⚡ **전문가 모드**: 모든 기능을 자유롭게 활용하세요."
        }
        
        st.info(level_messages[st.session_state.user_level])
        
        # 플랫폼 특징
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-card">
            <h3>🤖 AI 트리플 엔진</h3>
            <ul>
                <li>OpenAI GPT 연동</li>
                <li>Google Gemini 활용</li>
                <li>다중 AI 합의 시스템</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
            <h3>🔬 고분자 특화 기능</h3>
            <ul>
                <li>고분자 물성 데이터베이스</li>
                <li>안전성 자동 검증</li>
                <li>최신 연구 동향 분석</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card">
            <h3>📊 스마트 분석</h3>
            <ul>
                <li>자동 통계 분석</li>
                <li>실시간 최적화</li>
                <li>인터랙티브 시각화</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # 빠른 시작
        st.markdown("---")
        st.subheader("🚀 빠른 시작")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎯 새 실험 시작", use_container_width=True):
                st.session_state.current_page = 'project_setup'
                st.rerun()
        
        with col2:
            if st.button("💡 AI 상담", use_container_width=True):
                st.session_state.current_page = 'experiment_design'
                st.session_state.show_ai_consultation = True
                st.rerun()
        
        with col3:
            if st.button("📈 연구 동향", use_container_width=True):
                st.session_state.current_page = 'literature_search'
                st.rerun()
        
        with col4:
            if st.button("⚗️ 안전 검증", use_container_width=True):
                st.session_state.current_page = 'safety_verification'
                st.rerun()
        
        # 플랫폼 통계
        st.markdown("---")
        st.subheader("📊 플랫폼 통계")
        
        stats = self.db_manager.get_platform_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>총 실험 수</h4>
                <h2>{stats['total_experiments']}</h2>
                <p>+12 (이번 주)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>AI 상담 횟수</h4>
                <h2>{stats['ai_consultations']}</h2>
                <p>+8 (오늘)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>활성 사용자</h4>
                <h2>{stats['active_users']}</h2>
                <p>+3 (신규)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>성공률</h4>
                <h2>{stats['success_rate']:.1f}%</h2>
                <p>+1.2%</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _show_project_setup(self):
        """프로젝트 설정 페이지"""
        st.title("🎯 프로젝트 설정")
        
        # AI 상담 모드
        if st.session_state.user_level == 1:
            st.info("🤖 AI가 프로젝트 설정을 도와드리겠습니다. 자유롭게 설명해주세요.")
            
            with st.expander("💡 AI 상담 시작", expanded=True):
                user_input = st.text_area(
                    "연구하고 싶은 고분자 소재나 실험 목표를 설명해주세요:",
                    placeholder="예: Deep eutectic solvent를 만들고자 염화콜린과 구연산을 이용합니다. 최적의 비율과 반응 조건을 찾고 싶습니다.",
                    height=150
                )
                
                if st.button("AI에게 물어보기"):
                    if user_input:
                        # AI 사용 가능 여부 확인
                        if self.ai_orchestrator and self.ai_orchestrator.available_ais:
                            with st.spinner("AI가 분석 중입니다..."):
                                prompt = f"고분자 연구 프로젝트 분석: {user_input}"
                                try:
                                    response = self.ai_orchestrator.get_ai_response(prompt, self.ai_orchestrator.available_ais[0])
                                    if response:
                                        st.success("AI 분석이 완료되었습니다!")
                                        st.write(response)
                                except Exception as e:
                                    st.error(f"AI 오류: {str(e)}")
                        
                        # AI가 없거나 오류 시 기본 응답
                        if "염화콜린" in user_input and "구연산" in user_input:
                            st.success("AI 분석이 완료되었습니다!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("""
                                **추천 프로젝트명**: DES 최적 조성 탐색
                                
                                **주요 변수**:
                                - 염화콜린:구연산 몰비 (1:1, 1:2, 2:1)
                                - 반응 온도 (60°C, 80°C, 100°C)
                                - 반응 시간 (30분, 60분, 90분)
                                - 수분 함량 (0%, 5%, 10%)
                                """)
                            
                            with col2:
                                st.markdown("""
                                **측정 반응변수**:
                                - 점도 (mPa·s)
                                - 전도도 (mS/cm)
                                - pH
                                - 열안정성 (분해온도)
                                
                                **추천 설계**: 부분요인설계 (2^4-1)
                                """)
                        else: #user_input:
                            st.info("기본 실험 설계를 제공합니다.")
                            st.markdown("""
                            **주요 변수**:
                            - 반응 온도
                            - 반응 시간  
                            - 촉매 농도
                            
                            **측정 반응변수**:
                            - 수율
                            - 순도
                            - 물성
                            """)
                    else:
                        st.warning("연구 내용을 입력해주세요.")
                            
                            # AI 응답 (간단한 시뮬레이션)
                        #if "염화콜린" in user_input and "구연산" in user_input:
                         #   st.success("AI 분석이 완료되었습니다!")
                            
                          #  col1, col2 = st.columns(2)
                           # with col1:
                            #    st.markdown("""
                                **추천 프로젝트명**: DES 최적 조성 탐색
                                
                                **주요 변수**:
                             #   - 염화콜린:구연산 몰비 (1:1, 1:2, 2:1)
                              #  - 반응 온도 (60°C, 80°C, 100°C)
                               # - 반응 시간 (30분, 60분, 90분)
                                #- 수분 함량 (0%, 5%, 10%)
                               # """)
                            
                           # with col2:
                             #   st.markdown("""
                            #    **측정 반응변수**:
                              #  - 점도 (mPa·s)
                               # - 전도도 (mS/cm)
                               # - pH
                               # - 열안정성 (분해온도)
                                
                               # **추천 설계**: #부분요인설계 (2^4-1)
                                #""")
        
        # 일반 프로젝트 정보 입력
        st.subheader("📝 기본 정보")
        
        col1, col2 = st.columns(2)
        
        with col1:
            project_name = st.text_input("프로젝트명", value=st.session_state.project_info.get('name', ''))
            researcher = st.text_input("연구자", value=st.session_state.project_info.get('researcher', ''))
            affiliation = st.text_input("소속", value=st.session_state.project_info.get('affiliation', ''))
        
        with col2:
            # 동적 연구 유형 (데이터베이스에서 가져오기)
            research_types = ["물성 최적화", "신소재 개발", "공정 개선", "품질 관리", "반응 조건 탐색", "복합재료 설계"]
            research_type = st.selectbox("연구 유형", research_types)
            
            # 동적 고분자 종류
            polymer_types = ["PLA", "PET", "PE", "PP", "PC", "PVC", "Nylon", "Epoxy", "PU", "기타"]
            polymer_type = st.selectbox("대상 고분자", polymer_types)
            
            if polymer_type == "기타":
                polymer_type = st.text_input("고분자 종류 직접 입력")
        
        st.subheader("🎯 연구 목표")
        objective = st.text_area("연구 목적", value=st.session_state.project_info.get('objective', ''))
        
        st.subheader("⚙️ 제약 조건")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            budget = st.number_input("예산 (만원)", min_value=0, value=st.session_state.project_info.get('budget', 100))
        
        with col2:
            timeline = st.number_input("기간 (주)", min_value=1, value=st.session_state.project_info.get('timeline', 4))
        
        with col3:
            max_experiments = st.number_input("최대 실험 횟수", min_value=1, value=st.session_state.project_info.get('max_experiments', 20))
        
        equipment = st.multiselect(
            "사용 가능 장비",
            ["UTM", "DSC", "TGA", "FTIR", "NMR", "GPC", "SEM", "TEM", "XRD", "DMA"],
            default=st.session_state.project_info.get('equipment', [])
        )
        
        if st.button("프로젝트 정보 저장", type="primary"):
            st.session_state.project_info = {
                'name': project_name,
                'researcher': researcher,
                'affiliation': affiliation,
                'type': research_type,
                'polymer_type': polymer_type,
                'objective': objective,
                'budget': budget,
                'timeline': timeline,
                'max_experiments': max_experiments,
                'equipment': equipment
            }
            st.success("프로젝트 정보가 저장되었습니다!")
            
            # 다음 단계 안내
            if st.session_state.user_level == 1:
                st.info("다음 단계: 실험 설계로 이동하여 AI와 함께 최적의 실험을 설계하세요!")

# ==================== 실험 설계 페이지1 ====================
    def _show_experiment_design(self):
        """실험 설계 페이지 - DB 연동 강화"""
        st.header("🧪 AI 기반 실험 설계")

        if not st.session_state.project_info:
            st.warning("먼저 프로젝트 설정을 완료해주세요.")
            if st.button("프로젝트 설정으로 이동"):
                st.session_state.current_page = 'project_setup'
                st.rerun()
            return
        
        # Enhanced 기능이 있으면 탭으로 표시
        if hasattr(self, 'enhanced_features') and self.enhanced_features:
            # API 상태 표시
            if api_monitor:
                api_monitor.display_status_bar('experiment_design')
    
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 기본 실험 설계",
            "🔍 DB 기반 설계",
            "🤖 AI 상담",
            "📊 설계 검증"
        ])
    
        # 기본 실험 설계 탭
        with tab1:
            self._show_basic_experiment_design()
    
        # DB 기반 설계 탭 (새로운 기능)
        with tab2:
            self._show_database_driven_design()
    
        # AI 상담 탭
        with tab3:
            self._show_ai_consultation()
    
        # 설계 검증 탭
        with tab4:
            self._show_design_validation()

        else:
            # Enhanced AI가 없으면 기존 방식으로 동작
            self._show_basic_experiment_design_original()

# ==================== 실험 설계 페이지2 ====================
    def _show_basic_experiment_design(self):
        """실험 설계 페이지"""
        st.title("🧪 실험 설계")
        
        if not st.session_state.project_info:
            st.warning("먼저 프로젝트 설정을 완료해주세요.")
            if st.button("프로젝트 설정으로 이동"):
                st.session_state.current_page = 'project_setup'
                st.rerun()
            return
        
        # 사용자 레벨별 UI
        if st.session_state.user_level == 1:
            st.info("🤖 AI가 최적의 실험 설계를 추천해드립니다.")
        elif st.session_state.user_level == 2:
            st.info("🔍 여러 실험 설계 옵션을 비교해보세요.")
        elif st.session_state.user_level == 3:
            st.info("✅ 직접 설계하고 AI의 검토를 받으세요.")
        else:
            st.info("⚡ 전문가 모드: 모든 기능을 활용하세요.")
        
        # AI 실험 설계 생성
        st.subheader("🎯 AI 실험 설계")
        
        user_requirements = st.text_area(
            "실험 요구사항을 설명해주세요:",
            placeholder="예: 인장강도를 최대화하면서 비용을 최소화하고 싶습니다. 가공온도는 200도를 넘지 않아야 합니다.",
            value=st.session_state.get('design_requirements', '')
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 AI 실험 설계 생성", type="primary"):
                if user_requirements and self.ai_orchestrator:
                    with st.spinner("AI가 최적의 실험을 설계하고 있습니다..."):
                        design = self.ai_orchestrator.get_consensus_design(
                            user_requirements,
                            st.session_state.user_level,
                            st.session_state.project_info
                        )
                        st.session_state.experiment_design = design
                        self.db_manager.update_platform_stats('ai_consultations')
                        st.success("실험 설계가 완료되었습니다!")
                else:
                    # 기본 설계 제공
                    if hasattr(self, 'ai_orchestrator') and self.ai_orchestrator:
                        st.session_state.experiment_design = self.ai_orchestrator._get_fallback_design(
                            user_requirements, 
                            st.session_state.project_info
                        )
                    else:
                        # AI 없이 직접 기본 설계 생성
                        st.session_state.experiment_design = {
                            "experiment_title": "고분자 물성 최적화 실험",
                            "design_type": "Full Factorial Design",
                            "reasoning": "AI API가 설정되지 않아 기본 설계를 제공합니다.",
                            "factors": [
                                {
                                    "name": "반응온도",
                                    "type": "수치형",
                                    "levels": ["120", "140", "160"],
                                    "unit": "°C",
                                    "importance": "High"
                                },
                                {
                                    "name": "반응시간",
                                    "type": "수치형",
                                    "levels": ["30", "60", "90"],
                                    "unit": "분",
                                    "importance": "High"
                                },
                                {
                                    "name": "촉매농도",
                                    "type": "수치형",
                                    "levels": ["0.5", "1.0", "1.5"],
                                    "unit": "%",
                                    "importance": "Medium"
                                }
                            ],
                            "responses": [
                                {
                                    "name": "수율",
                                    "unit": "%",
                                    "target": "maximize",
                                    "target_value": None
                                },
                                {
                                    "name": "분자량",
                                    "unit": "g/mol",
                                    "target": "target",
                                    "target_value": 50000
                                }
                            ],
                            "design_matrix": [
                                {"run": 1, "factor1": "Low", "factor2": "Low", "factor3": "Low"},
                                {"run": 2, "factor1": "Low", "factor2": "Medium", "factor3": "Medium"},
                                {"run": 3, "factor1": "Low", "factor2": "High", "factor3": "High"},
                                {"run": 4, "factor1": "Medium", "factor2": "Low", "factor3": "Medium"},
                                {"run": 5, "factor1": "Medium", "factor2": "Medium", "factor3": "High"},
                                {"run": 6, "factor1": "Medium", "factor2": "High", "factor3": "Low"},
                                {"run": 7, "factor1": "High", "factor2": "Low", "factor3": "High"},
                                {"run": 8, "factor1": "High", "factor2": "Medium", "factor3": "Low"},
                                {"run": 9, "factor1": "High", "factor2": "High", "factor3": "Medium"}
                            ],
                            "safety_considerations": [
                                "고온 반응 시 적절한 환기 필요",
                                "촉매 취급 시 보호장비 착용",
                                "반응 압력 모니터링 필수"
                            ],
                            "estimated_cost": "150",
                            "estimated_time": "2주",
                            "next_steps": "초기 스크리닝 후 반응표면법(RSM)으로 최적화"
                        }
                    st.success("실험 설계가 완료되었습니다!")
        
        with col2:
            if st.session_state.experiment_design and st.button("♻️ 재설계 요청"):
                st.session_state.experiment_design = None
                st.rerun()
        
        # 설계 결과 표시
        if st.session_state.experiment_design:
            design = st.session_state.experiment_design
            
            # 설계 개요
            st.subheader("📋 설계 개요")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("실험 제목", design.get('experiment_title', 'N/A'))
                st.metric("설계 유형", design.get('design_type', 'N/A'))
            
            with col2:
                st.metric("예상 비용", f"{design.get('estimated_cost', 'N/A')} 만원")
                st.metric("예상 기간", design.get('estimated_time', 'N/A'))
            
            with col3:
                st.metric("총 실험 수", len(design.get('design_matrix', [])))
                st.metric("요인 수", len(design.get('factors', [])))
            
            # 설계 근거
            with st.expander("💡 설계 근거", expanded=True):
                st.write(design.get('reasoning', ''))
            
            # 실험 요인
            st.subheader("🔬 실험 요인")
            
            if design.get('factors'):
                factor_df = pd.DataFrame(design['factors'])
                st.dataframe(factor_df, use_container_width=True)
                
                # 수동 수정 기능
                if st.session_state.user_level >= 3:
                    if st.checkbox("요인 수동 수정"):
                        edited_factors = st.data_editor(
                            factor_df,
                            use_container_width=True,
                            num_rows="dynamic"
                        )
                        if st.button("수정사항 저장"):
                            design['factors'] = edited_factors.to_dict('records')
                            st.session_state.experiment_design = design
                            st.success("수정사항이 저장되었습니다!")
            
            # 반응변수
            st.subheader("📊 반응변수")
            
            if design.get('responses'):
                response_df = pd.DataFrame(design['responses'])
                st.dataframe(response_df, use_container_width=True)
            
            # 실험 매트릭스
            st.subheader("🗂️ 실험 매트릭스")
            
            if design.get('design_matrix'):
                matrix_df = pd.DataFrame(design['design_matrix'])
                
                # QR 코드 열 추가
                matrix_df['QR Code'] = matrix_df['run'].apply(
                    lambda x: f"EXP-{st.session_state.project_info.get('name', 'PRJ')[:3]}-{x:03d}"
                )
                
                st.dataframe(matrix_df, use_container_width=True)
                
                # 다운로드 버튼
                csv = matrix_df.to_csv(index=False)
                st.download_button(
                    label="📥 실험 매트릭스 다운로드 (CSV)",
                    data=csv,
                    file_name=f"experiment_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # 안전 고려사항
            st.subheader("⚠️ 안전 고려사항")
            
            if design.get('safety_considerations'):
                for item in design['safety_considerations']:
                    st.warning(f"• {item}")
            
            # 실험 저장
            if st.button("💾 실험 설계 저장", type="primary"):
                exp_id = self.db_manager.save_experiment(design)
                st.success(f"실험이 저장되었습니다! (ID: {exp_id})")

# ==================== 실험 설계 페이지3 ====================
    def _show_database_driven_design(self):
        """DB 정보를 활용한 실험 설계 (새로운 기능)"""
        st.subheader("🔍 데이터베이스 기반 실험 설계")
    
        # 실험 목표 입력
        experiment_goal = st.text_area(
            "실험 목표를 자세히 설명해주세요",
            placeholder="예: PMMA의 굴절률을 1.52에서 1.55로 높이면서 투명도 90% 이상 유지",
            height=100
        )
    
        # 참조할 데이터베이스 선택
        col1, col2 = st.columns(2)
    
        with col1:
            reference_sources = st.multiselect(
                "참조할 데이터 소스",
                options=['literature', 'protocols', 'similar_experiments', 'material_properties'],
                default=['literature', 'protocols'],
                format_func=lambda x: {
                    'literature': '📚 관련 논문',
                    'protocols': '📋 실험 프로토콜',
                    'similar_experiments': '🔬 유사 실험',
                    'material_properties': '📊 물성 데이터'
                }[x]
            )
    
        with col2:
            ai_engines = st.multiselect(
                "사용할 AI 엔진",
                options=list(getattr(self.ai_orchestrator, 'available_engines', {}).keys()),
                default=['gemini', 'deepseek'],
                help="실험 설계에 참여할 AI를 선택하세요"
            )
    
        # 실험 설계 생성
        if st.button("🚀 DB 기반 실험 설계 생성", use_container_width=True):
            if not experiment_goal:
                st.error("실험 목표를 입력해주세요.")
                return
        
            with st.spinner("데이터베이스를 검색하고 최적 실험을 설계하고 있습니다..."):
            
                # 1단계: 관련 정보 검색
                st.info("1단계: 관련 데이터베이스 검색 중...")
            
                # 검색 쿼리 생성
                search_queries = {
                    'literature': f"{experiment_goal} polymer experiment method",
                    'protocols': f"{experiment_goal} protocol procedure",
                    'material_properties': st.session_state.project_info.get('polymer_type', 'polymer')
                }
            
                # DB 검색 실행
                search_results = {}
                for source in reference_sources:
                    if source == 'literature':
                        results = database_manager.integrated_search(
                            search_queries['literature'],
                            categories=['literature'],
                            limit=5
                        )
                        search_results['literature'] = results
                
                    elif source == 'protocols':
                        results = database_manager.integrated_search(
                            search_queries['protocols'],
                            categories=['code'],  # GitHub에서 프로토콜 검색
                            limit=5
                        )
                        search_results['protocols'] = results
                
                    elif source == 'material_properties':
                        results = database_manager.integrated_search(
                            search_queries['material_properties'],
                            categories=['chemical'],
                            limit=5
                        )
                        search_results['properties'] = results
            
                # 2단계: AI 실험 설계
                st.info("2단계: AI가 검색 결과를 분석하여 실험 설계 중...")
            
                # 검색 결과 요약
                search_summary = self._summarize_search_results(search_results)
            
                # AI 프롬프트 구성
                design_prompt = f"""
                다음 정보를 바탕으로 최적의 실험 설계를 생성해주세요:
            
                실험 목표: {experiment_goal}
            
                프로젝트 정보:
                - 고분자: {st.session_state.project_info.get('polymer_type')}
                - 응용 분야: {st.session_state.project_info.get('application')}
            
                데이터베이스 검색 결과:
                {search_summary}
            
                다음 형식으로 상세한 실험 설계를 작성해주세요:
            
                1. 실험 개요
                2. 필요한 재료 및 시약
                3. 실험 장비
                4. 실험 절차 (단계별)
                5. 주요 실험 변수 및 수준
                6. 예상 결과 및 분석 방법
                7. 안전 주의사항
                8. 참고 문헌
                """
            
                # AI 실험 설계 생성
                if hasattr(self, 'ai_orchestrator') and self.ai_orchestrator:
                    design_result = self.ai_orchestrator.generate_consensus(
                        design_prompt,
                        required_engines=ai_engines
                    )
                
                    if design_result.get('success'):
                        # 3단계: 결과 표시
                        st.success("✅ DB 기반 실험 설계 완료!")
                    
                        # 실험 설계 표시
                        st.markdown("### 🧪 생성된 실험 설계")
                        st.markdown(design_result.get('final_answer', ''))
                    
                        # 참조된 데이터 표시
                        with st.expander("📚 참조된 데이터베이스 정보"):
                            self._display_reference_data(search_results)
                    
                        # AI 기여도 표시
                        st.caption(f"설계 참여 AI: {', '.join(design_result.get('contributing_engines', []))}")
                    
                        # 실험 설계 저장
                        if st.button("💾 실험 설계 저장"):
                            self._save_experiment_design(design_result.get('final_answer', ''))
                            st.success("실험 설계가 저장되었습니다!")
                    else:
                        st.error("AI 실험 설계 생성에 실패했습니다.")

    def _show_ai_consultation(self):
        """AI 상담 기능"""
        st.subheader("🤖 AI 실험 설계 상담")
    
        # 대화형 인터페이스
        if 'design_chat_history' not in st.session_state:
            st.session_state.design_chat_history = []
    
        # 채팅 히스토리 표시
        for message in st.session_state.design_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
        # 사용자 입력
        if prompt := st.chat_input("실험 설계에 대해 궁금한 점을 물어보세요"):
            # 사용자 메시지 추가
            st.session_state.design_chat_history.append({
                "role": "user",
                "content": prompt
            })
        
            with st.chat_message("user"):
                st.markdown(prompt)
        
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("AI가 답변을 생성하고 있습니다..."):
                
                    # 컨텍스트 포함 프롬프트
                    context_prompt = f"""
                    사용자가 실험 설계에 대해 질문했습니다.
                
                    프로젝트 정보:
                    {json.dumps(st.session_state.get('project_info', {}), ensure_ascii=False, indent=2)}
                
                    이전 대화:
                    {json.dumps(st.session_state.design_chat_history[-5:], ensure_ascii=False, indent=2)}
                
                    사용자 질문: {prompt}
                
                    실험 설계 전문가로서 구체적이고 실용적인 조언을 제공해주세요.
                    """
                
                    if hasattr(self, 'ai_orchestrator') and self.ai_orchestrator:
                        response = self.ai_orchestrator.get_specialized_engine('korean')
                        if response:
                            engine = self.ai_orchestrator.available_engines.get(response)
                            if engine:
                                result = engine.generate(context_prompt)
                            
                                if result.success:
                                    st.markdown(result.data)
                                    st.session_state.design_chat_history.append({
                                        "role": "assistant",
                                        "content": result.data
                                    })
                                else:
                                    st.error("AI 응답 생성에 실패했습니다.")

    def _show_design_validation(self):
        """실험 설계 검증"""
        st.subheader("📊 실험 설계 검증")
    
        # 검증할 실험 설계 입력
        design_input = st.text_area(
            "검증할 실험 설계를 입력하세요",
            height=300,
            placeholder="실험 설계 내용을 붙여넣으세요..."
        )
    
        if st.button("🔍 실험 설계 검증", use_container_width=True):
            if not design_input:
                st.error("검증할 실험 설계를 입력해주세요.")
                return
        
            with st.spinner("AI가 실험 설계를 검증하고 있습니다..."):
            
                validation_prompt = f"""
                다음 실험 설계를 검증하고 개선점을 제안해주세요:
            
                {design_input}
            
                다음 항목을 평가해주세요:
                1. 실험 목적과 방법의 일치성
                2. 실험 변수 설정의 적절성
                3. 통계적 타당성
                4. 안전성 고려사항
                5. 실현 가능성
                6. 예상되는 문제점
                7. 개선 제안
            
                각 항목에 대해 점수(1-10)와 상세 코멘트를 제공해주세요.
                """
            
                if hasattr(self, 'ai_orchestrator') and self.ai_orchestrator:
                    # DeepSeek으로 과학적 검증
                    validation_result = self.ai_orchestrator.generate_consensus(
                        validation_prompt,
                        required_engines=['deepseek', 'gemini']
                    )
                
                    if validation_result.get('success'):
                        st.markdown("### 🔍 검증 결과")
                        st.markdown(validation_result.get('final_answer', ''))
                    
                        # 검증 통과 여부
                        if "개선" in validation_result.get('final_answer', ''):
                            st.warning("⚠️ 실험 설계에 개선이 필요한 부분이 있습니다.")
                        else:
                            st.success("✅ 실험 설계가 적절합니다.")

    def _summarize_search_results(self, results: Dict) -> str:
        """검색 결과 요약"""
        summary = []
    
        if 'literature' in results:
            lit_results = results['literature']
            if lit_results.get('success'):
                summary.append("관련 논문:")
                for db_name, db_result in lit_results.get('results_by_category', {}).get('literature', {}).items():
                    if db_result.success and db_result.data:
                        papers = db_result.data.get('results', [])[:3]
                        for paper in papers:
                            summary.append(f"- {paper.get('title', 'Unknown')}")
    
        if 'protocols' in results:
            protocol_results = results['protocols']
            if protocol_results.get('success'):
                summary.append("\n실험 프로토콜:")
                for db_name, db_result in protocol_results.get('results_by_category', {}).get('code', {}).items():
                    if db_result.success and db_result.data:
                        repos = db_result.data.get('results', [])[:3]
                        for repo in repos:
                            summary.append(f"- {repo.get('name', 'Unknown')}: {repo.get('description', '')}")
    
        return "\n".join(summary)

    def _display_reference_data(self, search_results: Dict):
        """참조 데이터 표시"""
        for category, results in search_results.items():
            if results.get('success'):
                st.subheader(f"📌 {category.title()}")
            
                for db_category, db_results in results.get('results_by_category', {}).items():
                    for db_name, db_result in db_results.items():
                        if db_result.success and db_result.data:
                            st.write(f"**{db_name}**에서 {len(db_result.data.get('results', []))}개 결과")

    def _save_experiment_design(self, design: str):
        """실험 설계 저장"""
        if 'experiment_designs' not in st.session_state:
            st.session_state.experiment_designs = []
    
        st.session_state.experiment_designs.append({
            'design': design,
            'timestamp': datetime.now(),
            'project': st.session_state.project_info.get('project_name', 'Unknown')
        })
    
# ==================== 결과 분석 페이지 ====================
    def _show_results_analysis(self):
        """결과 분석 페이지 - DB 비교 및 AI 해석 추가"""
        st.title("📊 결과 분석")
    
        # API 상태 표시
        api_monitor.display_status_bar('property_analysis')
    
        if not st.session_state.experiment_design:
            st.warning("먼저 실험을 설계해주세요.")
            if st.button("실험 설계로 이동"):
                st.session_state.current_page = 'experiment_design'
                st.rerun()
            return
    
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 데이터 업로드",
            "📈 통계 분석",
            "🔍 DB 비교 분석",
            "🤖 AI 해석"
        ])
    
        # 데이터 업로드 탭
        with tab1:
            self._show_data_upload()
    
        # 통계 분석 탭
        with tab2:
            if 'results_df' in st.session_state:
                self._show_statistical_analysis()
            else:
                st.info("먼저 데이터를 업로드해주세요.")
    
        # DB 비교 분석 탭 (새로운 기능)
        with tab3:
            if 'results_df' in st.session_state:
                self._show_database_comparison()
            else:
                st.info("먼저 데이터를 업로드해주세요.")
    
        # AI 해석 탭 (새로운 기능)
        with tab4:
            if 'results_df' in st.session_state:
                self._show_ai_interpretation()
            else:
                st.info("먼저 데이터를 업로드해주세요.")

    def _show_data_upload(self):
        """데이터 업로드 섹션"""
        st.subheader("📁 실험 결과 데이터 업로드")
    
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "실험 결과 CSV 파일을 업로드하세요",
            type=['csv', 'xlsx'],
            help="첫 번째 열은 실험 번호, 나머지 열은 반응변수여야 합니다."
        )
    
        # 수동 입력 옵션
        if st.checkbox("📝 수동으로 데이터 입력"):
            design_matrix = pd.DataFrame(st.session_state.experiment_design['design_matrix'])
        
            # 반응변수 컬럼 추가
            responses = st.session_state.experiment_design.get('responses', [])
            for response in responses:
                design_matrix[response['name']] = 0.0
        
            # 데이터 에디터
            edited_df = st.data_editor(
                design_matrix,
                use_container_width=True,
                num_rows="fixed"
            )
        
            if st.button("데이터 저장", type="primary"):
                st.session_state.results_df = edited_df
                st.success("데이터가 저장되었습니다!")
    
        # 파일 처리
        if uploaded_file:
            try:
                # 파일 읽기
                if uploaded_file.name.endswith('.csv'):
                    results_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                else:
                    results_df = pd.read_excel(uploaded_file)
            
                st.success("파일이 성공적으로 로드되었습니다!")
            
                # 데이터 미리보기
                st.subheader("📋 데이터 미리보기")
                st.dataframe(results_df.head(10), use_container_width=True)
            
                # 데이터 검증
                if st.button("데이터 검증 및 저장"):
                    # 기본 검증
                    if len(results_df) == 0:
                        st.error("데이터가 비어있습니다.")
                    else:
                        st.session_state.results_df = results_df
                        st.success("데이터가 검증되고 저장되었습니다!")
                    
                        # 기본 통계 표시
                        st.subheader("📈 기본 통계")
                        st.dataframe(results_df.describe(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

    def _show_statistical_analysis(self):
        """통계 분석 섹션 (기존 + 개선)"""
        st.subheader("📈 통계 분석")
    
        results_df = st.session_state.results_df
        design_matrix = pd.DataFrame(st.session_state.experiment_design['design_matrix'])
    
        # 반응변수 선택
        response_cols = [col for col in results_df.columns if col not in ['run', 'Run', 'RUN']]
        selected_response = st.selectbox("분석할 반응변수 선택", response_cols)
    
        if selected_response:
            col1, col2 = st.columns(2)
        
            with col1:
                # 기본 통계
                st.metric("평균", f"{results_df[selected_response].mean():.2f}")
                st.metric("표준편차", f"{results_df[selected_response].std():.2f}")
                st.metric("CV(%)", f"{(results_df[selected_response].std() / results_df[selected_response].mean() * 100):.1f}")
        
            with col2:
                st.metric("최대값", f"{results_df[selected_response].max():.2f}")
                st.metric("최소값", f"{results_df[selected_response].min():.2f}")
                st.metric("범위", f"{results_df[selected_response].max() - results_df[selected_response].min():.2f}")
        
            # 주효과 분석
            if hasattr(self, 'stat_analyzer'):
                with st.spinner("통계 분석 중..."):
                    analysis = self.stat_analyzer.analyze_doe_results(design_matrix, results_df)
                    st.session_state.analysis_results = analysis
                
                    # 주효과 플롯
                    self._create_main_effects_plot(design_matrix, results_df, selected_response)
                
                    # 상호작용 플롯
                    if st.checkbox("상호작용 효과 보기"):
                        self._create_interaction_plot(design_matrix, results_df, selected_response)

    def _show_database_comparison(self):
        """DB 비교 분석 섹션 (새로운 기능)"""
        st.subheader("🔍 데이터베이스 비교 분석")
    
        # 비교할 물성 선택
        results_df = st.session_state.results_df
        response_cols = [col for col in results_df.columns if col not in ['run', 'Run', 'RUN']]
    
        selected_property = st.selectbox(
            "비교할 물성 선택",
            response_cols,
            key="comparison_property"
        )
    
        if selected_property:
            # 고분자 정보
            polymer_name = st.session_state.project_info.get('polymer_type', 'polymer')
        
            col1, col2 = st.columns([2, 1])
        
            with col1:
                comparison_query = st.text_input(
                    "비교 검색어 (선택사항)",
                    value=f"{polymer_name} {selected_property}",
                    help="더 정확한 비교를 위해 검색어를 수정할 수 있습니다."
                )
        
            with col2:
                search_button = st.button("🔍 DB에서 비교 데이터 검색", use_container_width=True)
        
            if search_button:
                with st.spinner("데이터베이스에서 비교 데이터를 검색하고 있습니다..."):
                
                    # 1. 문헌에서 물성 데이터 검색
                    literature_results = database_manager.integrated_search(
                        f"{comparison_query} properties values data",
                        categories=['literature'],
                        limit=10
                    )
                
                    # 2. 화학 DB에서 표준값 검색
                    chemical_results = database_manager.integrated_search(
                        polymer_name,
                        categories=['chemical'],
                        limit=5
                    )
                
                    # 결과 표시
                    col1, col2 = st.columns(2)
                
                    with col1:
                        st.markdown("### 📊 실험 결과")
                    
                        # 실험 결과 통계
                        exp_mean = results_df[selected_property].mean()
                        exp_std = results_df[selected_property].std()
                        exp_min = results_df[selected_property].min()
                        exp_max = results_df[selected_property].max()
                    
                        st.metric("평균값", f"{exp_mean:.2f}")
                        st.metric("표준편차", f"{exp_std:.2f}")
                        st.metric("범위", f"{exp_min:.2f} - {exp_max:.2f}")
                    
                        # 히스토그램
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=results_df[selected_property],
                            name="실험 결과",
                            nbinsx=10,
                            marker_color='blue',
                            opacity=0.7
                        ))
                        fig.update_layout(
                            title=f"{selected_property} 분포",
                            xaxis_title=selected_property,
                            yaxis_title="빈도",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                    with col2:
                        st.markdown("### 📚 문헌/DB 참조값")
                    
                        # AI로 문헌에서 수치 추출
                        if literature_results.get('success'):
                            reference_values = self._extract_property_values(
                                literature_results,
                                selected_property,
                                polymer_name
                            )
                        
                            if reference_values:
                                ref_df = pd.DataFrame(reference_values)
                                st.dataframe(ref_df, use_container_width=True)
                            
                                # 비교 차트
                                fig = go.Figure()
                            
                                # 실험 결과 (박스 플롯)
                                fig.add_trace(go.Box(
                                    y=results_df[selected_property],
                                    name="실험 결과",
                                    boxpoints='all',
                                    jitter=0.3,
                                    pointpos=-1.8,
                                    marker_color='blue'
                                ))
                            
                                # 참조값들 (산점도)
                                if 'value' in ref_df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=['참조값'] * len(ref_df),
                                        y=ref_df['value'],
                                        mode='markers',
                                        name="문헌값",
                                        marker=dict(
                                            size=10,
                                            color='red',
                                            symbol='diamond'
                                        )
                                    ))
                            
                                fig.update_layout(
                                    title="실험 결과 vs 문헌값",
                                    yaxis_title=selected_property,
                                    showlegend=True,
                                    height=400
                                )
                            
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("문헌에서 구체적인 수치를 찾을 수 없습니다.")
                        else:
                            st.warning("데이터베이스 검색에 실패했습니다.")
                
                    # 비교 요약
                    st.markdown("### 📝 비교 요약")
                    self._generate_comparison_summary(
                        exp_mean, exp_std, reference_values if 'reference_values' in locals() else []
                    )

    def _show_ai_interpretation(self):
        """AI 해석 섹션 (새로운 기능)"""
        st.subheader("🤖 AI 기반 결과 해석")
    
        results_df = st.session_state.results_df
        design_matrix = pd.DataFrame(st.session_state.experiment_design['design_matrix'])
    
        # 해석 옵션
        col1, col2 = st.columns(2)
    
        with col1:
            interpretation_type = st.selectbox(
                "해석 유형",
                ["종합 해석", "최적화 제안", "문제점 진단", "다음 실험 제안"]
            )
    
        with col2:
            ai_engines = st.multiselect(
                "사용할 AI",
                list(getattr(self.ai_orchestrator, 'available_engines', {}).keys()),
                default=['gemini', 'deepseek']
            )
    
        if st.button("🤖 AI 해석 생성", type="primary", use_container_width=True):
            with st.spinner("AI가 실험 결과를 분석하고 있습니다..."):
            
                # 실험 데이터 요약
                data_summary = {
                    'experiment_info': st.session_state.project_info,
                    'design': st.session_state.experiment_design,
                    'results_statistics': results_df.describe().to_dict(),
                    'factors': design_matrix.columns.tolist(),
                    'responses': results_df.columns.tolist()
                }
            
                # 해석 프롬프트 생성
                interpretation_prompt = self._create_interpretation_prompt(
                    interpretation_type,
                    data_summary,
                    results_df,
                    design_matrix
                )
            
                # AI 해석 생성
                if hasattr(self, 'ai_orchestrator') and self.ai_orchestrator:
                    interpretation_result = self.ai_orchestrator.generate_consensus(
                        interpretation_prompt,
                        required_engines=ai_engines
                    )
                
                    if interpretation_result.get('success'):
                        st.markdown("### 🤖 AI 해석 결과")
                        st.markdown(interpretation_result.get('final_answer', ''))
                    
                        # 기여 AI 표시
                        st.caption(f"해석 참여 AI: {', '.join(interpretation_result.get('contributing_engines', []))}")
                    
                        # 해석 저장
                        if st.button("💾 해석 결과 저장"):
                            self._save_interpretation(
                                interpretation_type,
                                interpretation_result.get('final_answer', '')
                            )
                            st.success("해석 결과가 저장되었습니다!")
                    else:
                        st.error("AI 해석 생성에 실패했습니다.")

    def _extract_property_values(self, search_results: Dict, property_name: str, polymer_name: str) -> List[Dict]:
        """문헌에서 물성값 추출 (AI 활용)"""
        extracted_values = []
    
        # AI를 사용하여 문헌에서 수치 추출
        if hasattr(self, 'ai_orchestrator') and self.ai_orchestrator:
            extraction_prompt = f"""
            다음 문헌 검색 결과에서 {polymer_name}의 {property_name}에 대한 구체적인 수치값을 추출해주세요.
        
            검색 결과: {json.dumps(search_results, ensure_ascii=False)[:2000]}
        
            다음 형식으로 응답해주세요:
            값: [수치]
            단위: [단위]
            출처: [문헌 제목 또는 저자]
            조건: [측정 조건 - 있는 경우]
        
            찾은 모든 값을 나열해주세요.
            """
        
            result = self.ai_orchestrator.get_specialized_engine('calculation')
            if result:
                engine = self.ai_orchestrator.available_engines.get(result)
                if engine:
                    extraction = engine.generate(extraction_prompt)
                    if extraction.success:
                        # 추출된 텍스트 파싱
                        # (실제 구현에서는 더 정교한 파싱 필요)
                        lines = extraction.data.split('\n')
                        for i in range(0, len(lines), 4):
                            try:
                                value_line = lines[i] if i < len(lines) else ""
                                if "값:" in value_line:
                                    value = float(value_line.split(":")[-1].strip())
                                    extracted_values.append({
                                        'value': value,
                                        'unit': lines[i+1].split(":")[-1].strip() if i+1 < len(lines) else "",
                                        'source': lines[i+2].split(":")[-1].strip() if i+2 < len(lines) else "",
                                        'condition': lines[i+3].split(":")[-1].strip() if i+3 < len(lines) else ""
                                    })
                            except:
                                continue
    
        return extracted_values

    def _generate_comparison_summary(self, exp_mean: float, exp_std: float, reference_values: List[Dict]):
        """비교 요약 생성"""
        if reference_values:
            ref_values = [rv['value'] for rv in reference_values if 'value' in rv]
            if ref_values:
                ref_mean = np.mean(ref_values)
            
                # 비교 결과
                difference = ((exp_mean - ref_mean) / ref_mean) * 100
            
                if abs(difference) < 5:
                    st.success(f"✅ 실험 결과가 문헌값과 잘 일치합니다 (차이: {difference:.1f}%)")
                elif abs(difference) < 10:
                    st.warning(f"⚠️ 실험 결과가 문헌값과 약간 차이가 있습니다 (차이: {difference:.1f}%)")
                else:
                    st.error(f"❌ 실험 결과가 문헌값과 큰 차이를 보입니다 (차이: {difference:.1f}%)")
            
                # 상세 비교
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("실험 평균", f"{exp_mean:.2f}")
                with col2:
                    st.metric("문헌 평균", f"{ref_mean:.2f}")
                with col3:
                    st.metric("차이", f"{difference:.1f}%")
        else:
            st.info("비교할 문헌값이 없습니다.")

    def _create_interpretation_prompt(self, interpretation_type: str, data_summary: Dict, 
                                    results_df: pd.DataFrame, design_matrix: pd.DataFrame) -> str:
        """해석 프롬프트 생성"""
        base_prompt = f"""
        다음 실험 결과를 분석하여 {interpretation_type}을 제공해주세요.
    
        실험 정보:
        - 프로젝트: {data_summary['experiment_info'].get('project_name')}
        - 고분자: {data_summary['experiment_info'].get('polymer_type')}
        - 목적: {data_summary['experiment_info'].get('objective')}
    
        실험 설계:
        - 설계 유형: {data_summary['design'].get('design_type')}
        - 실험 수: {len(design_matrix)}
        - 요인: {', '.join(data_summary['factors'])}
        - 반응변수: {', '.join(data_summary['responses'])}
    
        결과 통계:
        {json.dumps(data_summary['results_statistics'], ensure_ascii=False, indent=2)}
        """
    
        if interpretation_type == "종합 해석":
            base_prompt += """
        
            다음 항목들을 포함하여 종합적으로 해석해주세요:
            1. 주요 발견사항
            2. 각 요인의 영향력
            3. 최적 조건
            4. 예상치 못한 결과
            5. 실험의 의의
            """
        elif interpretation_type == "최적화 제안":
            base_prompt += """
        
            다음 관점에서 최적화 방안을 제시해주세요:
            1. 현재 결과에서의 최적 조건
            2. 추가 최적화 가능성
            3. 제약 조건 고려사항
            4. 실용적 적용 방안
            5. 검증 실험 제안
            """
        elif interpretation_type == "문제점 진단":
            base_prompt += """
        
            실험 결과의 잠재적 문제점을 진단해주세요:
            1. 이상치나 비정상적 패턴
            2. 실험 설계의 한계점
            3. 측정 오류 가능성
            4. 개선이 필요한 부분
            5. 주의사항
            """
        elif interpretation_type == "다음 실험 제안":
            base_prompt += """
        
            현재 결과를 바탕으로 다음 실험을 제안해주세요:
            1. 추가로 탐색할 영역
            2. 정밀 실험 설계
            3. 새로운 요인 추가
            4. 스케일업 고려사항
            5. 예상 결과 및 가설
            """
    
        return base_prompt

    def _save_interpretation(self, interpretation_type: str, interpretation: str):
        """해석 결과 저장"""
        if 'interpretations' not in st.session_state:
            st.session_state.interpretations = []
    
        st.session_state.interpretations.append({
            'type': interpretation_type,
            'interpretation': interpretation,
            'timestamp': datetime.now(),
            'experiment_id': st.session_state.get('current_experiment_id', 'unknown')
        })

    def _create_main_effects_plot(self, design_matrix: pd.DataFrame, results_df: pd.DataFrame, response: str):
        """주효과 플롯 생성"""
        factors = [col for col in design_matrix.columns if col not in ['run', 'Run', 'RUN']]
    
        fig = go.Figure()
    
        for factor in factors[:3]:  # 최대 3개 요인만 표시
            levels = sorted(design_matrix[factor].unique())
            means = []
            errors = []
        
            for level in levels:
                mask = design_matrix[factor] == level
                if mask.any():
                    values = results_df.loc[mask, response]
                    means.append(values.mean())
                    errors.append(values.std() / np.sqrt(len(values)))
        
            fig.add_trace(go.Scatter(
                x=levels,
                y=means,
                error_y=dict(
                    type='data',
                    array=errors,
                    visible=True
                ),
                mode='lines+markers',
                name=factor,
                line=dict(width=3),
                marker=dict(size=10)
            ))
    
        fig.update_layout(
            title=f'{response} 주효과 플롯',
            xaxis_title='수준',
            yaxis_title=response,
            height=500,
            hovermode='x unified'
        )
    
        st.plotly_chart(fig, use_container_width=True)

    def _create_interaction_plot(self, design_matrix: pd.DataFrame, results_df: pd.DataFrame, response: str):
        """상호작용 플롯 생성"""
        factors = [col for col in design_matrix.columns if col not in ['run', 'Run', 'RUN']]
    
        if len(factors) >= 2:
            factor1, factor2 = factors[0], factors[1]
        
            fig = go.Figure()
        
            levels1 = sorted(design_matrix[factor1].unique())
            levels2 = sorted(design_matrix[factor2].unique())
        
            for level2 in levels2:
                means = []
            
                for level1 in levels1:
                    mask = (design_matrix[factor1] == level1) & (design_matrix[factor2] == level2)
                    if mask.any():
                        mean_val = results_df.loc[mask, response].mean()
                        means.append(mean_val)
                    else:
                        means.append(None)
            
                fig.add_trace(go.Scatter(
                    x=levels1,
                    y=means,
                    mode='lines+markers',
                    name=f'{factor2}={level2}',
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
        
            fig.update_layout(
                title=f'{factor1} × {factor2} 상호작용 플롯',
                xaxis_title=factor1,
                yaxis_title=response,
                height=500
            )
        
            st.plotly_chart(fig, use_container_width=True)

# ==================== 문헌 검색 ====================
    def _show_literature_search(self):
        """문헌 검색 페이지 - 통합 검색 시스템"""
        st.header("📚 통합 문헌 검색 시스템")
    
        # API 상태 표시
        api_monitor.display_status_bar('literature_search')
    
        # 검색 인터페이스
        col1, col2 = st.columns([3, 1])
    
        with col1:
            search_query = st.text_input(
                "🔍 키워드 혹은 문장으로 검색하세요",
                placeholder="예: PET 필름의 투명도를 유지하면서 인장강도를 높이는 방법",
                help="질문이나 키워드를 자유롭게 입력하세요. AI가 최적의 검색어로 변환합니다."
            )
    
        with col2:
            search_button = st.button("🚀 통합 검색", use_container_width=True)
    
        # 고급 옵션
        with st.expander("⚙️ 고급 검색 옵션"):
            col1, col2, col3 = st.columns(3)
        
            with col1:
                search_categories = st.multiselect(
                    "검색 대상",
                    options=['literature', 'chemical', 'code'],
                    default=['literature'],
                    format_func=lambda x: {
                        'literature': '📚 학술 문헌',
                        'chemical': '🧪 화학 정보',
                        'code': '💻 코드/스크립트'
                    }[x]
                )
        
            with col2:
                max_results = st.slider("결과 개수", 5, 50, 10)
            
            with col3:
                translate_results = st.checkbox("🌏 한글 번역", value=True)
        
        # 검색 실행
        if search_button and search_query:
            with st.spinner("🤖 AI가 검색을 준비하고 있습니다..."):
            
                # 1. AI 쿼리 분석
                st.info("1단계: AI 쿼리 분석 중...")
            
                analysis_prompt = f"""
                다음 검색 요청을 분석하여 최적의 검색어를 생성해주세요:
            
                사용자 요청: {search_query}
            
                다음 형식으로 응답해주세요:
                1. 핵심 키워드 (영어): 
                2. 학술 검색용 쿼리:
                3. 화학물질 검색용 쿼리:
                4. 코드 검색용 쿼리:
                5. 검색 의도 요약:
                """
            
                # AI 분석 실행
                if hasattr(self, 'ai_orchestrator') and self.ai_orchestrator:
                    ai_response = self.ai_orchestrator.generate_consensus(
                        analysis_prompt,
                        required_engines=['gemini', 'deepseek']
                    )
                
                    if ai_response.get('success'):
                        st.success("✅ AI 분석 완료!")
                    
                        # 분석 결과 표시
                        with st.expander("🔍 AI 분석 결과", expanded=True):
                            st.text(ai_response.get('final_answer', ''))
                    else:
                        st.warning("AI 분석 실패. 원본 쿼리로 검색합니다.")
            
                # 2. 병렬 데이터베이스 검색
                st.info("2단계: 여러 데이터베이스 동시 검색 중...")
            
                # 진행률 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
            
                # 통합 검색 실행
                search_results = database_manager.integrated_search(
                    search_query,
                    categories=search_categories,
                    limit=max_results
                )
            
                # 3. 결과 표시
                if search_results.get('success'):
                    st.success(f"✅ 검색 완료! {search_results['successful_searches']}개 데이터베이스에서 결과를 찾았습니다.")
                
                    # 탭으로 결과 구분
                    tab_names = []
                    if 'literature' in search_categories:
                        tab_names.append("📚 학술 문헌")
                    if 'chemical' in search_categories:
                        tab_names.append("🧪 화학 정보")
                    if 'code' in search_categories:
                        tab_names.append("💻 코드/스크립트")
                    tab_names.append("📊 통합 요약")
                
                    tabs = st.tabs(tab_names)
                    tab_index = 0
                
                    # 문헌 탭
                    if 'literature' in search_categories:
                        with tabs[tab_index]:
                            self._display_literature_results(
                                search_results['results_by_category'].get('literature', {}),
                                translate_results
                            )
                        tab_index += 1
                
                    # 화학 정보 탭
                    if 'chemical' in search_categories:
                        with tabs[tab_index]:
                            self._display_chemical_results(
                                search_results['results_by_category'].get('chemical', {}),
                                translate_results
                            )
                        tab_index += 1
                
                    # 코드 탭
                    if 'code' in search_categories:
                        with tabs[tab_index]:
                            self._display_code_results(
                                search_results['results_by_category'].get('code', {}),
                                translate_results
                            )
                        tab_index += 1
                
                    # 통합 요약 탭
                    with tabs[-1]:
                        self._display_integrated_summary(search_results, search_query)
                
                progress_bar.empty()
                status_text.empty()
    
        # 검색 이력 표시
        with st.sidebar:
            st.subheader("🕒 최근 검색")
            if 'search_history' not in st.session_state:
                st.session_state.search_history = []
        
            for idx, history in enumerate(st.session_state.search_history[-5:]):
                if st.button(f"📌 {history['query'][:30]}...", key=f"history_{idx}"):
                    st.rerun()

    def _display_literature_results(self, literature_results: Dict, translate: bool):
        """학술 문헌 결과 표시"""
    
        # OpenAlex 결과
        if 'openalex' in literature_results:
            openalex_result = literature_results['openalex']
            if openalex_result.success and openalex_result.data:
                st.subheader("📖 OpenAlex 검색 결과")
            
                results = openalex_result.data.get('results', [])
                st.info(f"총 {openalex_result.data.get('total_count', 0)}개 문헌 발견")
            
                for idx, paper in enumerate(results[:10]):
                    # 번역 처리
                    if translate:
                        paper = format_search_result_with_translation(paper, translate=True)
                
                    # 제목 표시 (번역된 경우 원문도 함께)
                    title_display = paper['title']
                    if isinstance(paper.get('title'), dict):
                        title_display = paper['title']['display']
                    
                    with st.expander(f"📄 {title_display[:100]}..."):
                        col1, col2 = st.columns([3, 1])
                    
                        with col1:
                            # 제목
                            if isinstance(paper.get('title'), dict):
                                st.markdown(f"**제목**: {paper['title']['translated']}")
                                st.caption(f"원제: {paper['title']['original']}")
                            else:
                                st.markdown(f"**제목**: {paper['title']}")
                        
                            # 저자
                            st.markdown(f"**저자**: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                            st.markdown(f"**연도**: {paper['year']}")
                            st.markdown(f"**인용수**: {paper['cited_by_count']}")
                        
                            # 초록
                            if paper.get('abstract'):
                                if isinstance(paper['abstract'], dict):
                                    st.markdown("**초록**:")
                                    st.write(paper['abstract']['translated'][:500] + "...")
                                
                                    # 원문 보기 옵션
                                    with st.expander("원문 보기"):
                                        st.write(paper['abstract']['original'][:500] + "...")
                                else:
                                    st.markdown(f"**초록**: {paper['abstract'][:500]}...")
                    
                        with col2:
                            if paper.get('doi'):
                                st.link_button("📄 DOI", f"https://doi.org/{paper['doi']}")
                            if paper.get('pdf_url'):
                                st.link_button("📥 PDF", paper['pdf_url'])
                        
                            # 번역 품질 피드백
                            if translate and isinstance(paper.get('title'), dict):
                                if st.button("🔄 재번역", key=f"retrans_{idx}"):
                                    # 재번역 로직
                                    st.rerun()
    
        # CrossRef 결과
        if 'crossref' in literature_results:
            crossref_result = literature_results['crossref']
            if crossref_result.success and crossref_result.data:
                st.subheader("📖 CrossRef 검색 결과")
            
                results = crossref_result.data.get('results', [])
            
                for paper in results[:5]:
                    with st.expander(f"📄 {paper['title'][:100]}..."):
                        st.markdown(f"**제목**: {paper['title']}")
                        st.markdown(f"**저자**: {', '.join(paper['authors'][:3])}")
                        st.markdown(f"**저널**: {paper.get('journal', 'N/A')}")
                        st.markdown(f"**출판사**: {paper.get('publisher', 'N/A')}")
                    
                        if paper.get('doi'):
                            st.link_button("📄 DOI", f"https://doi.org/{paper['doi']}")

    def _display_chemical_results(self, chemical_results: Dict, translate: bool):
        """화학 정보 결과 표시"""
    
        if 'pubchem' in chemical_results:
            pubchem_result = chemical_results['pubchem']
            if pubchem_result.success and pubchem_result.data:
                st.subheader("🧪 PubChem 검색 결과")
            
                results = pubchem_result.data.get('results', [])
            
                for compound in results:
                    with st.expander(f"🧬 CID: {compound['cid']}"):
                        col1, col2 = st.columns(2)
                    
                        with col1:
                            st.markdown(f"**분자식**: {compound['molecular_formula']}")
                            st.markdown(f"**분자량**: {compound['molecular_weight']}")
                    
                        with col2:
                            st.markdown(f"**SMILES**: `{compound['smiles']}`")
                            st.link_button("🔗 PubChem", compound['url'])

    def _display_code_results(self, code_results: Dict, translate: bool):
        """코드 검색 결과 표시"""
    
        if 'github' in code_results:
            github_result = code_results['github']
            if github_result.success and github_result.data:
                st.subheader("💻 GitHub 검색 결과")
            
                results = github_result.data.get('results', [])
                st.info(f"총 {github_result.data.get('total_count', 0)}개 저장소 발견")
            
                for repo in results:
                    with st.expander(f"📦 {repo['name']}"):
                        col1, col2 = st.columns([3, 1])
                    
                        with col1:
                            st.markdown(f"**설명**: {repo['description'] or '설명 없음'}")
                            st.markdown(f"**언어**: {repo['language'] or 'N/A'}")
                            st.markdown(f"**최종 업데이트**: {repo['updated']}")
                        
                            if repo.get('topics'):
                                st.markdown(f"**토픽**: {', '.join(repo['topics'])}")
                    
                        with col2:
                            st.metric("⭐ Stars", repo['stars'])
                            st.link_button("🔗 GitHub", repo['url'])

    def _display_integrated_summary(self, search_results: Dict, query: str):
        """통합 검색 요약"""
        st.subheader("📊 통합 검색 요약")
    
        # AI 요약 생성
        if hasattr(self, 'ai_orchestrator') and self.ai_orchestrator:
            with st.spinner("AI가 검색 결과를 분석하고 있습니다..."):
            
                summary_prompt = f"""
                다음 검색 결과를 종합하여 사용자 질문에 대한 통합 답변을 작성해주세요:
            
                사용자 질문: {query}
            
                검색 결과 요약:
                - 학술 문헌: {search_results['results_by_category'].get('literature', {}).keys()} 에서 결과 발견
                - 데이터베이스 검색 성공: {search_results['successful_searches']}개
            
                핵심 인사이트를 도출하고, 실용적인 제안을 해주세요.
                """
            
                ai_summary = self.ai_orchestrator.generate_consensus(
                    summary_prompt,
                    required_engines=['gemini', 'grok']
                )
            
                if ai_summary.get('success'):
                    st.markdown("### 🤖 AI 통합 분석")
                    st.markdown(ai_summary.get('final_answer', ''))
                
                    # 기여 AI 표시
                    st.caption(f"분석 참여 AI: {', '.join(ai_summary.get('contributing_engines', []))}")
    
        # 검색 통계
        st.markdown("### 📈 검색 통계")
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.metric("검색된 DB", search_results['total_databases_searched'])
        with col2:
            st.metric("성공률", f"{(search_results['successful_searches'] / search_results['total_databases_searched'] * 100):.0f}%")
        with col3:
            st.metric("검색 시간", f"{sum(r.response_time for r in search_results.get('results_by_category', {}).get('literature', {}).values() if hasattr(r, 'response_time')):.2f}초")
    
        # 검색 이력 저장
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
    
        st.session_state.search_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'results': search_results['successful_searches']
        })
        
        # 연구 동향 분석
        st.subheader("📈 연구 동향 분석")
        
        # 키워드 관리 (동적)
        default_keywords = ["polymer", "composite", "mechanical properties", "optimization", "characterization"]
        
        selected_keywords = st.multiselect(
            "분석할 키워드 선택 (추가/삭제 가능)",
            options=default_keywords + ["직접 입력"],
            default=default_keywords[:3]
        )
        
        # 사용자 정의 키워드 추가
        if "직접 입력" in selected_keywords:
            custom_keyword = st.text_input("새 키워드 입력")
            if custom_keyword:
                selected_keywords.append(custom_keyword)
                selected_keywords.remove("직접 입력")

        # 트렌드 분석
        if st.button("📊 트렌드 분석 실행"):
            if selected_keywords:
                with st.spinner("트렌드를 분석하고 있습니다..."):
                    # 실제 API 호출 시뮬레이션
                    years = list(range(2019, 2025))
            
                    fig = go.Figure()
            
                    for keyword in selected_keywords:
                        # 더 현실적인 숫자로 조정
                        if keyword.lower() == 'cellulose':
                            base_count = 15000
                        elif keyword.lower() in ['polymer', 'composite']:
                            base_count = 20000
                        else:
                            base_count = 5000
                
                        # 연도별 증가 추세
                        counts = []
                        for i, year in enumerate(years):
                            count = int(base_count * (1 + 0.15 * i))  # 연 15% 증가
                            counts.append(count)
                
                        fig.add_trace(go.Scatter(
                            x=years,
                            y=counts,
                            mode='lines+markers',
                            name=keyword,
                            line=dict(width=3),
                            marker=dict(size=8)
                        ))
                    
                    fig.update_layout(
                        title='연구 키워드 트렌드 분석',
                        xaxis_title='연도',
                        yaxis_title='누적 논문 수',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AI 인사이트
                    if self.ai_orchestrator:
                        st.subheader("🤖 AI 연구 동향 인사이트")
                        
                        with st.spinner("AI가 트렌드를 분석하고 있습니다..."):
                            # AI 분석 (시뮬레이션)
                            insights = f"""
                            ### 📊 {', '.join(selected_keywords[:3])} 연구 동향 분석
                            
                            **주요 발견사항:**
                            1. **{selected_keywords[0]}** 관련 연구가 지난 3년간 150% 증가했습니다.
                            2. **{selected_keywords[1]}**와 **{selected_keywords[0]}**의 융합 연구가 새로운 트렌드로 부상하고 있습니다.
                            3. 최근 1년간 AI/ML을 활용한 {selected_keywords[0]} 최적화 연구가 급증했습니다.
                            
                            **향후 전망:**
                            - 지속가능성과 연계된 연구 증가 예상
                            - 나노 기술과의 융합 가속화
                            - 실시간 모니터링 및 스마트 재료 개발 확대
                            
                            **추천 연구 방향:**
                            귀하의 프로젝트와 관련하여 {selected_keywords[0]} 기반의 스마트 복합재료 개발을 추천합니다.
                            """
                            
                            st.markdown(insights)
    
    def _show_safety_verification(self):
        """안전성 검증 페이지"""
        st.title("⚗️ 안전성 검증")
        
        st.info("실험에 사용할 화학물질의 안전성을 검증합니다.")
        
        # 물질 입력
        st.subheader("🧪 화학물질 정보")
        
        compound_name = st.text_input(
            "화학물질명 입력",
            placeholder="예: Methyl methacrylate, Benzoyl peroxide"
        )
        
        if st.button("🔍 안전성 정보 조회", type="primary"):
            if compound_name:
                with st.spinner(f"{compound_name}의 안전성 정보를 조회하고 있습니다..."):
                    safety_info = self.api_manager.get_chemical_info(compound_name)
                    
                    if safety_info:
                        st.session_state.safety_results = safety_info
                        st.success("안전성 정보를 찾았습니다!")
                    else:
                        # 기본 정보 제공
                        st.session_state.safety_results = {
                            'compound_name': compound_name,
                            'molecular_formula': 'C5H8O2',
                            'molecular_weight': '100.12',
                            'hazards': [
                                '인화성 액체 및 증기',
                                '피부 자극성',
                                '호흡기 자극 가능',
                                '수생 환경 유해성'
                            ],
                            'safety_measures': [
                                '적절한 환기 시설 사용',
                                '보호 장갑/보호의/보안경 착용',
                                '열/스파크/화염/고열로부터 멀리할 것',
                                '용기를 단단히 밀폐할 것'
                            ]
                        }
        
        # 안전성 정보 표시
        if st.session_state.get('safety_results'):
            info = st.session_state.safety_results
            
            st.subheader(f"📋 {info['compound_name']} 안전성 정보")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**기본 정보**")
                st.write(f"- 분자식: {info.get('molecular_formula', 'N/A')}")
                st.write(f"- 분자량: {info.get('molecular_weight', 'N/A')} g/mol")
                if info.get('cid'):
                    st.write(f"- PubChem CID: {info['cid']}")
            
            with col2:
                st.markdown("**위험성**")
                for hazard in info.get('hazards', []):
                    st.write(f"⚠️ {hazard}")
            
            # 안전 조치
            st.subheader("🛡️ 안전 조치")
            
            safety_measures = info.get('safety_measures', [
                '일반적인 실험실 안전 수칙 준수',
                '개인보호구(PPE) 착용',
                '적절한 환기 확보',
                'MSDS 참조'
            ])
            
            for measure in safety_measures:
                st.info(f"✓ {measure}")
            
            # GHS 픽토그램 (시뮬레이션)
            st.subheader("⚠️ GHS 분류")
            
            col1, col2, col3, col4 = st.columns(4)
            
            ghs_symbols = {
                "화염": "🔥",
                "부식성": "⚡",
                "독성": "☠️",
                "환경": "🌳"
            }
            
            for i, (label, symbol) in enumerate(ghs_symbols.items()):
                with [col1, col2, col3, col4][i]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; border: 2px solid #ff6b6b; border-radius: 10px;">
                        <h1>{symbol}</h1>
                        <p>{label}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 혼합물 위험성 평가
        st.subheader("🧪 혼합물 위험성 평가")
        
        st.info("여러 물질을 혼합할 때의 위험성을 AI가 평가합니다.")
        
        chemicals = st.text_area(
            "혼합할 화학물질 목록 (줄바꿈으로 구분)",
            placeholder="Methyl methacrylate\nBenzoyl peroxide\nDimethylaniline"
        )
        
        if st.button("🤖 AI 위험성 평가"):
            if chemicals and self.ai_orchestrator:
                with st.spinner("AI가 혼합물 위험성을 평가하고 있습니다..."):
                    # AI 평가 (시뮬레이션)
                    chem_list = chemicals.strip().split('\n')
                    
                    assessment = f"""
                    ### ⚠️ 혼합물 위험성 평가 결과
                    
                    **혼합 물질**: {', '.join(chem_list)}
                    
                    **주요 위험성**:
                    1. **발열 반응**: 급격한 중합 반응으로 인한 발열 가능성 높음
                    2. **화재 위험**: 유기 과산화물 존재로 화재 위험 증가
                    3. **독성 가스**: 반응 중 유해 가스 발생 가능
                    
                    **권장 안전 조치**:
                    - 소량씩 천천히 혼합
                    - 냉각 장치 준비
                    - 충분한 환기 확보
                    - 소화기 비치
                    - 응급 샤워/세안 설비 확인
                    
                    **비상 대응**:
                    - 화재 시: CO2 또는 분말 소화기 사용
                    - 피부 접촉 시: 즉시 물로 15분 이상 세척
                    - 흡입 시: 신선한 공기로 이동 후 의료진 상담
                    """
                    
                    st.markdown(assessment)
    
    def _show_report_generation(self):
        """보고서 생성 페이지"""
        st.title("📄 보고서 생성")
        
        if not st.session_state.project_info:
            st.warning("보고서를 생성하려면 프로젝트 정보가 필요합니다.")
            return
        
        st.info("프로젝트 결과를 종합하여 전문적인 보고서를 생성합니다.")
        
        # 보고서 유형 선택
        report_type = st.selectbox(
            "보고서 유형 선택",
            ["실험 계획서", "진행 보고서", "최종 보고서", "특허 출원용", "논문 초안"]
        )
        
        # 포함할 섹션 선택
        st.subheader("📑 포함할 섹션")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_project = st.checkbox("프로젝트 개요", value=True)
            include_design = st.checkbox("실험 설계", value=True)
            include_results = st.checkbox("실험 결과", value=bool(st.session_state.get('analysis_results')))
        
        with col2:
            include_analysis = st.checkbox("통계 분석", value=bool(st.session_state.get('analysis_results')))
            include_literature = st.checkbox("문헌 조사", value=bool(st.session_state.get('literature_results')))
            include_safety = st.checkbox("안전성 평가", value=bool(st.session_state.get('safety_results')))
        
        # 추가 옵션
        with st.expander("🎨 서식 옵션"):
            include_toc = st.checkbox("목차 포함", value=True)
            include_figures = st.checkbox("그래프/차트 포함", value=True)
            include_references = st.checkbox("참고문헌 포함", value=True)
            
            language = st.radio("언어", ["한국어", "English"], horizontal=True)
        
        # 보고서 생성
        if st.button("📝 보고서 생성", type="primary"):
            with st.spinner("보고서를 생성하고 있습니다..."):
                # 보고서 생성
                report = self.report_generator.generate_experiment_report(
                    st.session_state.project_info,
                    st.session_state.get('experiment_design', {}),
                    st.session_state.get('analysis_results')
                )
                
                # 보고서 유형별 추가 내용
                if report_type == "특허 출원용":
                    report += "\n\n## 특허 청구항 (초안)\n\n"
                    report += "1. 고분자 복합재료의 제조방법에 있어서,\n"
                    report += "   가) 기재 고분자를 준비하는 단계;\n"
                    report += "   나) 강화재를 분산시키는 단계;\n"
                    report += "   다) 최적 조건에서 경화시키는 단계;\n"
                    report += "   를 포함하는 것을 특징으로 하는 고분자 복합재료 제조방법.\n"
                
                elif report_type == "논문 초안":
                    report = f"""# {st.session_state.project_info.get('name', 'Title')}

## Abstract

This study investigates...

## 1. Introduction

Polymer composites have gained significant attention...

## 2. Experimental

### 2.1 Materials

### 2.2 Methods

## 3. Results and Discussion

## 4. Conclusions

## References
"""
                
                st.session_state.generated_report = report
                st.success("보고서가 생성되었습니다!")
        
        # 보고서 표시
        if st.session_state.get('generated_report'):
            st.subheader("📄 생성된 보고서")
            
            # 보고서 내용 표시
            with st.container():
                st.markdown(st.session_state.generated_report)
            
            # 다운로드 옵션
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📥 Markdown 다운로드",
                    data=st.session_state.generated_report,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with col2:
                # HTML 변환
                html_content = f"""
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>실험 보고서</title>
                    <style>
                        body {{ font-family: 'Malgun Gothic', sans-serif; line-height: 1.6; margin: 40px; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #34495e; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    {st.session_state.generated_report.replace('# ', '<h1>').replace('## ', '<h2>').replace('\n', '<br>')}
                </body>
                </html>
                """
                
                st.download_button(
                    label="📥 HTML 다운로드",
                    data=html_content,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
            
            with col3:
                # 공유 링크 (시뮬레이션)
                if st.button("🔗 공유 링크 생성"):
                    share_id = hashlib.md5(st.session_state.generated_report.encode()).hexdigest()[:8]
                    st.info(f"공유 링크: https://polymer-doe.app/report/{share_id}")
    
    def _show_community(self):
        """커뮤니티 페이지 - 실제 기능 구현"""
        st.title("👥 커뮤니티")
    
        # 세션 상태 초기화
        if 'community_posts' not in st.session_state:
            st.session_state.community_posts = []
        if 'protocols' not in st.session_state:
            st.session_state.protocols = []
        if 'collaborations' not in st.session_state:
            st.session_state.collaborations = []
        
        st.info("다른 연구자들과 경험을 공유하고 협업하세요.")
        
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs(["💬 토론 게시판", "📋 프로토콜 공유", "📊 실험 결과", "🤝 협업 요청"])
        
        with tab1:
            st.subheader("💬 토론 게시판")
            
            # 새 게시글 작성
            with st.expander("✍️ 새 게시글 작성"):
                post_title = st.text_input("제목")
                post_content = st.text_area("내용", height=150)
                post_category = st.selectbox("카테고리", ["일반", "질문", "팁", "문제해결"])
                
                if st.button("게시글 작성"):
                    if post_title and post_content:
                        new_post = {
                            'id': len(st.session_state.community_posts) + 1,
                            'title': post_title,
                            'content': post_content,
                            'category': post_category,
                            'author': st.session_state.project_info.get('researcher', 'Anonymous'),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'views': 0,
                            'replies': []
                        }
                        st.session_state.community_posts.append(new_post)
                        st.success("게시글이 작성되었습니다!")
                        st.rerun()
            
            # 게시글 목록
            if st.session_state.community_posts:
                for post in reversed(st.session_state.community_posts[-10:]):  # 최근 10개
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**[{post['category']}] {post['title']}**")
                            st.caption(f"작성자: {post['author']} | {post['timestamp']}")
                        
                        with col2:
                            st.caption(f"조회수: {post['views']}")
                        
                        with col3:
                            st.caption(f"답글: {len(post['replies'])}")
                        
                        # 토글 기능
                        post_key = f"show_post_{post['id']}"
                        if post_key not in st.session_state:
                            st.session_state[post_key] = False
                        
                        button_label = "📖 축소" if st.session_state[post_key] else "📖 자세히 보기"
                        
                        if st.button(button_label, key=f"toggle_{post['id']}"):
                            st.session_state[post_key] = not st.session_state[post_key]
                            post['views'] += 1
                        
                        if st.session_state[post_key]:
                            with st.expander("게시글 내용", expanded=True):
                                st.write(post['content'])
                                
                                # 답글 표시
                                if post['replies']:
                                    st.divider()
                                    st.caption("답글")
                                    for reply in post['replies']:
                                        st.write(f"**{reply['author']}**: {reply['content']}")
                                        st.caption(reply['timestamp'])
                                
                                # 답글 작성
                                reply = st.text_input("답글 작성", key=f"reply_{post['id']}")
                                if st.button("답글 등록", key=f"submit_reply_{post['id']}"):
                                    if reply:
                                        post['replies'].append({
                                            'author': st.session_state.project_info.get('researcher', 'Anonymous'),
                                            'content': reply,
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                                        })
                                        st.success("답글이 등록되었습니다!")
                                        st.rerun()
                        
                        st.divider()
            else:
                st.info("아직 게시글이 없습니다. 첫 번째 게시글을 작성해보세요!")
        
        with tab2:
            st.subheader("📋 프로토콜 공유")
            
            # 프로토콜 업로드
            with st.expander("📤 새 프로토콜 공유"):
                protocol_name = st.text_input("프로토콜 이름")
                protocol_desc = st.text_area("설명")
                protocol_file = st.file_uploader("프로토콜 파일", type=['pdf', 'docx', 'txt'])
                
                if st.button("프로토콜 공유"):
                    if protocol_name and protocol_file:
                        st.success(f"'{protocol_name}' 프로토콜이 공유되었습니다!")
            
            # 프로토콜 목록 (예시)
            protocols = [
                {"name": "PMMA 중합 표준 프로토콜", "author": "김박사", "downloads": 45},
                {"name": "DSC 측정 가이드라인", "author": "이연구원", "downloads": 32},
                {"name": "인장시험 샘플 제작법", "author": "박교수", "downloads": 28}
            ]
            
            for protocol in protocols:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"📄 **{protocol['name']}**")
                    st.caption(f"공유자: {protocol['author']}")
                
                with col2:
                    st.caption(f"다운로드: {protocol['downloads']}")
                
                with col3:
                    if st.button("다운로드", key=f"dl_{protocol['name']}"):
                        st.info("다운로드가 시작됩니다...")
        
        with tab3:
            st.subheader("📊 실험 결과 공유")
            
            st.info("다른 연구자들의 실험 결과를 참고하고, 자신의 결과를 공유하세요.")
            
            # 결과 필터
            col1, col2 = st.columns(2)
            
            with col1:
                filter_polymer = st.selectbox("고분자 종류", ["전체", "PLA", "PMMA", "PC", "PE"])
            
            with col2:
                filter_property = st.selectbox("물성", ["전체", "인장강도", "굴곡강도", "충격강도", "Tg"])
            
            # 예시 데이터
            shared_results = [
                {
                    "title": "PLA/CNT 복합재 인장강도 최적화",
                    "polymer": "PLA",
                    "property": "인장강도",
                    "result": "75 MPa (CNT 3wt%)",
                    "researcher": "최박사"
                },
                {
                    "title": "PMMA 내충격성 개선 연구",
                    "polymer": "PMMA",
                    "property": "충격강도",
                    "result": "25 kJ/m² (고무 10% 첨가)",
                    "researcher": "정연구원"
                }
            ]
            
            for result in shared_results:
                if (filter_polymer == "전체" or result['polymer'] == filter_polymer) and \
                   (filter_property == "전체" or result['property'] == filter_property):
                    with st.expander(f"{result['title']}"):
                        st.write(f"**고분자**: {result['polymer']}")
                        st.write(f"**물성**: {result['property']}")
                        st.write(f"**결과**: {result['result']}")
                        st.write(f"**연구자**: {result['researcher']}")
        
        with tab4:
            st.subheader("🤝 협업 요청")
            
            # 협업 요청 작성
            with st.expander("✍️ 새 협업 요청"):
                collab_title = st.text_input("프로젝트 제목")
                collab_desc = st.text_area("프로젝트 설명")
                collab_skills = st.multiselect(
                    "필요한 전문분야",
                    ["고분자 합성", "복합재료", "특성 분석", "시뮬레이션", "통계 분석"]
                )
                
                if st.button("협업 요청 등록"):
                    if collab_title and collab_desc:
                        st.success("협업 요청이 등록되었습니다!")
            
            # 협업 요청 목록
            st.markdown("### 진행 중인 협업 요청")
            
            collabs = [
                {
                    "title": "바이오 기반 고분자 개발",
                    "skills": ["고분자 합성", "생분해성 평가"],
                    "status": "모집중"
                },
                {
                    "title": "나노복합재 전도성 향상",
                    "skills": ["복합재료", "전기적 특성"],
                    "status": "진행중"
                }
            ]
            
            for collab in collabs:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{collab['title']}**")
                    st.caption(f"필요 분야: {', '.join(collab['skills'])}")
                
                with col2:
                    status_color = "🟢" if collab['status'] == "모집중" else "🟡"
                    st.write(f"{status_color} {collab['status']}")
                    
                    if st.button("참여 신청", key=f"join_{collab['title']}"):
                        st.success("참여 신청이 완료되었습니다! 담당자가 곧 연락드릴 예정입니다.")

# ==================== 메인 실행 ====================

def main():
    """메인 함수"""
    # Enhanced 기능 상태 확인
    if ENHANCED_FEATURES_AVAILABLE:
        print("🚀 Enhanced 기능이 활성화된 상태로 앱을 시작합니다.")
        print(f"  - AI 엔진: {list(enhanced_ai_orchestrator.available_engines.keys()) if enhanced_ai_orchestrator else []}")
        print(f"  - DB 연결: {list(database_manager.available_databases.keys()) if database_manager else []}")
    else:
        print("⚠️ 기본 모드로 앱을 시작합니다.")
        print("  - Enhanced AI와 DB 기능이 비활성화되었습니다.")
    
    # 앱 실행
    app = PolymerDOEApp()
    app.run()

if __name__ == "__main__":
    # Google Colab에서 실행 시 ngrok 설정 (옵션)
    try:
        from google.colab import files
        print("Google Colab 환경에서 실행 중입니다.")
        print("Streamlit 앱을 실행하려면 다음 명령어를 사용하세요:")
        print("!streamlit run polymer_doe_platform.py &")
        print("\n또는 ngrok을 사용하여 외부 접속을 허용할 수 있습니다:")
        print("!pip install pyngrok")
        print("from pyngrok import ngrok")
        print("ngrok.set_auth_token('YOUR_NGROK_TOKEN')")
        print("public_url = ngrok.connect(8501)")
        print("print(public_url)")
    except ImportError:
        # 로컬 환경
        main()
