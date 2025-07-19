import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
import logging
import gspread
from google.oauth2 import service_account
import io

# xlsxwriter 설치 확인 및 대체 방안
try:
    import xlsxwriter
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False
    st.warning("xlsxwriter가 설치되지 않았습니다. CSV 형식으로만 다운로드 가능합니다.")

# ==================== 기본 설정 ====================
# Streamlit 페이지 설정
st.set_page_config(
    page_title="🧬 고분자 실험 설계 플랫폼",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 로깅 설정 ====================
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== API 상태 타입 정의 ====================
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

# ==================== 전역 API 키 매니저 ====================
class APIKeyManager:
    """API 키를 중앙에서 관리하는 클래스"""
    
    def __init__(self):
        # 세션 상태 초기화
        if 'api_keys_initialized' not in st.session_state:
            st.session_state.api_keys_initialized = False
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
            
        # API 구성 정의
        self.api_configs = {
            # AI APIs
            'gemini': {
                'name': 'Gemini',
                'env_key': 'GEMINI_API_KEY',
                'required': False,
                'test_endpoint': None,
                'category': 'ai'
            },
            'grok': {
                'name': 'Grok',
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
            },
            'materials_commons': {
                'name': 'Materials Commons',
                'env_key': 'MC_API_KEY',
                'required': False,
                'test_endpoint': 'https://materialscommons.org/api',
                'category': 'database'
            },
            'zenodo': {
                'name': 'Zenodo',
                'env_key': 'ZENODO_API_KEY',
                'required': False,
                'test_endpoint': 'https://zenodo.org/api',
                'category': 'database'
            },
            'protocols_io': {
                'name': 'Protocols.io',
                'env_key': 'PROTOCOLS_IO_API_KEY',
                'required': False,
                'test_endpoint': 'https://www.protocols.io/api/v3',
                'category': 'database'
            },
            'figshare': {
                'name': 'Figshare',
                'env_key': 'FIGSHARE_API_KEY',
                'required': False,
                'test_endpoint': 'https://api.figshare.com/v2',
                'category': 'database'
            }
        }
        self.api_keys = {}
        self.api_status = {}
        self.load_keys()

    def load_keys(self):
        """Streamlit secrets 및 환경변수에서 API 키 로드"""
        for api_id, config in self.required_apis.items():
            key = None
            
            # 1. Streamlit secrets에서 먼저 확인
            try:
                key = st.secrets.get(config['env_key'].lower(), None)
                if not key:
                    key = st.secrets.get(api_id + '_api_key', None)
            except:
                pass
            
            # 2. 환경변수에서 확인
            if not key:
                key = os.getenv(config['env_key'])
            
            # 3. 세션 상태에서 확인 (사용자가 직접 입력한 경우)
            if not key and f'{api_id}_api_key' in st.session_state:
                key = st.session_state[f'{api_id}_api_key']
            
            if key:
                self.api_keys[api_id] = key
                self.api_status[api_id] = 'configured'
            else:
                self.api_status[api_id] = 'missing'
    
    def get_masked_key(self, api_id: str) -> str:
        """API 키의 앞 3자리만 보여주고 나머지는 마스킹"""
        if api_id not in self.api_keys:
            return "미설정"
        
        key = self.api_keys[api_id]
        if len(key) > 7:
            return f"{key[:3]}{'*' * (len(key) - 7)}{key[-4:]}"
        else:
            return "*" * len(key)
    
    def validate_key_format(self, api_id: str, key: str) -> bool:
        """API 키 형식 검증"""
        if api_id not in self.required_apis:
            return False
        
        prefix = self.required_apis[api_id]['prefix']
        if prefix and not key.startswith(prefix):
            return False
        
        # 기본 길이 체크
        if len(key) < 10:
            return False
        
        return True
    
    def save_key(self, api_id: str, key: str):
        """API 키 저장 (세션 상태)"""
        if self.validate_key_format(api_id, key):
            st.session_state[f'{api_id}_api_key'] = key
            self.api_keys[api_id] = key
            self.api_status[api_id] = 'configured'
            return True
        return False
    
    def get_status_color(self, api_id: str) -> str:
        """API 상태에 따른 색상 반환"""
        status = self.api_status.get(api_id, 'missing')
        if status == 'active':
            return "🟢"
        elif status == 'configured':
            return "🟡"
        else:
            return "🔴"
    
    def initialize_keys(self):
        """API 키 초기화"""
        # Streamlit secrets에서 먼저 확인
        if hasattr(st, 'secrets'):
            for key_id, config in self.api_configs.items():
                secret_key = config['env_key']
                if secret_key in st.secrets:
                    st.session_state.api_keys[key_id] = st.secrets[secret_key]
        
        # 환경 변수에서 확인
        for key_id, config in self.api_configs.items():
            if key_id not in st.session_state.api_keys:
                env_value = os.getenv(config['env_key'])
                if env_value:
                    st.session_state.api_keys[key_id] = env_value
        
        st.session_state.api_keys_initialized = True
    
    def get_key(self, key_id: str) -> Optional[str]:
        """API 키 반환"""
        # 세션 상태에서 확인
        if key_id in st.session_state.api_keys:
            return st.session_state.api_keys[key_id]
        
        # Streamlit secrets에서 확인
        config = self.api_configs.get(key_id)
        if config and hasattr(st, 'secrets'):
            if config['env_key'] in st.secrets:
                return st.secrets[config['env_key']]
        
        # 환경 변수에서 확인
        if config:
            return os.getenv(config['env_key'])
        
        return None
    
    def set_key(self, key_id: str, value: str):
        """API 키 설정"""
        st.session_state.api_keys[key_id] = value
        config = self.api_configs.get(key_id)
        if config:
            os.environ[config['env_key']] = value
    
    def _mask_key(self, key: str) -> str:
        """API 키를 마스킹 처리"""
        if not key:
            return ""
        if len(key) <= 8:
            return "*" * len(key)
        return key[:4] + "*" * (len(key) - 8) + key[-4:]

# 전역 API 키 매니저 인스턴스 생성
api_key_manager = APIKeyManager()

# ==================== Enhanced 모듈 임포트 시도 ====================
ENHANCED_FEATURES_AVAILABLE = False

try:
    # Enhanced 기능에 필요한 추가 라이브러리
    import google.generativeai as genai
    from groq import Groq
    import httpx
    from github import Github
    import xml.etree.ElementTree as ET
    import asyncio
    import aiohttp
    from functools import lru_cache
    import pickle
    from retrying import retry
    import langdetect
    from deep_translator import GoogleTranslator
    from urllib.parse import quote, urlencode
    from queue import Queue
    import threading
    import gspread
    from google.oauth2.service_account import Credentials
    from huggingface_hub import InferenceClient
    
    ENHANCED_FEATURES_AVAILABLE = True
    logger.info("✅ Enhanced 기능이 활성화되었습니다.")
    
except Exception as e:
    logger.warning(f"⚠️ Enhanced 기능 초기화 실패: {e}")
    logger.info("기본 모드로 실행됩니다.")

# ==================== CSS 스타일 정의 ====================
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

# ==================== StateManager 클래스 ====================
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
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

# ==================== 데이터베이스 매니저 ====================
class DatabaseManager:
    """Google Sheets 데이터베이스 관리"""
    
    def __init__(self):
        self.client = None
        self.sheet = None
        self.initialize_connection()
    
    def initialize_connection(self):
        """Google Sheets 연결 초기화"""
        try:
            # Streamlit secrets에서 서비스 계정 정보 로드
            if 'gcp_service_account' in st.secrets:
                credentials_dict = dict(st.secrets['gcp_service_account'])
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_dict,
                    scopes=['https://www.googleapis.com/auth/spreadsheets',
                            'https://www.googleapis.com/auth/drive']
                )
                self.client = gspread.authorize(credentials)
                
                # 스프레드시트 URL 확인
                if 'private_gsheets_url' in st.secrets:
                    try:
                        self.sheet = self.client.open_by_url(st.secrets['private_gsheets_url'])
                        logger.info("Google Sheets 연결 성공")
                    except Exception as e:
                        logger.error(f"스프레드시트 열기 실패: {e}")
                        st.error("스프레드시트에 접근할 수 없습니다. URL과 권한을 확인하세요.")
            else:
                logger.warning("Google Sheets 인증 정보가 없습니다.")
        except Exception as e:
            logger.error(f"Google Sheets 연결 실패: {e}")
            self.client = None
            self.sheet = None
    
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self.client is not None and self.sheet is not None
    
    def get_worksheet(self, name: str):
        """워크시트 가져오기 또는 생성"""
        if not self.is_connected():
            return None
        
        try:
            return self.sheet.worksheet(name)
        except gspread.exceptions.WorksheetNotFound:
            # 워크시트가 없으면 생성
            worksheet = self.sheet.add_worksheet(title=name, rows=1000, cols=26)
            return worksheet
    
    def save_project(self, project_data: dict) -> bool:
        """프로젝트 저장"""
        if not self.is_connected():
            return False
        
        try:
            worksheet = self.get_worksheet('projects')
            if worksheet:
                # 헤더가 없으면 추가
                if worksheet.row_count == 0 or not worksheet.row_values(1):
                    headers = list(project_data.keys())
                    worksheet.update('A1', [headers])
                
                # 데이터 추가
                values = list(project_data.values())
                worksheet.append_row(values)
                return True
        except Exception as e:
            logger.error(f"프로젝트 저장 실패: {e}")
        return False

# ==================== UI 컴포넌트 ====================
def show_api_status():
    """API 상태 표시"""
    st.markdown("### 🔌 API 연결 상태")
    
    api_manager = st.session_state.api_key_manager
    api_manager.load_keys()  # 키 다시 로드
    
    cols = st.columns(2)
    for idx, (api_id, config) in enumerate(api_manager.required_apis.items()):
        col = cols[idx % 2]
        with col:
            status_color = api_manager.get_status_color(api_id)
            masked_key = api_manager.get_masked_key(api_id)
            
            with st.expander(f"{status_color} {config['name']}"):
                st.text(f"키: {masked_key}")
                
                # API 키 입력
                new_key = st.text_input(
                    f"{config['name']} API 키",
                    type="password",
                    key=f"input_{api_id}_key",
                    placeholder=f"{config['prefix']}..."
                )
                
                if st.button(f"저장", key=f"save_{api_id}"):
                    if api_manager.save_key(api_id, new_key):
                        st.success(f"{config['name']} API 키가 저장되었습니다.")
                        st.rerun()
                    else:
                        st.error("올바른 API 키 형식이 아닙니다.")

def show_user_settings():
    """사용자 설정 표시"""
    st.markdown("### 👤 사용자 설정")
    
    # 사용자 레벨
    user_levels = {
        1: "가이드 모드",
        2: "선택 모드", 
        3: "검증 모드",
        4: "전문가 모드"
    }
    
    current_level = st.session_state.user_level
    
    # 라디오 버튼으로 레벨 선택
    selected_level = st.radio(
        "사용자 레벨",
        options=list(user_levels.keys()),
        format_func=lambda x: user_levels[x],
        index=current_level - 1,
        horizontal=True
    )
    
    if selected_level != current_level:
        st.session_state.user_level = selected_level
        st.rerun()

def create_doe_matrix(factors: dict, design_type: str = "full_factorial") -> pd.DataFrame:
    """실험 설계 매트릭스 생성"""
    if design_type == "full_factorial":
        # 완전요인설계
        import itertools
        
        factor_names = list(factors.keys())
        factor_levels = [factors[f] for f in factor_names]
        
        # 모든 조합 생성
        combinations = list(itertools.product(*factor_levels))
        
        # DataFrame 생성
        df = pd.DataFrame(combinations, columns=factor_names)
        df.insert(0, 'Run', range(1, len(df) + 1))
        
        return df
    else:
        # 다른 설계 방법 구현 가능
        return pd.DataFrame()

def show_experiment_visualization():
    """실험점 시각화"""
    if st.session_state.experiment_design is None:
        return
    
    design_df = st.session_state.experiment_design
    
    # 시각화 유형 선택 (세션 상태에 저장)
    if 'viz_type' not in st.session_state:
        st.session_state.viz_type = '2D 산점도'
    
    viz_type = st.selectbox(
        "시각화 유형",
        ['2D 산점도', '3D 산점도', '평행 좌표계', '히트맵'],
        index=['2D 산점도', '3D 산점도', '평행 좌표계', '히트맵'].index(st.session_state.viz_type)
    )
    
    # 시각화 유형이 변경되면 세션 상태 업데이트
    if viz_type != st.session_state.viz_type:
        st.session_state.viz_type = viz_type
    
    # 선택된 시각화 표시
    if viz_type == '2D 산점도':
        factors = [col for col in design_df.columns if col != 'Run']
        if len(factors) >= 2:
            fig = px.scatter(design_df, x=factors[0], y=factors[1], 
                           title='2D 실험 공간',
                           hover_data=['Run'])
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == '3D 산점도':
        factors = [col for col in design_df.columns if col != 'Run']
        if len(factors) >= 3:
            fig = px.scatter_3d(design_df, x=factors[0], y=factors[1], z=factors[2],
                              title='3D 실험 공간',
                              hover_data=['Run'])
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == '평행 좌표계':
        factors = [col for col in design_df.columns if col != 'Run']
        fig = go.Figure(data=
            go.Parcoords(
                dimensions=[dict(label=col, values=design_df[col]) for col in factors]
            )
        )
        fig.update_layout(title='평행 좌표계 시각화')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == '히트맵':
        factors = [col for col in design_df.columns if col != 'Run']
        if len(factors) >= 2:
            # 각 실험점의 밀도를 계산하여 히트맵 생성
            pivot_data = design_df.pivot_table(
                index=factors[0], 
                columns=factors[1] if len(factors) > 1 else factors[0],
                values='Run',
                aggfunc='count',
                fill_value=0
            )
            fig = px.imshow(pivot_data, title='실험점 분포 히트맵')
            st.plotly_chart(fig, use_container_width=True)

def download_csv(df: pd.DataFrame, filename: str):
    """CSV 다운로드 버튼 생성"""
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 CSV 다운로드",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )

def download_excel(df: pd.DataFrame, filename: str):
    """Excel 다운로드 버튼 생성 (xlsxwriter 없이)"""
    # BytesIO 버퍼 생성
    buffer = io.BytesIO()
    
    # pandas의 to_excel 사용 (engine 지정하지 않음)
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    buffer.seek(0)
    
    st.download_button(
        label="📥 Excel 다운로드",
        data=buffer,
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# ==================== 상태 관리 ====================
def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'user_level': 1,
        'current_page': 'home',
        'project_info': {},
        'experiment_design': None,
        'results_df': None,
        'show_visualization': False,
        'visualization_type': '2D 산점도',
        'api_key_manager': APIKeyManager(),
        'db_manager': DatabaseManager()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== Enhanced 기능들 ====================
if ENHANCED_FEATURES_AVAILABLE:
    
    # ==================== API 모니터링 시스템 ====================
    class APIMonitor:
        """API 상태를 실시간으로 모니터링하는 클래스"""
        
        def __init__(self):
            if 'api_status' not in st.session_state:
                st.session_state.api_status = {}
            if 'api_metrics' not in st.session_state:
                st.session_state.api_metrics = {}
                
            # API 그룹 정의
            self.api_groups = {
                'experiment_design': {
                    'name': '실험 설계',
                    'apis': ['gemini', 'grok', 'sambanova'],
                    'description': 'AI 기반 실험 설계 생성'
                },
                'data_analysis': {
                    'name': '데이터 분석',
                    'apis': ['deepseek', 'groq', 'huggingface'],
                    'description': '실험 결과 분석 및 최적화'
                },
                'literature_search': {
                    'name': '문헌 검색',
                    'apis': ['github', 'materials_project'],
                    'description': '관련 연구 및 데이터 검색'
                }
            }
        
        def update_status(self, api_name: str, status: APIStatus, 
                         response_time: float = 0, error_msg: str = None):
            """API 상태 업데이트"""
            st.session_state.api_status[api_name] = {
                'status': status,
                'last_check': datetime.now(),
                'response_time': response_time,
                'error': error_msg
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
                metrics['total_response_time'] += response_time
            elif error_msg:
                metrics['errors'].append({
                    'time': datetime.now(),
                    'error': error_msg
                })
        
        def get_status(self, api_name: str) -> Optional[Dict]:
            """API 상태 조회"""
            return st.session_state.api_status.get(api_name)
        
        def get_all_status(self) -> Dict:
            """모든 API 상태 조회"""
            return st.session_state.api_status
        
        def get_context_apis(self, context: str) -> List[str]:
            """컨텍스트에 필요한 API 목록"""
            return self.api_groups.get(context, {}).get('apis', [])
        
        async def check_api_health(self, api_name: str) -> APIResponse:
            """API 헬스 체크"""
            try:
                start_time = time.time()
                
                # API별 헬스 체크 로직
                api_key = api_key_manager.get_key(api_name)
                if not api_key:
                    self.update_status(api_name, APIStatus.UNAUTHORIZED, 
                                     error_msg="API key not found")
                    return APIResponse(
                        success=False,
                        data=None,
                        error="API key not found",
                        api_name=api_name
                    )
                
                # 간단한 연결 테스트
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
                        headers = self._get_auth_headers(api_name, api_key)
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
        
        def _get_auth_headers(self, api_name: str, api_key: str) -> dict:
            """API별 인증 헤더 생성"""
            if api_name in ['grok', 'sambanova', 'deepseek', 'groq']:
                return {'Authorization': f'Bearer {api_key}'}
            elif api_name == 'huggingface':
                return {'Authorization': f'Bearer {api_key}'}
            elif api_name == 'materials_project':
                return {'X-API-KEY': api_key}
            elif api_name == 'github':
                return {'Authorization': f'token {api_key}'}
            else:
                return {}
        
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
    
    # API 모니터 인스턴스 생성
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
        """Grok AI 엔진 (X.AI)"""
        
        def __init__(self):
            super().__init__("Grok 3 Mini", "grok")
            self.base_url = "https://api.x.ai/v1"
            
        async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
            try:
                start_time = time.time()
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "grok-3-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1024)
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_time = time.time() - start_time
                            api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
                            
                            return APIResponse(
                                success=True,
                                data=result['choices'][0]['message']['content'],
                                response_time=response_time,
                                api_name=self.name
                            )
                        else:
                            raise Exception(f"API error: {response.status}")
                            
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
            super().__init__("SambaNova Cloud", "sambanova")
            self.base_url = "https://api.sambanova.ai/v1"
            
        async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
            try:
                start_time = time.time()
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": kwargs.get("model", "Meta-Llama-3.1-8B-Instruct"),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "stream": False
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_time = time.time() - start_time
                            api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
                            
                            return APIResponse(
                                success=True,
                                data=result['choices'][0]['message']['content'],
                                response_time=response_time,
                                api_name=self.name
                            )
                        else:
                            raise Exception(f"API error: {response.status}")
                            
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
            super().__init__("DeepSeek Coder", "deepseek")
            self.base_url = "https://api.deepseek.com/v1"
            
        async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
            try:
                start_time = time.time()
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "deepseek-coder",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.3),
                    "max_tokens": kwargs.get("max_tokens", 2048)
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_time = time.time() - start_time
                            api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
                            
                            return APIResponse(
                                success=True,
                                data=result['choices'][0]['message']['content'],
                                response_time=response_time,
                                api_name=self.name
                            )
                        else:
                            raise Exception(f"API error: {response.status}")
                            
            except Exception as e:
                api_monitor.update_status(self.api_key_id, APIStatus.ERROR, error_msg=str(e))
                return APIResponse(
                    success=False,
                    data=None,
                    error=str(e),
                    api_name=self.name
                )
    
    class GroqEngine(BaseAIEngine):
        """Groq AI 엔진"""
        
        def __init__(self):
            super().__init__("Groq LPU", "groq")
            self.client = None
            
        def initialize(self):
            if super().initialize():
                try:
                    self.client = Groq(api_key=self.api_key)
                    return True
                except Exception as e:
                    logger.error(f"Groq initialization failed: {e}")
                    return False
            return False
        
        async def generate_async(self, prompt: str, **kwargs) -> APIResponse:
            try:
                start_time = time.time()
                
                # Groq는 초고속 응답이 특징
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=kwargs.get("model", "llama3-8b-8192"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.5),
                    max_tokens=kwargs.get("max_tokens", 1024)
                )
                
                response_time = time.time() - start_time
                api_monitor.update_status(self.api_key_id, APIStatus.ONLINE, response_time)
                
                return APIResponse(
                    success=True,
                    data=response.choices[0].message.content,
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
    
    class HuggingFaceEngine(BaseAIEngine):
        """HuggingFace AI 엔진"""
        
        def __init__(self):
            super().__init__("HuggingFace Hub", "huggingface")
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
                    max_new_tokens=kwargs.get("max_tokens", 512),
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
            tasks = [
                self.generate_single(engine, prompt, **kwargs)
                for engine in engines
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 정리
            results = {}
            for engine, response in zip(engines, responses):
                if isinstance(response, Exception):
                    results[engine] = APIResponse(
                        success=False,
                        data=None,
                        error=str(response),
                        api_name=engine
                    )
                else:
                    results[engine] = response
            
            return results
        
        async def generate_consensus(self, prompt: str, **kwargs) -> Dict:
            """다중 AI 합의 도출"""
            # 병렬로 모든 엔진 실행
            results = await self.generate_parallel(prompt, **kwargs)
            
            # 성공한 응답만 필터링
            successful_responses = {
                engine: response.data
                for engine, response in results.items()
                if response.success and response.data
            }
            
            if not successful_responses:
                return {
                    'success': False,
                    'error': 'No successful responses',
                    'responses': results
                }
            
            # 합의 도출 프롬프트
            consensus_prompt = f"""
다음은 여러 AI가 동일한 질문에 대해 제공한 답변들입니다:

원래 질문: {prompt}

AI 답변들:
"""
            for engine, response in successful_responses.items():
                consensus_prompt += f"\n[{engine}의 답변]\n{response}\n"
            
            consensus_prompt += """
위의 답변들을 종합하여:
1. 공통적으로 언급된 핵심 내용을 추출하고
2. 상충되는 부분이 있다면 가장 타당한 것을 선택하며
3. 각 AI의 장점을 살려 최적의 답변을 도출하되, 중복은 제거하고 핵심만 정리해주세요.
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
        """OpenAlex 학술 문헌 API"""
        
        def __init__(self):
            super().__init__("OpenAlex")
            self.base_url = "https://api.openalex.org"
        
        async def search_async(self, query: str, **kwargs) -> APIResponse:
            try:
                start_time = time.time()
                
                params = {
                    'search': query,
                    'per-page': kwargs.get('limit', 10),
                    'filter': 'is_oa:true'
                }
                
                # 고분자 필터 추가
                if kwargs.get('polymer_filter', True):
                    params['search'] += ' polymer'
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/works",
                        params=params,
                        headers={'User-Agent': 'PolymerDoE/1.0'}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            response_time = time.time() - start_time
                            
                            # 결과 포맷팅
                            formatted_results = []
                            for work in data.get('results', []):
                                formatted_results.append({
                                    'title': work.get('title'),
                                    'authors': [a.get('author', {}).get('display_name') 
                                              for a in work.get('authorships', [])],
                                    'year': work.get('publication_year'),
                                    'doi': work.get('doi'),
                                    'abstract': work.get('abstract'),
                                    'cited_by_count': work.get('cited_by_count', 0),
                                    'open_access': work.get('open_access', {}).get('is_oa', False)
                                })
                            
                            return APIResponse(
                                success=True,
                                data={'results': formatted_results, 'total': data.get('meta', {}).get('count', 0)},
                                response_time=response_time,
                                api_name=self.name
                            )
                        else:
                            raise Exception(f"API error: {response.status}")
                            
            except Exception as e:
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
                
                # 화합물 검색
                search_type = kwargs.get('search_type', 'name')
                
                if search_type == 'name':
                    url = f"{self.base_url}/compound/name/{quote(query)}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName/JSON"
                elif search_type == 'smiles':
                    url = f"{self.base_url}/compound/smiles/{quote(query)}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
                else:
                    url = f"{self.base_url}/compound/cid/{query}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName/JSON"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            response_time = time.time() - start_time
                            
                            # 결과 포맷팅
                            properties = data.get('PropertyTable', {}).get('Properties', [])
                            formatted_results = []
                            
                            for prop in properties:
                                formatted_results.append({
                                    'cid': prop.get('CID'),
                                    'molecular_formula': prop.get('MolecularFormula'),
                                    'molecular_weight': prop.get('MolecularWeight'),
                                    'smiles': prop.get('CanonicalSMILES'),
                                    'iupac_name': prop.get('IUPACName'),
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
                            
                            return APIResponse(
                                success=True,
                                data=data,
                                response_time=response_time,
                                api_name=self.name
                            )
                        else:
                            raise Exception(f"API error: {response.status}")
                            
            except Exception as e:
                return APIResponse(
                    success=False,
                    data=None,
                    error=str(e),
                    api_name=self.name
                )
    
    # ==================== 통합 데이터베이스 매니저 ====================
    class IntegratedDatabaseManager:
        """모든 데이터베이스 API를 통합 관리"""
        
        def __init__(self):
            self.apis = {
                'openalex': OpenAlexAPI(),
                'pubchem': PubChemAPI(),
                'github': GitHubAPI(),
                'materials_project': MaterialsProjectAPI()
            }
            
            self.available_apis = {}
            self._initialize_apis()
        
        def _initialize_apis(self):
            """사용 가능한 API 초기화"""
            for name, api in self.apis.items():
                if api.initialize():
                    self.available_apis[name] = api
                    logger.info(f"Database API initialized: {name}")
                else:
                    logger.warning(f"Database API not available: {name}")
        
        async def search_all(self, query: str, **kwargs) -> Dict:
            """모든 데이터베이스에서 통합 검색"""
            if not self.available_apis:
                return {'success': False, 'error': 'No database APIs available'}
            
            # 병렬 검색 실행
            tasks = []
            api_names = []
            
            for name, api in self.available_apis.items():
                tasks.append(api.search_async(query, **kwargs))
                api_names.append(name)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 정리
            results = {}
            for name, response in zip(api_names, responses):
                if isinstance(response, Exception):
                    results[name] = {
                        'success': False,
                        'error': str(response)
                    }
                else:
                    results[name] = {
                        'success': response.success,
                        'data': response.data,
                        'error': response.error
                    }
            
            return {
                'success': True,
                'results': results,
                'query': query
            }
        
        def search_specific(self, api_name: str, query: str, **kwargs) -> APIResponse:
            """특정 데이터베이스에서만 검색"""
            api = self.available_apis.get(api_name)
            if api:
                return api.search(query, **kwargs)
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    error=f"API {api_name} not available",
                    api_name=api_name
                )
    
    # ==================== 번역 서비스 ====================
    class TranslationService:
        """다국어 번역 서비스"""
        
        def __init__(self):
            self.translator = GoogleTranslator(source='auto', target='ko')
            self.cache = {}
        
        def translate(self, text: str, target_lang: str = 'ko', source_lang: str = 'auto') -> str:
            """텍스트 번역"""
            cache_key = f"{text}_{source_lang}_{target_lang}"
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            try:
                self.translator.source = source_lang
                self.translator.target = target_lang
                translated = self.translator.translate(text)
                self.cache[cache_key] = translated
                return translated
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return text
        
        def detect_language(self, text: str) -> str:
            """언어 감지"""
            try:
                return langdetect.detect(text)
            except:
                return 'en'
    
    # Enhanced 컴포넌트 인스턴스 생성
    enhanced_ai_orchestrator = EnhancedAIOrchestrator()
    database_manager = IntegratedDatabaseManager()
    translation_service = TranslationService()
    
else:
    # Enhanced 기능이 없을 때 더미 객체 생성
    api_monitor = None
    enhanced_ai_orchestrator = None
    database_manager = None
    translation_service = None
    AIOrchestrator = None

# ==================== 기본 기능 클래스들 ====================

class APIManager:
    """외부 API 통합 관리 (기본 버전)"""
    
    def __init__(self):
        self.pubchem_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.openalex_base = "https://api.openalex.org"
    
    def search_pubchem(self, compound_name):
        """PubChem에서 화합물 정보 검색"""
        try:
            # 화합물 이름으로 CID 검색
            search_url = f"{self.pubchem_base}/compound/name/{compound_name}/cids/JSON"
            response = requests.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                    cid = data['IdentifierList']['CID'][0]
                    
                    # CID로 상세 정보 가져오기
                    detail_url = f"{self.pubchem_base}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
                    detail_response = requests.get(detail_url, timeout=10)
                    
                    if detail_response.status_code == 200:
                        return detail_response.json()
            
            return None
        except Exception as e:
            st.error(f"PubChem 검색 오류: {str(e)}")
            return None
    
    def search_literature(self, query, limit=10):
        """OpenAlex에서 문헌 검색"""
        try:
            params = {
                'search': query,
                'per-page': limit,
                'filter': 'is_oa:true'
            }
            
            response = requests.get(
                f"{self.openalex_base}/works",
                params=params,
                headers={'User-Agent': 'PolymerDoE/1.0'},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            
            return None
        except Exception as e:
            st.error(f"문헌 검색 오류: {str(e)}")
            return None

class StatisticalAnalyzer:
    """통계 분석 도구"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def factorial_design(self, factors, levels):
        """완전요인배치법 설계"""
        import itertools
        
        # 각 요인의 수준 생성
        factor_levels = []
        for factor, level_values in zip(factors, levels):
            factor_levels.append(level_values)
        
        # 모든 조합 생성
        combinations = list(itertools.product(*factor_levels))
        
        # DataFrame으로 변환
        design = pd.DataFrame(combinations, columns=factors)
        
        # 중심점 추가 (선택사항)
        if len(factors) > 1:
            center_point = []
            for level_values in levels:
                if len(level_values) >= 2:
                    center_point.append(np.mean([level_values[0], level_values[-1]]))
                else:
                    center_point.append(level_values[0])
            
            # 중심점 3회 반복
            for _ in range(3):
                design = pd.concat([design, pd.DataFrame([center_point], columns=factors)], 
                                 ignore_index=True)
        
        return design
    
    def response_surface_design(self, factors, levels):
        """반응표면설계 (중심합성설계)"""
        n_factors = len(factors)
        
        # 기본 2^k 요인설계
        base_design = self.factorial_design(factors, [[l[0], l[-1]] for l in levels])
        
        # 축점 추가
        alpha = np.sqrt(n_factors)
        axial_points = []
        
        for i in range(n_factors):
            # +alpha 점
            point_plus = [np.mean([l[0], l[-1]]) for l in levels]
            point_plus[i] = np.mean([levels[i][0], levels[i][-1]]) + alpha * (levels[i][-1] - levels[i][0]) / 2
            axial_points.append(point_plus)
            
            # -alpha 점
            point_minus = [np.mean([l[0], l[-1]]) for l in levels]
            point_minus[i] = np.mean([levels[i][0], levels[i][-1]]) - alpha * (levels[i][-1] - levels[i][0]) / 2
            axial_points.append(point_minus)
        
        # 축점 DataFrame
        axial_df = pd.DataFrame(axial_points, columns=factors)
        
        # 중심점 추가 (5회)
        center_points = []
        center_point = [np.mean([l[0], l[-1]]) for l in levels]
        for _ in range(5):
            center_points.append(center_point)
        center_df = pd.DataFrame(center_points, columns=factors)
        
        # 전체 설계 합치기
        final_design = pd.concat([base_design, axial_df, center_df], ignore_index=True)
        
        return final_design
    
    def analyze_results(self, design_matrix, response_data):
        """실험 결과 분석"""
        try:
            # 기본 통계
            stats_summary = {
                'mean': np.mean(response_data),
                'std': np.std(response_data),
                'min': np.min(response_data),
                'max': np.max(response_data),
                'cv': (np.std(response_data) / np.mean(response_data)) * 100 if np.mean(response_data) != 0 else 0
            }
            
            # 상관관계 분석
            if design_matrix.shape[1] > 0:
                correlations = {}
                for col in design_matrix.columns:
                    corr, p_value = stats.pearsonr(design_matrix[col], response_data)
                    correlations[col] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                stats_summary['correlations'] = correlations
            
            # ANOVA 분석 (간단한 버전)
            if len(design_matrix.columns) >= 2:
                # 각 인자의 주효과 계산
                main_effects = {}
                for col in design_matrix.columns:
                    unique_levels = design_matrix[col].unique()
                    if len(unique_levels) >= 2:
                        level_means = []
                        for level in unique_levels:
                            mask = design_matrix[col] == level
                            level_means.append(np.mean(np.array(response_data)[mask]))
                        main_effects[col] = max(level_means) - min(level_means)
                stats_summary['main_effects'] = main_effects
            
            return stats_summary
            
        except Exception as e:
            st.error(f"분석 오류: {str(e)}")
            return None
    
    def optimize_response(self, design_matrix, response_data, target='maximize'):
        """반응 최적화"""
        try:
            # 2차 회귀 모델 fitting
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            # 2차 항 생성
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(design_matrix)
            
            # 모델 학습
            model = LinearRegression()
            model.fit(X_poly, response_data)
            
            # 예측값 계산
            predictions = model.predict(X_poly)
            
            # R-squared
            from sklearn.metrics import r2_score
            r2 = r2_score(response_data, predictions)
            
            # 최적점 찾기 (간단한 그리드 서치)
            if target == 'maximize':
                best_idx = np.argmax(predictions)
            else:
                best_idx = np.argmin(predictions)
            
            optimal_conditions = design_matrix.iloc[best_idx].to_dict()
            optimal_response = predictions[best_idx]
            
            return {
                'model': model,
                'r2': r2,
                'optimal_conditions': optimal_conditions,
                'predicted_response': optimal_response,
                'feature_names': poly.get_feature_names_out(design_matrix.columns.tolist())
            }
            
        except Exception as e:
            st.error(f"최적화 오류: {str(e)}")
            return None

class ReportGenerator:
    """보고서 생성기"""
    
    def __init__(self):
        pass
    
    def generate_report(self, project_info, design, results=None):
        """실험 보고서 생성"""
        report = f"""
# 고분자 실험 설계 보고서

## 1. 프로젝트 개요
- **생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **연구 목표**: {project_info.get('goal', 'N/A')}
- **대상 고분자**: {project_info.get('polymer', 'N/A')}
- **목표 물성**: {project_info.get('properties', 'N/A')}

## 2. 실험 설계
### 설계 방법: {design.get('method', 'N/A')}

### 실험 인자 및 수준
"""
        
        # 인자 정보 추가
        if 'factors' in design:
            for factor, levels in zip(design['factors'], design['levels']):
                report += f"- **{factor}**: {levels}\n"
        
        # 설계 매트릭스 추가
        if 'matrix' in design:
            report += "\n### 실험 설계 매트릭스\n"
            report += design['matrix'].to_string()
        
        # 결과 추가 (있는 경우)
        if results:
            report += "\n\n## 3. 실험 결과\n"
            report += f"- 평균: {results.get('mean', 'N/A'):.2f}\n"
            report += f"- 표준편차: {results.get('std', 'N/A'):.2f}\n"
            report += f"- 최소값: {results.get('min', 'N/A'):.2f}\n"
            report += f"- 최대값: {results.get('max', 'N/A'):.2f}\n"
            
            if 'correlations' in results:
                report += "\n### 상관관계 분석\n"
                for factor, corr_data in results['correlations'].items():
                    report += f"- {factor}: r = {corr_data['correlation']:.3f} "
                    report += f"(p = {corr_data['p_value']:.3f})\n"
            
            if 'main_effects' in results:
                report += "\n### 주효과 분석\n"
                for factor, effect in results['main_effects'].items():
                    report += f"- {factor}: {effect:.2f}\n"
        
        # 다음 단계
        report += "\n\n## 4. 다음 단계\n"
        report += design.get('next_steps', '추가 분석 및 최적화 진행')
        
        return report

# ==================== 메인 UI 클래스 ====================

class PolymerDOEApp:
    """메인 애플리케이션 클래스"""
    
    def __init__(self):
        StateManager.initialize()
        api_key_manager.initialize_keys()
        
        self.db_manager = DatabaseManager()
        
        # Enhanced 기능 통합
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                # Enhanced AI 시스템 사용
                self.ai_orchestrator = AIOrchestrator()
                
                # 새로운 컴포넌트들
                self.api_db_manager = database_manager
                self.translation_service = translation_service
                self.enhanced_features = True
                
                logger.info("✅ Enhanced AI 시스템이 연결되었습니다.")
            except Exception as e:
                logger.error(f"⚠️ Enhanced 기능 연결 실패: {e}")
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
            st.markdown("## 🧬 고분자 DoE 플랫폼")
            
            # 네비게이션
            st.markdown("### 📍 네비게이션")
            
            nav_buttons = [
                ("🏠 홈", "home"),
                ("📋 프로젝트 설정", "project_setup"),
                ("🔬 실험 설계", "experiment_design"),
                ("📊 결과 분석", "results_analysis"),
                ("📚 문헌 검색", "literature_search"),
                ("⚠️ 안전성 검증", "safety_verification"),
                ("📄 보고서 생성", "report_generation"),
                ("👥 커뮤니티", "community")
            ]
            
            for label, page in nav_buttons:
                if st.button(label, use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
            
            # 사용자 레벨
            st.markdown("### 👤 사용자 설정")
            user_level = st.select_slider(
                "사용자 레벨",
                options=[1, 2, 3, 4],
                value=st.session_state.get('user_level', 1),
                format_func=lambda x: {
                    1: "🌱 가이드 모드",
                    2: "🌿 선택 모드", 
                    3: "🌳 검증 모드",
                    4: "🎓 전문가 모드"
                }[x]
            )
            st.session_state.user_level = user_level
            
            # 플랫폼 통계
            st.markdown("### 📈 플랫폼 통계")
            stats = self.db_manager.get_platform_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("총 실험", stats.get('total_experiments', 0))
                st.metric("AI 상담", stats.get('ai_consultations', 0))
            with col2:
                st.metric("활성 사용자", stats.get('active_users', 0))
                st.metric("성공률", f"{stats.get('success_rate', 0):.1f}%")
            
            # Enhanced 기능 상태
            if self.enhanced_features:
                st.markdown("### 🚀 Enhanced 기능")
                st.success("✅ 활성화됨")
                
                # API 상태 표시
                if api_monitor:
                    with st.expander("API 상태"):
                        api_status = api_monitor.get_all_status()
                        if api_status:
                            for api_name, status in api_status.items():
                                if status['status'] == APIStatus.ONLINE:
                                    st.success(f"✅ {api_name}")
                                elif status['status'] == APIStatus.SLOW:
                                    st.warning(f"🐌 {api_name}")
                                else:
                                    st.error(f"❌ {api_name}")
                        else:
                            st.info("API 상태 확인 중...")
            
            # API 키 설정
            with st.expander("🔑 API 키 설정"):
                if st.button("API 키 관리 페이지로 이동"):
                    st.session_state.current_page = 'api_settings'
                    st.rerun()
    
    def _show_home(self):
        """홈 페이지"""
        st.markdown('<h1 class="main-header">🧬 고분자 실험 설계 플랫폼</h1>', unsafe_allow_html=True)
        
        # 환영 메시지
        st.markdown("""
        <div class="info-card">
        <h3>🎯 AI 기반 고분자 실험 설계의 새로운 패러다임</h3>
        <p>복잡한 통계 지식 없이도 전문가 수준의 실험을 설계하고, 사용하면서 자연스럽게 전문가로 성장하세요!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 주요 기능 소개
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>🤖 6개 AI 통합</h4>
            <p>Gemini, Grok, SambaNova 등 최신 AI의 합의</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>📚 통합 DB 검색</h4>
            <p>문헌, 코드, 물성 데이터를 한 번에</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>⚠️ 안전성 검증</h4>
            <p>AI 기반 위험성 사전 예측</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
            <h4>🎓 학습 시스템</h4>
            <p>4단계 레벨업 시스템</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 빠른 시작
        st.markdown("### 🚀 빠른 시작")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🆕 새 프로젝트 시작", use_container_width=True):
                st.session_state.current_page = 'project_setup'
                st.rerun()
        
        with col2:
            if st.button("📖 튜토리얼 보기", use_container_width=True):
                self._show_tutorial()
        
        with col3:
            if st.button("🔧 API 설정", use_container_width=True):
                st.session_state.current_page = 'api_settings'
                st.rerun()
        
        # 최근 업데이트
        st.markdown("### 📢 플랫폼 특징")
        
        features = {
            "✨ 다중 AI 합의 시스템": "6개 AI가 협력하여 최적의 실험 설계 도출",
            "🔍 통합 데이터베이스": "OpenAlex, PubChem, GitHub, Materials Project 동시 검색",
            "📊 고급 통계 분석": "ANOVA, 반응표면분석, 최적화 알고리즘 내장",
            "🌏 다국어 지원": "한국어 우선, 자동 번역 기능",
            "🎯 고분자 특화": "고분자 연구에 최적화된 실험 설계",
            "💾 클라우드 동기화": "Google Sheets 연동으로 어디서나 접근"
        }
        
        for feature, description in features.items():
            st.markdown(f"**{feature}**: {description}")
        
        # 성공 사례
        if st.checkbox("🏆 성공 사례 보기"):
            st.markdown("#### 사용자 성공 스토리")
            success_stories = [
                {
                    "title": "PET 필름 투명도 개선",
                    "result": "투명도 15% 향상, 기계적 강도 유지",
                    "time": "2주"
                },
                {
                    "title": "바이오 기반 고분자 개발",
                    "result": "생분해성 90% 달성, 원가 20% 절감",
                    "time": "1개월"
                }
            ]
            
            for story in success_stories:
                with st.expander(story["title"]):
                    st.write(f"**결과**: {story['result']}")
                    st.write(f"**소요 시간**: {story['time']}")
    
    def _show_tutorial(self):
        """튜토리얼 표시"""
        with st.expander("📖 플랫폼 사용 가이드", expanded=True):
            st.markdown("""
            ### 🎯 4단계 학습 시스템
            
            1. **🌱 가이드 모드 (Level 1)**
               - AI가 모든 결정을 도와드립니다
               - 단계별 상세 설명 제공
               - "왜 이렇게 하는지" 이해하기
            
            2. **🌿 선택 모드 (Level 2)**
               - AI가 2-3개 옵션 제시
               - 장단점 비교 후 선택
               - 선택 결과에 대한 피드백
            
            3. **🌳 검증 모드 (Level 3)**
               - 직접 설계 후 AI 검토
               - 개선점 제안
               - 실수를 통한 학습
            
            4. **🎓 전문가 모드 (Level 4)**
               - 완전 독립적 설계
               - AI는 요청 시에만 조언
               - 고급 기능 전체 활용
            
            ### 💡 사용 팁
            - 처음에는 가이드 모드로 시작하세요
            - 실험을 반복하며 자연스럽게 레벨업
            - 모르는 것은 AI에게 물어보세요
            """)
    
    def _show_project_setup(self):
        """프로젝트 설정 페이지"""
        st.title("📋 프로젝트 설정")
        
        # AI 상담 섹션
        if self.enhanced_features and self.ai_orchestrator:
            st.markdown("### 🤖 AI 상담")
            
            consultation_type = st.radio(
                "상담 유형 선택",
                ["빠른 설정", "상세 상담", "기존 프로젝트 개선"]
            )
            
            if st.button("💬 AI 상담 시작", use_container_width=True):
                with st.spinner("AI가 준비 중입니다..."):
                    # 플랫폼 통계 업데이트
                    self.db_manager.update_platform_stats('ai_consultations')
                    
                    if consultation_type == "빠른 설정":
                        prompt = """
고분자 실험 설계를 시작하려는 연구자입니다. 다음 정보를 수집해주세요:
1. 연구하려는 고분자 종류
2. 개선하고자 하는 주요 물성
3. 현재 직면한 문제점
4. 사용 가능한 장비/예산

각 항목에 대해 간단한 질문을 하고, 초보자도 이해할 수 있게 설명해주세요.
"""
                    elif consultation_type == "상세 상담":
                        prompt = """
고분자 실험 설계 전문가로서 상세한 프로젝트 상담을 진행해주세요. 다음을 포함하여:
1. 연구 배경과 목적
2. 기존 연구 검토
3. 예상되는 도전과제
4. 실험 설계 전략
5. 성공 지표 설정

각 단계마다 구체적인 예시와 함께 설명해주세요.
"""
                    else:
                        prompt = """
기존 고분자 실험의 개선점을 찾고 있습니다. 다음을 분석해주세요:
1. 현재 실험 방법의 문제점
2. 개선 가능한 영역
3. 새로운 접근 방법 제안
4. 예상 개선 효과

실제 사례를 들어 설명해주세요.
"""
                    
                    # AI 응답 생성
                    response = asyncio.run(
                        self.ai_orchestrator.generate_consensus(prompt)
                    )
                    
                    if response['success']:
                        st.markdown("### 💡 AI 상담 결과")
                        st.markdown(response['final_answer'])
                        
                        # 기여한 AI 표시
                        with st.expander("🤝 참여 AI 엔진"):
                            for engine in response['contributing_engines']:
                                st.write(f"- {engine}")
                    else:
                        st.error("AI 상담 생성 실패")
        
        # 프로젝트 정보 입력
        st.markdown("### 📝 프로젝트 정보")
        
        col1, col2 = st.columns(2)
        
        with col1:
            polymer = st.text_input(
                "대상 고분자",
                value=st.session_state.project_info.get('polymer', ''),
                placeholder="예: PET, PP, Nylon 6,6, PLA"
            )
            
            goal = st.text_area(
                "연구 목표",
                value=st.session_state.project_info.get('goal', ''),
                placeholder="예: 인장강도 20% 향상, 투명도 유지하면서 내열성 개선",
                height=100
            )
            
            processing_method = st.selectbox(
                "주요 가공 방법",
                options=["사출성형", "압출", "필름 캐스팅", "3D 프린팅", "용액 방사", "기타"],
                index=0
            )
        
        with col2:
            properties = st.multiselect(
                "목표 물성",
                options=[
                    "인장강도", "신장률", "충격강도", "굴곡강도",
                    "열변형온도", "유리전이온도", "용융온도", "결정화도",
                    "투명도", "색상", "광택도", "표면조도",
                    "전기전도도", "열전도도", "가스차단성", "내화학성"
                ],
                default=st.session_state.project_info.get('properties', [])
            )
            
            constraints = st.text_area(
                "제약 조건",
                value=st.session_state.project_info.get('constraints', ''),
                placeholder="예: 가공온도 250°C 이하, 식품 접촉 승인 필요, 원가 10% 이내",
                height=100
            )
            
            budget = st.select_slider(
                "예산 수준",
                options=["매우 제한적", "제한적", "보통", "충분", "매우 충분"],
                value="보통"
            )
        
        # 고급 설정
        with st.expander("🔧 고급 설정"):
            col1, col2 = st.columns(2)
            
            with col1:
                equipment = st.multiselect(
                    "사용 가능 장비",
                    options=[
                        "UTM (만능시험기)", "DSC", "TGA", "DMA",
                        "FTIR", "XRD", "SEM", "TEM",
                        "유변물성측정기", "용융지수측정기", "충격시험기"
                    ]
                )
                
                team_size = st.number_input(
                    "연구팀 규모",
                    min_value=1,
                    max_value=20,
                    value=3
                )
            
            with col2:
                timeline = st.select_slider(
                    "프로젝트 기간",
                    options=["1주", "2주", "1개월", "3개월", "6개월", "1년"],
                    value="3개월"
                )
                
                experience_level = st.radio(
                    "고분자 연구 경험",
                    options=["초보 (<1년)", "중급 (1-3년)", "숙련 (3-5년)", "전문가 (>5년)"],
                    index=1
                )
        
        # 프로젝트 정보 저장
        if st.button("💾 프로젝트 정보 저장", use_container_width=True):
            st.session_state.project_info = {
                'polymer': polymer,
                'goal': goal,
                'properties': properties,
                'constraints': constraints,
                'processing_method': processing_method,
                'budget': budget,
                'equipment': equipment,
                'team_size': team_size,
                'timeline': timeline,
                'experience_level': experience_level
            }
            
            # 데이터베이스에 저장
            project_id = self.db_manager.save_experiment(st.session_state.project_info)
            
            st.success(f"✅ 프로젝트 정보가 저장되었습니다! (ID: {project_id})")
            
            # AI 조언 생성 (Enhanced 모드)
            if self.enhanced_features and self.ai_orchestrator:
                with st.spinner("AI가 실험 설계 조언을 생성 중입니다..."):
                    advice_prompt = f"""
다음 고분자 프로젝트에 대한 실험 설계 조언을 제공해주세요:

고분자: {polymer}
목표: {goal}
목표 물성: {', '.join(properties)}
제약조건: {constraints}
가공방법: {processing_method}
예산: {budget}
기간: {timeline}

사용자 레벨: {st.session_state.user_level}/4

다음을 포함해 조언해주세요:
1. 추천 실험 인자 (3-4개)
2. 각 인자의 수준 범위
3. 적절한 실험 설계 방법
4. 예상되는 도전과제
5. 성공 가능성 평가
"""
                    
                    response = asyncio.run(
                        self.ai_orchestrator.generate_consensus(advice_prompt)
                    )
                    
                    if response['success']:
                        st.markdown("### 💡 AI 실험 설계 조언")
                        st.markdown(response['final_answer'])
    
    def _show_experiment_design(self):
        """실험 설계 페이지"""
        st.title("🔬 실험 설계")
        
        # 프로젝트 정보 확인
        if not st.session_state.project_info:
            st.warning("먼저 프로젝트 설정을 완료해주세요.")
            if st.button("프로젝트 설정으로 이동"):
                st.session_state.current_page = 'project_setup'
                st.rerun()
            return
        
        # 설계 방법 선택
        st.markdown("### 🎯 실험 설계 방법")
        
        design_method = st.selectbox(
            "설계 방법 선택",
            options=[
                "완전요인배치법 (Full Factorial Design)",
                "부분요인배치법 (Fractional Factorial Design)",
                "반응표면설계 (Response Surface Design)",
                "혼합물설계 (Mixture Design)",
                "다구치설계 (Taguchi Design)",
                "최적설계 (Optimal Design)"
            ],
            help="초보자는 완전요인배치법을 추천합니다"
        )
        
        # AI 추천 받기
        if self.enhanced_features and st.button("🤖 AI에게 설계 방법 추천받기"):
            with st.spinner("AI가 분석 중..."):
                prompt = f"""
프로젝트 정보:
- 고분자: {st.session_state.project_info.get('polymer')}
- 목표: {st.session_state.project_info.get('goal')}
- 목표 물성: {st.session_state.project_info.get('properties')}

위 프로젝트에 가장 적합한 실험 설계 방법을 추천하고 이유를 설명해주세요.
각 방법의 장단점도 비교해주세요.
"""
                response = asyncio.run(
                    self.ai_orchestrator.generate_single('gemini', prompt)
                )
                
                if response.success:
                    with st.expander("💡 AI 추천 결과", expanded=True):
                        st.markdown(response.data)
        
        # 실험 인자 설정
        st.markdown("### 🔧 실험 인자 설정")
        
        num_factors = st.number_input(
            "실험 인자 개수",
            min_value=1,
            max_value=10,
            value=3,
            help="처음에는 3-4개로 시작하는 것을 추천합니다"
        )
        
        factors = []
        levels = []
        
        for i in range(num_factors):
            st.markdown(f"#### 인자 {i+1}")
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                factor_name = st.text_input(
                    f"인자 이름",
                    key=f"factor_{i}",
                    placeholder="예: 온도, 압력, 시간"
                )
                factors.append(factor_name)
            
            with col2:
                if "반응표면" in design_method:
                    num_levels = 3
                    st.info("반응표면설계는 3수준이 필요합니다")
                else:
                    num_levels = st.number_input(
                        f"수준 개수",
                        min_value=2,
                        max_value=5,
                        value=2,
                        key=f"num_levels_{i}"
                    )
                
                level_values = []
                cols = st.columns(num_levels)
                for j, col in enumerate(cols):
                    with col:
                        value = st.number_input(
                            f"수준 {j+1}",
                            key=f"level_{i}_{j}",
                            value=0.0
                        )
                        level_values.append(value)
                levels.append(level_values)
            
            with col3:
                unit = st.text_input(
                    "단위",
                    key=f"unit_{i}",
                    placeholder="°C, MPa, min"
                )
        
        # 실험 설계 생성
        if st.button("🎲 실험 설계 생성", use_container_width=True):
            if all(factors) and all(levels):
                # 설계 매트릭스 생성
                if "완전요인" in design_method:
                    design_matrix = self.stat_analyzer.factorial_design(factors, levels)
                elif "반응표면" in design_method:
                    design_matrix = self.stat_analyzer.response_surface_design(factors, levels)
                else:
                    # 기본적으로 완전요인배치법 사용
                    design_matrix = self.stat_analyzer.factorial_design(factors, levels)
                
                # 랜덤화
                design_matrix = design_matrix.sample(frac=1).reset_index(drop=True)
                design_matrix.index = range(1, len(design_matrix) + 1)
                design_matrix.index.name = '실험번호'
                
                # 설계 정보 저장
                st.session_state.experiment_design = {
                    'method': design_method,
                    'factors': factors,
                    'levels': levels,
                    'matrix': design_matrix
                }
                
                # 플랫폼 통계 업데이트
                self.db_manager.update_platform_stats('total_experiments')
                
                # 설계 매트릭스 표시
                st.success(f"✅ 실험 설계가 생성되었습니다! (총 {len(design_matrix)}개 실험)")
                
                # 설계 요약
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 실험 수", len(design_matrix))
                with col2:
                    st.metric("실험 인자", len(factors))
                with col3:
                    st.metric("설계 방법", design_method.split('(')[0])
                
                # 설계 매트릭스 표시
                st.markdown("### 📊 실험 설계 매트릭스")
                st.dataframe(
                    design_matrix.style.highlight_max(axis=0, color='lightgreen')
                                     .highlight_min(axis=0, color='lightcoral')
                )
                
                # 시각화
                if len(factors) >= 2:
                    st.markdown("### 📈 실험점 시각화")
                    
                    viz_type = st.radio(
                        "시각화 유형",
                        ["2D 산점도", "3D 산점도", "평행 좌표계", "히트맵"]
                    )
                    
                    if viz_type == "2D 산점도":
                        x_axis = st.selectbox("X축", factors)
                        y_axis = st.selectbox("Y축", [f for f in factors if f != x_axis])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=design_matrix[x_axis],
                            y=design_matrix[y_axis],
                            mode='markers+text',
                            marker=dict(size=12, color='blue'),
                            text=[f"실험 {i}" for i in design_matrix.index],
                            textposition="top center",
                            name='실험점'
                        ))
                        fig.update_layout(
                            xaxis_title=x_axis,
                            yaxis_title=y_axis,
                            title="2D 실험 공간",
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "3D 산점도" and len(factors) >= 3:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter3d(
                            x=design_matrix[factors[0]],
                            y=design_matrix[factors[1]],
                            z=design_matrix[factors[2]],
                            mode='markers+text',
                            marker=dict(size=8, color='blue'),
                            text=[f"실험 {i}" for i in design_matrix.index],
                            name='실험점'
                        ))
                        fig.update_layout(
                            scene=dict(
                                xaxis_title=factors[0],
                                yaxis_title=factors[1],
                                zaxis_title=factors[2]
                            ),
                            title="3D 실험 공간"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "평행 좌표계":
                        # 정규화
                        normalized_df = design_matrix.copy()
                        for col in factors:
                            min_val = normalized_df[col].min()
                            max_val = normalized_df[col].max()
                            if max_val > min_val:
                                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                        
                        fig = go.Figure(data=
                            go.Parcoords(
                                line=dict(color=list(range(len(normalized_df))),
                                         colorscale='Viridis'),
                                dimensions=[
                                    dict(label=factor,
                                         values=normalized_df[factor],
                                         range=[0, 1])
                                    for factor in factors
                                ]
                            )
                        )
                        fig.update_layout(title="평행 좌표계 - 실험 설계")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # 히트맵
                        fig = px.imshow(
                            design_matrix[factors].T,
                            labels=dict(x="실험 번호", y="실험 인자", color="값"),
                            x=[f"실험 {i}" for i in design_matrix.index],
                            y=factors,
                            aspect="auto"
                        )
                        fig.update_layout(title="실험 설계 히트맵")
                        st.plotly_chart(fig, use_container_width=True)
                
                # 다운로드 옵션
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = design_matrix.to_csv()
                    st.download_button(
                        label="📥 CSV 다운로드",
                        data=csv,
                        file_name=f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Excel 다운로드
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        design_matrix.to_excel(writer, sheet_name='실험설계')
                        
                        # 프로젝트 정보도 추가
                        project_df = pd.DataFrame([st.session_state.project_info])
                        project_df.to_excel(writer, sheet_name='프로젝트정보')
                    
                    st.download_button(
                        label="📥 Excel 다운로드",
                        data=buffer.getvalue(),
                        file_name=f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                
                # QR 코드 생성 (실험 라벨용)
                if st.checkbox("🏷️ 실험 라벨 QR 코드 생성"):
                    st.markdown("### QR 코드 라벨")
                    
                    cols = st.columns(4)
                    for idx, row in design_matrix.iterrows():
                        col = cols[(idx-1) % 4]
                        with col:
                            # QR 데이터 생성
                            qr_data = {
                                'exp_no': idx,
                                'project': st.session_state.project_info.get('polymer', 'Unknown'),
                                'conditions': row.to_dict()
                            }
                            qr_text = json.dumps(qr_data)
                            
                            # QR 코드 URL 생성 (무료 API 사용)
                            qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={quote(qr_text)}"
                            
                            st.image(qr_url, caption=f"실험 {idx}")
            else:
                st.error("모든 인자와 수준을 입력해주세요.")
    
    def _show_results_analysis(self):
        """결과 분석 페이지"""
        st.title("📊 결과 분석")
        
        # 실험 설계 확인
        if not st.session_state.experiment_design:
            st.warning("먼저 실험 설계를 완료해주세요.")
            if st.button("실험 설계로 이동"):
                st.session_state.current_page = 'experiment_design'
                st.rerun()
            return
        
        design = st.session_state.experiment_design
        
        # 결과 입력 방법 선택
        st.markdown("### 📥 결과 입력")
        
        input_method = st.radio(
            "입력 방법 선택",
            options=["직접 입력", "파일 업로드", "실시간 입력"]
        )
        
        if input_method == "직접 입력":
            # 실험 결과 직접 입력
            st.markdown("### 📝 실험 결과 입력")
            
            # 반응변수 설정
            num_responses = st.number_input(
                "반응변수 개수",
                min_value=1,
                max_value=10,
                value=1
            )
            
            response_names = []
            for i in range(num_responses):
                name = st.text_input(
                    f"반응변수 {i+1} 이름",
                    key=f"response_name_{i}",
                    placeholder="예: 인장강도, 신장률"
                )
                response_names.append(name)
            
            # 결과 입력 테이블
            results_data = {}
            
            for resp_name in response_names:
                if resp_name:
                    st.markdown(f"#### {resp_name}")
                    results = []
                    
                    cols = st.columns(4)
                    for i in range(len(design['matrix'])):
                        col = cols[i % 4]
                        with col:
                            conditions = design['matrix'].iloc[i].to_dict()
                            cond_str = ", ".join([f"{k}: {v}" for k, v in conditions.items()])
                            
                            result = st.number_input(
                                f"실험 {i+1}",
                                key=f"result_{resp_name}_{i}",
                                value=0.0,
                                help=cond_str
                            )
                            results.append(result)
                    
                    results_data[resp_name] = results
        
        elif input_method == "파일 업로드":
            # 파일 업로드
            uploaded_file = st.file_uploader(
                "결과 파일 업로드",
                type=['csv', 'xlsx']
            )
            
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    results_df = pd.read_csv(uploaded_file)
                else:
                    results_df = pd.read_excel(uploaded_file)
                
                st.dataframe(results_df)
                
                # 결과 컬럼 선택
                result_columns = st.multiselect(
                    "결과 컬럼 선택",
                    options=results_df.columns.tolist()
                )
                
                results_data = {col: results_df[col].tolist() for col in result_columns}
        
        else:  # 실시간 입력
            st.info("실시간 입력 모드는 실험을 진행하면서 결과를 하나씩 입력할 수 있습니다.")
            
            # 세션 상태에 결과 저장
            if 'realtime_results' not in st.session_state:
                st.session_state.realtime_results = {}
            
            exp_no = st.number_input(
                "실험 번호",
                min_value=1,
                max_value=len(design['matrix']),
                value=1
            )
            
            # 해당 실험 조건 표시
            st.write("실험 조건:")
            st.write(design['matrix'].iloc[exp_no-1].to_dict())
            
            # 결과 입력
            response_name = st.text_input("반응변수 이름")
            response_value = st.number_input("측정값")
            
            if st.button("결과 추가"):
                if response_name not in st.session_state.realtime_results:
                    st.session_state.realtime_results[response_name] = [None] * len(design['matrix'])
                
                st.session_state.realtime_results[response_name][exp_no-1] = response_value
                st.success(f"실험 {exp_no}의 {response_name} 결과가 저장되었습니다.")
            
            results_data = st.session_state.realtime_results
        
        # 분석 실행
        if st.button("🔍 분석 실행", use_container_width=True):
            if 'results_data' in locals() and results_data:
                # 각 반응변수별 분석
                analysis_results = {}
                
                for response_name, response_values in results_data.items():
                    if response_name and all(v is not None for v in response_values):
                        # 통계 분석
                        analysis = self.stat_analyzer.analyze_results(
                            design['matrix'],
                            response_values
                        )
                        
                        # 최적화
                        optimization = self.stat_analyzer.optimize_response(
                            design['matrix'],
                            response_values,
                            target='maximize'
                        )
                        
                        analysis_results[response_name] = {
                            'basic_stats': analysis,
                            'optimization': optimization
                        }
                
                st.session_state.analysis_results = analysis_results
                
                # 분석 결과 표시
                st.success("✅ 분석이 완료되었습니다!")
                
                # 탭으로 결과 구성
                tabs = st.tabs(list(analysis_results.keys()) + ["종합 분석"])
                
                for i, (response_name, results) in enumerate(analysis_results.items()):
                    with tabs[i]:
                        self._display_analysis_results(
                            response_name,
                            results,
                            design['matrix'],
                            results_data[response_name]
                        )
                
                # 종합 분석 탭
                with tabs[-1]:
                    self._display_comprehensive_analysis(
                        analysis_results,
                        design['matrix'],
                        results_data
                    )
    
    def _display_analysis_results(self, response_name, results, design_matrix, response_values):
        """개별 반응변수 분석 결과 표시"""
        st.markdown(f"### {response_name} 분석 결과")
        
        # 기본 통계
        basic_stats = results['basic_stats']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("평균", f"{basic_stats['mean']:.2f}")
        with col2:
            st.metric("표준편차", f"{basic_stats['std']:.2f}")
        with col3:
            st.metric("최소값", f"{basic_stats['min']:.2f}")
        with col4:
            st.metric("최대값", f"{basic_stats['max']:.2f}")
        with col5:
            st.metric("CV (%)", f"{basic_stats['cv']:.1f}")
        
        # 상관관계 분석
        if 'correlations' in basic_stats:
            st.markdown("#### 🔗 인자별 상관관계")
            
            corr_data = []
            for factor, corr_info in basic_stats['correlations'].items():
                corr_data.append({
                    '인자': factor,
                    '상관계수': f"{corr_info['correlation']:.3f}",
                    'P-값': f"{corr_info['p_value']:.3f}",
                    '유의성': '✅ 유의함' if corr_info['significant'] else '❌ 유의하지 않음'
                })
            
            corr_df = pd.DataFrame(corr_data)
            st.dataframe(corr_df)
        
        # 주효과 분석
        if 'main_effects' in basic_stats:
            st.markdown("#### 📊 주효과 분석")
            
            effects = basic_stats['main_effects']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(effects.keys()),
                y=list(effects.values()),
                marker_color='lightblue'
            ))
            fig.update_layout(
                title=f"{response_name}에 대한 주효과",
                xaxis_title="실험 인자",
                yaxis_title="주효과 크기"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 주효과 플롯
        st.markdown("#### 📈 주효과 플롯")
        
        factors = design_matrix.columns.tolist()
        num_plots = len(factors)
        cols = st.columns(min(3, num_plots))
        
        for i, factor in enumerate(factors):
            col = cols[i % 3]
            with col:
                # 각 수준별 평균 계산
                levels = sorted(design_matrix[factor].unique())
                means = []
                
                for level in levels:
                    mask = design_matrix[factor] == level
                    level_mean = np.mean(np.array(response_values)[mask])
                    means.append(level_mean)
                
                # 플롯 생성
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=levels,
                    y=means,
                    mode='lines+markers',
                    marker=dict(size=10),
                    line=dict(width=2)
                ))
                fig.update_layout(
                    title=f"{factor}의 효과",
                    xaxis_title=factor,
                    yaxis_title=response_name,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 최적화 결과
        optimization = results.get('optimization')
        if optimization:
            st.markdown("#### 🎯 최적화 결과")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("모델 R²", f"{optimization['r2']:.3f}")
                st.write("**최적 조건:**")
                for factor, value in optimization['optimal_conditions'].items():
                    st.write(f"- {factor}: {value:.2f}")
            
            with col2:
                st.metric("예측 최적값", f"{optimization['predicted_response']:.2f}")
                
                # 신뢰구간 계산 (간단한 추정)
                ci = 1.96 * basic_stats['std'] / np.sqrt(len(response_values))
                st.write(f"**95% 신뢰구간**: {optimization['predicted_response']-ci:.2f} ~ {optimization['predicted_response']+ci:.2f}")
        
        # 잔차 분석
        if optimization and 'model' in optimization:
            st.markdown("#### 🔍 잔차 분석")
            
            # 예측값 계산
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(design_matrix)
            predictions = optimization['model'].predict(X_poly)
            
            # 잔차 계산
            residuals = response_values - predictions
            
            # 잔차 플롯
            col1, col2 = st.columns(2)
            
            with col1:
                # 잔차 vs 예측값
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=predictions,
                    y=residuals,
                    mode='markers',
                    marker=dict(size=8),
                    name='잔차'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title="잔차 vs 예측값",
                    xaxis_title="예측값",
                    yaxis_title="잔차"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 정규 확률 플롯
                sorted_residuals = np.sort(residuals)
                norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=norm_quantiles,
                    y=sorted_residuals,
                    mode='markers',
                    marker=dict(size=8),
                    name='잔차'
                ))
                
                # 이상적인 선
                fig.add_trace(go.Scatter(
                    x=[-3, 3],
                    y=[-3*np.std(residuals), 3*np.std(residuals)],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='정규분포'
                ))
                
                fig.update_layout(
                    title="정규 확률 플롯",
                    xaxis_title="이론적 분위수",
                    yaxis_title="잔차"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_comprehensive_analysis(self, all_results, design_matrix, all_responses):
        """종합 분석 결과 표시"""
        st.markdown("### 🎯 종합 분석")
        
        # 다목적 최적화
        if len(all_results) > 1:
            st.markdown("#### 다목적 최적화")
            
            # 각 반응의 목표 설정
            targets = {}
            weights = {}
            
            cols = st.columns(len(all_results))
            for i, response_name in enumerate(all_results.keys()):
                with cols[i]:
                    targets[response_name] = st.selectbox(
                        f"{response_name} 목표",
                        options=["최대화", "최소화", "목표값"],
                        key=f"target_{response_name}"
                    )
                    
                    if targets[response_name] == "목표값":
                        target_value = st.number_input(
                            "목표값",
                            key=f"target_value_{response_name}"
                        )
                        targets[response_name] = ('target', target_value)
                    
                    weights[response_name] = st.slider(
                        "가중치",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        key=f"weight_{response_name}"
                    )
            
            if st.button("🎯 다목적 최적화 실행"):
                # 간단한 가중 합 방법으로 최적화
                # 실제로는 더 복잡한 알고리즘 사용 가능
                
                st.info("다목적 최적화를 실행합니다...")
                
                # 각 반응의 정규화
                normalized_responses = {}
                for response_name, response_values in all_responses.items():
                    min_val = min(response_values)
                    max_val = max(response_values)
                    
                    if targets[response_name] == "최대화":
                        normalized = [(v - min_val) / (max_val - min_val) for v in response_values]
                    elif targets[response_name] == "최소화":
                        normalized = [(max_val - v) / (max_val - min_val) for v in response_values]
                    else:  # 목표값
                        target_val = targets[response_name][1]
                        normalized = [1 - abs(v - target_val) / max(abs(max_val - target_val), abs(min_val - target_val)) 
                                    for v in response_values]
                    
                    normalized_responses[response_name] = normalized
                
                # 가중 합 계산
                overall_scores = []
                for i in range(len(design_matrix)):
                    score = sum(
                        weights[name] * normalized_responses[name][i]
                        for name in all_results.keys()
                    )
                    overall_scores.append(score)
                
                # 최적 조건 찾기
                best_idx = np.argmax(overall_scores)
                optimal_conditions = design_matrix.iloc[best_idx].to_dict()
                
                st.success("✅ 다목적 최적화 완료!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**최적 조건:**")
                    for factor, value in optimal_conditions.items():
                        st.write(f"- {factor}: {value:.2f}")
                
                with col2:
                    st.write("**예상 결과:**")
                    for response_name, response_values in all_responses.items():
                        st.write(f"- {response_name}: {response_values[best_idx]:.2f}")
                
                # 전체 점수 분포
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(overall_scores)+1)),
                    y=overall_scores,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=overall_scores,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"실험 {i+1}" for i in range(len(overall_scores))]
                ))
                fig.add_trace(go.Scatter(
                    x=[best_idx + 1],
                    y=[overall_scores[best_idx]],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='star'),
                    name='최적점'
                ))
                fig.update_layout(
                    title="다목적 최적화 점수",
                    xaxis_title="실험 번호",
                    yaxis_title="종합 점수"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 상관관계 매트릭스
        if len(all_results) > 1:
            st.markdown("#### 🔗 반응변수 간 상관관계")
            
            # 상관관계 계산
            response_df = pd.DataFrame(all_responses)
            correlation_matrix = response_df.corr()
            
            # 히트맵
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False
            ))
            fig.update_layout(
                title="반응변수 간 상관관계",
                width=600,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 프로세스 능력 분석
        st.markdown("#### 📊 프로세스 능력 분석")
        
        selected_response = st.selectbox(
            "분석할 반응변수 선택",
            options=list(all_results.keys())
        )
        
        if selected_response:
            response_values = all_responses[selected_response]
            
            col1, col2 = st.columns(2)
            
            with col1:
                lsl = st.number_input("하한 규격 (LSL)", value=min(response_values))
                usl = st.number_input("상한 규격 (USL)", value=max(response_values))
            
            with col2:
                target = st.number_input("목표값", value=np.mean(response_values))
            
            if st.button("프로세스 능력 계산"):
                # Cp, Cpk 계산
                mean = np.mean(response_values)
                std = np.std(response_values, ddof=1)
                
                cp = (usl - lsl) / (6 * std)
                cpu = (usl - mean) / (3 * std)
                cpl = (mean - lsl) / (3 * std)
                cpk = min(cpu, cpl)
                
                # 결과 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Cp", f"{cp:.3f}")
                with col2:
                    st.metric("Cpk", f"{cpk:.3f}")
                with col3:
                    st.metric("Cpu", f"{cpu:.3f}")
                with col4:
                    st.metric("Cpl", f"{cpl:.3f}")
                
                # 히스토그램과 정규분포
                fig = go.Figure()
                
                # 히스토그램
                fig.add_trace(go.Histogram(
                    x=response_values,
                    name='실제 데이터',
                    nbinsx=10,
                    opacity=0.7
                ))
                
                # 정규분포 곡선
                x_range = np.linspace(min(response_values) - 2*std, max(response_values) + 2*std, 100)
                y_norm = stats.norm.pdf(x_range, mean, std) * len(response_values) * (max(response_values) - min(response_values)) / 10
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_norm,
                    mode='lines',
                    name='정규분포',
                    line=dict(color='red', width=2)
                ))
                
                # 규격선
                fig.add_vline(x=lsl, line_dash="dash", line_color="green", annotation_text="LSL")
                fig.add_vline(x=usl, line_dash="dash", line_color="green", annotation_text="USL")
                fig.add_vline(x=target, line_dash="dash", line_color="blue", annotation_text="Target")
                
                fig.update_layout(
                    title=f"{selected_response} 프로세스 능력",
                    xaxis_title=selected_response,
                    yaxis_title="빈도"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 프로세스 능력 해석
                if cpk >= 1.33:
                    st.success("✅ 프로세스 능력이 우수합니다 (Cpk ≥ 1.33)")
                elif cpk >= 1.0:
                    st.warning("⚠️ 프로세스 능력이 양호하지만 개선이 필요합니다 (1.0 ≤ Cpk < 1.33)")
                else:
                    st.error("❌ 프로세스 능력이 부족합니다 (Cpk < 1.0)")
    
    def _show_literature_search(self):
        """문헌 검색 페이지"""
        st.title("📚 문헌 검색")
        
        # 검색 설정
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "검색어 입력",
                placeholder="예: PET mechanical properties improvement, polymer nanocomposite"
            )
        
        with col2:
            search_lang = st.selectbox(
                "언어",
                options=["모든 언어", "영어", "한국어"],
                index=0
            )
        
        # 고급 검색 옵션
        with st.expander("🔍 고급 검색 옵션"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                year_from = st.number_input(
                    "출판년도 (시작)",
                    min_value=2000,
                    max_value=2024,
                    value=2020
                )
                
                result_limit = st.number_input(
                    "결과 개수",
                    min_value=5,
                    max_value=100,
                    value=20
                )
            
            with col2:
                year_to = st.number_input(
                    "출판년도 (끝)",
                    min_value=2000,
                    max_value=2024,
                    value=2024
                )
                
                sort_by = st.selectbox(
                    "정렬 기준",
                    options=["관련성", "최신순", "인용수"]
                )
            
            with col3:
                search_type = st.multiselect(
                    "검색 대상",
                    options=["학술논문", "특허", "화합물", "코드"],
                    default=["학술논문"]
                )
                
                open_access_only = st.checkbox("오픈액세스만", value=True)
        
        # 검색 실행
        if st.button("🔍 검색", use_container_width=True):
            if search_query:
                # Enhanced 모드: 통합 검색
                if self.enhanced_features and self.api_db_manager:
                    with st.spinner("통합 데이터베이스 검색 중..."):
                        # 비동기 검색을 동기적으로 실행
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        search_results = loop.run_until_complete(
                            self.api_db_manager.search_all(
                                search_query,
                                limit=result_limit,
                                polymer_filter=True
                            )
                        )
                        loop.close()
                        
                        if search_results['success']:
                            # 탭으로 결과 구성
                            available_results = {
                                k: v for k, v in search_results['results'].items()
                                if v['success'] and v.get('data')
                            }
                            
                            if available_results:
                                tabs = st.tabs(list(available_results.keys()))
                                
                                for i, (db_name, result) in enumerate(available_results.items()):
                                    with tabs[i]:
                                        self._display_search_results(db_name, result['data'])
                            else:
                                st.warning("검색 결과가 없습니다.")
                        else:
                            st.error("검색 중 오류가 발생했습니다.")
                
                # 기본 모드: 개별 검색
                else:
                    with st.spinner("검색 중..."):
                        # OpenAlex 검색
                        if "학술논문" in search_type:
                            results = self.api_manager.search_literature(
                                search_query,
                                limit=result_limit
                            )
                            
                            if results and 'results' in results:
                                self._display_search_results('openalex', {'results': results['results']})
                            else:
                                st.error("검색 결과를 찾을 수 없습니다.")
                        
                        # PubChem 검색
                        if "화합물" in search_type:
                            results = self.api_manager.search_pubchem(search_query)
                            
                            if results:
                                st.markdown("### 🧪 화합물 정보")
                                props = results.get('PropertyTable', {}).get('Properties', [])
                                if props:
                                    for prop in props:
                                        with st.expander(f"CID: {prop.get('CID', 'N/A')}"):
                                            st.write(f"**분자식**: {prop.get('MolecularFormula', 'N/A')}")
                                            st.write(f"**분자량**: {prop.get('MolecularWeight', 'N/A')}")
                                            st.write(f"**IUPAC 이름**: {prop.get('IUPACName', 'N/A')}")
                            else:
                                st.info("화합물 정보를 찾을 수 없습니다.")
            else:
                st.warning("검색어를 입력해주세요.")
        
        # AI 기반 문헌 분석
        if self.enhanced_features and st.checkbox("🤖 AI 문헌 분석 활성화"):
            st.markdown("### 🤖 AI 기반 문헌 분석")
            
            analysis_type = st.selectbox(
                "분석 유형",
                options=[
                    "연구 동향 분석",
                    "핵심 기술 추출",
                    "연구 갭 분석",
                    "메타 분석"
                ]
            )
            
            if st.button("AI 분석 실행"):
                with st.spinner("AI가 문헌을 분석 중입니다..."):
                    prompt = f"""
최근 검색된 "{search_query}"에 대한 문헌들을 바탕으로 {analysis_type}을 수행해주세요.

다음을 포함해주세요:
1. 주요 발견사항
2. 연구 트렌드
3. 향후 연구 방향
4. 실용적 시사점
"""
                    
                    response = asyncio.run(
                        self.ai_orchestrator.generate_consensus(prompt)
                    )
                    
                    if response['success']:
                        st.markdown(response['final_answer'])
    
    def _display_search_results(self, db_name, data):
        """검색 결과 표시"""
        db_names = {
            'openalex': '📚 학술 문헌',
            'pubchem': '🧪 화합물 정보',
            'github': '💻 코드 저장소',
            'materials_project': '🔬 재료 데이터'
        }
        
        st.markdown(f"### {db_names.get(db_name, db_name)}")
        
        if db_name == 'openalex':
            results = data.get('results', [])
            st.info(f"총 {len(results)}개의 문헌을 찾았습니다.")
            
            for i, paper in enumerate(results):
                with st.expander(f"📄 {paper.get('title', 'No title')}"):
                    # 저자
                    authors = paper.get('authors', [])
                    if authors:
                        author_names = [a for a in authors if a][:5]
                        if len(authors) > 5:
                            author_names.append("et al.")
                        st.write(f"**저자**: {', '.join(author_names)}")
                    
                    # 출판 정보
                    st.write(f"**발행일**: {paper.get('year', 'Unknown')}")
                    st.write(f"**인용수**: {paper.get('cited_by_count', 0)}")
                    
                    # DOI
                    if paper.get('doi'):
                        st.write(f"**DOI**: [{paper['doi']}](https://doi.org/{paper['doi']})")
                    
                    # 초록
                    if paper.get('abstract'):
                        st.write("**초록**:")
                        st.write(paper['abstract'])
                    
                    # 오픈액세스 여부
                    if paper.get('open_access'):
                        st.success("✅ 오픈액세스")
        
        elif db_name == 'github':
            results = data.get('results', [])
            st.info(f"총 {results[0].get('total_count', len(results))}개의 저장소를 찾았습니다.")
            
            for repo in results[:10]:  # 상위 10개만 표시
                with st.expander(f"💻 {repo['name']}"):
                    st.write(f"**설명**: {repo.get('description', 'No description')}")
                    st.write(f"**별**: ⭐ {repo['stars']}")
                    st.write(f"**언어**: {repo.get('language', 'Unknown')}")
                    st.write(f"**최종 업데이트**: {repo.get('updated', 'Unknown')}")
                    st.write(f"**링크**: [{repo['url']}]({repo['url']})")
                    
                    if repo.get('topics'):
                        st.write(f"**토픽**: {', '.join(repo['topics'])}")
        
        elif db_name == 'pubchem':
            results = data.get('results', [])
            for compound in results:
                with st.expander(f"🧪 CID: {compound.get('cid', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**분자식**: {compound.get('molecular_formula', 'N/A')}")
                        st.write(f"**분자량**: {compound.get('molecular_weight', 'N/A')}")
                    
                    with col2:
                        st.write(f"**SMILES**: {compound.get('smiles', 'N/A')}")
                        if compound.get('url'):
                            st.write(f"**링크**: [{compound['url']}]({compound['url']})")
    
    def _show_safety_verification(self):
        """안전성 검증 페이지"""
        st.title("⚠️ 안전성 검증")
        
        st.markdown("""
        실험을 시작하기 전에 안전성을 검증하고 위험을 예방하세요.
        AI가 화학물질의 위험성과 안전 조치를 분석해드립니다.
        """)
        
        # 검증 방법 선택
        verification_method = st.radio(
            "검증 방법",
            options=["개별 화합물 검증", "실험 조건 검증", "혼합물 호환성 검증"]
        )
        
        if verification_method == "개별 화합물 검증":
            st.markdown("### 🧪 화합물 안전성 검증")
            
            compound_name = st.text_input(
                "화합물 이름",
                placeholder="예: Benzoyl peroxide, Toluene diisocyanate"
            )
            
            if st.button("안전성 검증"):
                if compound_name:
                    with st.spinner("안전 정보 검색 중..."):
                        # PubChem에서 기본 정보 검색
                        compound_info = self.api_manager.search_pubchem(compound_name)
                        
                        if compound_info:
                            st.success("✅ 화합물 정보를 찾았습니다.")
                            
                            # 기본 정보 표시
                            props = compound_info.get('PropertyTable', {}).get('Properties', [])
                            if props:
                                prop = props[0]
                                st.write(f"**분자식**: {prop.get('MolecularFormula', 'N/A')}")
                                st.write(f"**분자량**: {prop.get('MolecularWeight', 'N/A')}")
                        
                        # AI 안전성 분석
                        if self.enhanced_features:
                            prompt = f"""
{compound_name}의 안전성 정보를 제공해주세요:

1. 주요 위험성 (물리적, 건강, 환경)
2. GHS 분류 및 위험 문구
3. 취급 시 주의사항
4. 개인보호구 (PPE) 권장사항
5. 응급조치 방법
6. 보관 조건

고분자 연구실에서 특히 주의해야 할 점도 포함해주세요.
"""
                            response = asyncio.run(
                                self.ai_orchestrator.generate_single('gemini', prompt)
                            )
                            
                            if response.success:
                                st.markdown("### 🛡️ AI 안전성 분석")
                                st.markdown(response.data)
                else:
                    st.warning("화합물 이름을 입력해주세요.")
        
        elif verification_method == "실험 조건 검증":
            st.markdown("### 🔬 실험 조건 안전성 검증")
            
            if st.session_state.experiment_design:
                design = st.session_state.experiment_design
                st.info("현재 설계된 실험 조건을 검증합니다.")
                
                # 실험 조건 표시
                st.write("**실험 인자:**")
                for factor, levels in zip(design['factors'], design['levels']):
                    st.write(f"- {factor}: {min(levels)} ~ {max(levels)}")
                
                # 추가 정보 입력
                polymer_type = st.text_input(
                    "사용할 고분자",
                    value=st.session_state.project_info.get('polymer', '')
                )
                
                additives = st.text_area(
                    "첨가제 및 용매",
                    placeholder="사용할 첨가제, 용매, 촉매 등을 입력하세요"
                )
                
                if st.button("조건 검증"):
                    with st.spinner("안전성 검증 중..."):
                        if self.enhanced_features:
                            prompt = f"""
다음 고분자 실험 조건의 안전성을 검증해주세요:

고분자: {polymer_type}
실험 조건: {design['factors']} = {design['levels']}
첨가제/용매: {additives}

다음을 평가해주세요:
1. 온도/압력 조건의 위험성
2. 화학 반응 위험성 (발열, 가스 발생 등)
3. 장비 안전성 (압력 용기, 가열 장치 등)
4. 환경 제어 필요성 (환기, 불활성 가스 등)
5. 권장 안전 조치

위험도를 낮음/중간/높음으로 평가해주세요.
"""
                            response = asyncio.run(
                                self.ai_orchestrator.generate_consensus(prompt)
                            )
                            
                            if response['success']:
                                st.markdown("### 🛡️ 실험 조건 안전성 평가")
                                st.markdown(response['final_answer'])
            else:
                st.warning("먼저 실험을 설계해주세요.")
        
        else:  # 혼합물 호환성 검증
            st.markdown("### 🧪 혼합물 호환성 검증")
            
            st.info("여러 화학물질을 함께 사용할 때의 호환성을 검증합니다.")
            
            num_compounds = st.number_input(
                "화합물 개수",
                min_value=2,
                max_value=10,
                value=2
            )
            
            compounds = []
            for i in range(num_compounds):
                compound = st.text_input(
                    f"화합물 {i+1}",
                    key=f"compat_compound_{i}"
                )
                compounds.append(compound)
            
            conditions = st.text_area(
                "혼합 조건",
                placeholder="온도, 압력, 용매, 농도 등"
            )
            
            if st.button("호환성 검증"):
                if all(compounds):
                    with st.spinner("호환성 분석 중..."):
                        if self.enhanced_features:
                            prompt = f"""
다음 화학물질들의 혼합 호환성을 분석해주세요:

화합물: {', '.join(compounds)}
조건: {conditions}

분석 내용:
1. 화학적 호환성 (반응 가능성)
2. 물리적 호환성 (상 분리, 침전 등)
3. 위험한 반응 가능성
4. 안전한 혼합 순서 및 방법
5. 특별 주의사항

각 조합에 대해 호환성을 평가하고 위험도를 표시해주세요.
"""
                            response = asyncio.run(
                                self.ai_orchestrator.generate_consensus(prompt)
                            )
                            
                            if response['success']:
                                st.markdown("### 🛡️ 혼합물 호환성 분석")
                                st.markdown(response['final_answer'])
                                
                                # 호환성 매트릭스 생성
                                st.markdown("#### 호환성 매트릭스")
                                
                                # 간단한 호환성 표시 (실제로는 AI 응답을 파싱해야 함)
                                compat_matrix = pd.DataFrame(
                                    index=compounds,
                                    columns=compounds
                                )
                                
                                for i, comp1 in enumerate(compounds):
                                    for j, comp2 in enumerate(compounds):
                                        if i == j:
                                            compat_matrix.loc[comp1, comp2] = "✓"
                                        elif i < j:
                                            compat_matrix.loc[comp1, comp2] = "?"
                                        else:
                                            compat_matrix.loc[comp1, comp2] = compat_matrix.loc[comp2, comp1]
                                
                                st.dataframe(compat_matrix)
                else:
                    st.warning("모든 화합물을 입력해주세요.")
        
        # 안전 체크리스트
        st.markdown("### 📋 안전 체크리스트")
        
        with st.expander("실험 전 안전 체크리스트", expanded=True):
            safety_items = [
                "MSDS 확인 완료",
                "개인보호구(PPE) 착용",
                "환기 시스템 작동 확인",
                "비상 샤워/세안기 위치 확인",
                "소화기 위치 확인",
                "폐기물 처리 방법 확인",
                "비상 연락처 확인",
                "실험 절차 숙지"
            ]
            
            for item in safety_items:
                st.checkbox(item)
        
        # 비상 대응 정보
        st.markdown("### 🚨 비상 대응")
        
        emergency_tabs = st.tabs(["화재", "화학물질 노출", "유출", "부상"])
        
        with emergency_tabs[0]:
            st.markdown("""
            **화재 발생 시:**
            1. 즉시 대피 - 생명이 최우선
            2. 화재경보기 작동
            3. 119 신고
            4. 가능한 경우 전원 차단
            5. 소화기 사용 (PASS 방법)
               - Pull: 안전핀 제거
               - Aim: 화염 아래쪽 조준
               - Squeeze: 손잡이 압착
               - Sweep: 좌우로 분사
            """)
        
        with emergency_tabs[1]:
            st.markdown("""
            **화학물질 노출 시:**
            1. 오염된 의복 즉시 제거
            2. 다량의 물로 15분 이상 세척
            3. 눈 노출: 흐르는 물로 15분 이상 세안
            4. 흡입: 신선한 공기가 있는 곳으로 이동
            5. 섭취: 구토 유도 금지, 즉시 병원
            6. MSDS 지참하여 의료진에게 제공
            """)
        
        with emergency_tabs[2]:
            st.markdown("""
            **화학물질 유출 시:**
            1. 주변 인원 대피 및 출입 통제
            2. 개인보호구 착용
            3. 점화원 제거 (가연성 물질의 경우)
            4. 유출 확산 방지 (흡착재 사용)
            5. 환기 강화
            6. 적절한 방법으로 청소 및 폐기
            """)
        
        with emergency_tabs[3]:
            st.markdown("""
            **부상 발생 시:**
            1. 부상자 안전한 곳으로 이동
            2. 의식 및 호흡 확인
            3. 출혈: 직접 압박으로 지혈
            4. 골절: 부목 고정, 움직이지 않기
            5. 화상: 찬물로 냉각 (최소 10분)
            6. 119 신고 또는 병원 이송
            """)
    
    def _show_report_generation(self):
        """보고서 생성 페이지"""
        st.title("📄 보고서 생성")
        
        # 보고서 유형 선택
        report_type = st.selectbox(
            "보고서 유형",
            options=[
                "실험 설계 보고서",
                "결과 분석 보고서",
                "종합 연구 보고서",
                "프레젠테이션 자료"
            ]
        )
        
        # 보고서에 포함할 내용 선택
        st.markdown("### 📋 보고서 구성")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_project = st.checkbox(
                "프로젝트 정보",
                value=bool(st.session_state.project_info)
            )
            
            include_design = st.checkbox(
                "실험 설계",
                value=bool(st.session_state.experiment_design)
            )
            
            include_results = st.checkbox(
                "실험 결과 및 분석",
                value=bool(st.session_state.analysis_results)
            )
        
        with col2:
            include_literature = st.checkbox(
                "참고 문헌",
                value=bool(st.session_state.literature_results)
            )
            
            include_safety = st.checkbox(
                "안전성 평가",
                value=bool(st.session_state.safety_results)
            )
            
            include_recommendations = st.checkbox(
                "결론 및 제언",
                value=True
            )
        
        # 보고서 스타일 설정
        with st.expander("📝 보고서 스타일 설정"):
            col1, col2 = st.columns(2)
            
            with col1:
                language = st.selectbox(
                    "언어",
                    options=["한국어", "영어", "한국어/영어 병기"]
                )
                
                format_style = st.selectbox(
                    "형식",
                    options=["학술 논문", "기술 보고서", "사업 보고서"]
                )
            
            with col2:
                detail_level = st.select_slider(
                    "상세 수준",
                    options=["요약", "표준", "상세"],
                    value="표준"
                )
                
                include_visuals = st.checkbox("그래프 및 도표 포함", value=True)
        
        # 보고서 생성
        if st.button("📄 보고서 생성", use_container_width=True):
            with st.spinner("보고서를 생성하고 있습니다..."):
                # 기본 보고서 생성
                report_content = self.report_generator.generate_report(
                    st.session_state.project_info if include_project else {},
                    st.session_state.experiment_design if include_design else {},
                    st.session_state.analysis_results if include_results else None
                )
                
                # AI 향상 (Enhanced 모드)
                if self.enhanced_features and st.checkbox("AI 보고서 향상"):
                    enhancement_prompt = f"""
다음 기본 보고서를 {format_style} 형식의 {language} {detail_level} 수준 보고서로 향상시켜주세요:

{report_content}

다음을 포함해주세요:
1. 전문적인 서론
2. 명확한 섹션 구분
3. 과학적 해석과 고찰
4. 실용적 시사점
5. 향후 연구 방향

{report_type}에 적합한 형식으로 작성해주세요.
"""
                    
                    response = asyncio.run(
                        self.ai_orchestrator.generate_consensus(enhancement_prompt)
                    )
                    
                    if response['success']:
                        report_content = response['final_answer']
                
                # 보고서 표시
                st.markdown("### 📑 생성된 보고서")
                
                # 보고서 내용을 탭으로 구성
                if report_type == "프레젠테이션 자료":
                    # 슬라이드 형식으로 변환
                    slides = report_content.split('\n## ')
                    
                    slide_tabs = st.tabs([f"슬라이드 {i+1}" for i in range(len(slides))])
                    
                    for i, (tab, slide) in enumerate(zip(slide_tabs, slides)):
                        with tab:
                            if i > 0:
                                st.markdown(f"## {slide}")
                            else:
                                st.markdown(slide)
                else:
                    # 일반 보고서
                    st.markdown(report_content)
                
                # 다운로드 옵션
                st.markdown("### 💾 다운로드")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Markdown 다운로드
                    st.download_button(
                        label="📥 Markdown",
                        data=report_content,
                        file_name=f"polymer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # HTML 변환 및 다운로드
                    import markdown
                    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>고분자 실험 보고서</title>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 40px; }}
        h1 {{ color: #667eea; }}
        h2 {{ color: #764ba2; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{markdown.markdown(report_content, extensions=['tables', 'fenced_code'])}
</body>
</html>
"""
                    st.download_button(
                        label="📥 HTML",
                        data=html_content,
                        file_name=f"polymer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                
                with col3:
                    # PDF 생성 안내
                    st.info("PDF 변환은 HTML 파일을 브라우저에서 인쇄하세요")
        
        # 보고서 템플릿
        st.markdown("### 📋 보고서 템플릿")
        
        with st.expander("사용 가능한 템플릿 보기"):
            templates = {
                "학술 논문": """
# 제목

## Abstract
[영문 초록]

## 1. 서론
### 1.1 연구 배경
### 1.2 연구 목적

## 2. 실험
### 2.1 재료
### 2.2 실험 방법
### 2.3 특성 평가

## 3. 결과 및 고찰
### 3.1 [주요 결과 1]
### 3.2 [주요 결과 2]

## 4. 결론

## 참고문헌
""",
                "기술 보고서": """
# 기술 보고서

## 요약

## 1. 개요
- 프로젝트명:
- 기간:
- 목표:

## 2. 실험 설계
### 2.1 설계 방법론
### 2.2 실험 조건

## 3. 결과
### 3.1 실험 결과
### 3.2 데이터 분석

## 4. 결론 및 제언

## 부록
""",
                "사업 보고서": """
# 프로젝트 보고서

## Executive Summary

## 1. 프로젝트 개요
- 배경 및 필요성
- 목표 및 범위

## 2. 기술적 접근
- 방법론
- 주요 실험

## 3. 결과 및 성과
- 핵심 성과
- 기술적 달성도

## 4. 사업화 전략
- 시장 분석
- 경쟁 우위

## 5. 결론 및 향후 계획
"""
            }
            
            selected_template = st.selectbox(
                "템플릿 선택",
                options=list(templates.keys())
            )
            
            st.text_area(
                "템플릿 내용",
                value=templates[selected_template],
                height=300
            )
            
            if st.button("템플릿 사용"):
                st.session_state.report_template = templates[selected_template]
                st.success("템플릿이 적용되었습니다!")
    
    def _show_community(self):
        """커뮤니티 페이지"""
        st.title("👥 연구 커뮤니티")
        
        st.markdown("""
        고분자 연구자들과 지식을 공유하고 협업하세요!
        질문하고, 경험을 나누며, 함께 성장하는 공간입니다.
        """)
        
        # 커뮤니티 기능 탭
        tabs = st.tabs(["💬 토론", "❓ Q&A", "📊 공유 데이터", "🤝 협업 찾기"])
        
        with tabs[0]:  # 토론
            st.markdown("### 💬 최근 토론")
            
            # 토론 주제 (예시 데이터)
            discussions = [
                {
                    "title": "PLA의 결정화 속도 제어 방법",
                    "author": "김연구",
                    "replies": 12,
                    "views": 156,
                    "last_activity": "2시간 전"
                },
                {
                    "title": "나노필러 분산성 개선 팁 공유",
                    "author": "박과학",
                    "replies": 8,
                    "views": 89,
                    "last_activity": "5시간 전"
                },
                {
                    "title": "DSC 측정 시 주의사항",
                    "author": "이박사",
                    "replies": 15,
                    "views": 234,
                    "last_activity": "1일 전"
                }
            ]
            
            for discussion in discussions:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{discussion['title']}**")
                        st.caption(f"작성자: {discussion['author']}")
                    
                    with col2:
                        st.metric("답글", discussion['replies'])
                    
                    with col3:
                        st.metric("조회", discussion['views'])
                    
                    with col4:
                        st.caption(discussion['last_activity'])
                    
                    st.divider()
            
            # 새 토론 시작
            if st.button("➕ 새 토론 시작"):
                with st.form("new_discussion"):
                    title = st.text_input("제목")
                    category = st.selectbox(
                        "카테고리",
                        options=["일반", "실험 방법", "분석", "문제 해결", "기타"]
                    )
                    content = st.text_area("내용", height=200)
                    
                    if st.form_submit_button("게시"):
                        st.success("토론이 게시되었습니다!")
        
        with tabs[1]:  # Q&A
            st.markdown("### ❓ 질문과 답변")
            
            # 질문 필터
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search = st.text_input("질문 검색", placeholder="키워드 입력...")
            
            with col2:
                filter_type = st.selectbox(
                    "필터",
                    options=["모든 질문", "미해결", "해결됨", "내 질문"]
                )
            
            # 질문 목록 (예시)
            questions = [
                {
                    "title": "FTIR 피크 해석 도움 요청",
                    "status": "해결됨",
                    "answers": 3,
                    "tags": ["FTIR", "분석"]
                },
                {
                    "title": "고온에서 PP의 색상 변화 원인?",
                    "status": "미해결",
                    "answers": 1,
                    "tags": ["PP", "열화"]
                }
            ]
            
            for q in questions:
                with st.expander(f"{q['status']} - {q['title']}"):
                    st.write(f"답변: {q['answers']}개")
                    st.write(f"태그: {', '.join(q['tags'])}")
                    
                    if st.button(f"답변하기", key=f"answer_{q['title']}"):
                        st.text_area("답변 작성", key=f"answer_text_{q['title']}")
        
        with tabs[2]:  # 공유 데이터
            st.markdown("### 📊 공유된 실험 데이터")
            
            # 데이터 카테고리
            data_category = st.selectbox(
                "데이터 유형",
                options=["모든 데이터", "실험 설계", "측정 결과", "분석 데이터"]
            )
            
            # 공유 데이터 목록 (예시)
            shared_data = [
                {
                    "title": "PET/PBT 블렌드 기계적 물성 데이터",
                    "type": "측정 결과",
                    "format": "Excel",
                    "downloads": 45
                },
                {
                    "title": "나일론 6,6 반응표면 설계 매트릭스",
                    "type": "실험 설계",
                    "format": "CSV",
                    "downloads": 23
                }
            ]
            
            for data in shared_data:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{data['title']}**")
                        st.caption(f"유형: {data['type']} | 형식: {data['format']}")
                    
                    with col2:
                        st.metric("다운로드", data['downloads'])
                    
                    with col3:
                        st.button("⬇️ 다운로드", key=f"download_{data['title']}")
                    
                    st.divider()
            
            # 데이터 공유
            if st.checkbox("📤 내 데이터 공유하기"):
                with st.form("share_data"):
                    title = st.text_input("데이터 제목")
                    description = st.text_area("설명")
                    file = st.file_uploader("파일 업로드", type=['csv', 'xlsx', 'json'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        data_type = st.selectbox(
                            "데이터 유형",
                            options=["실험 설계", "측정 결과", "분석 데이터"]
                        )
                    
                    with col2:
                        license = st.selectbox(
                            "라이선스",
                            options=["CC BY", "CC BY-SA", "CC BY-NC", "All Rights Reserved"]
                        )
                    
                    if st.form_submit_button("공유하기"):
                        st.success("데이터가 공유되었습니다!")
        
        with tabs[3]:  # 협업 찾기
            st.markdown("### 🤝 협업 기회")
            
            # 협업 요청 목록
            st.markdown("#### 진행 중인 협업 요청")
            
            collabs = [
                {
                    "title": "바이오 기반 고분자 개발",
                    "skills": ["고분자 합성", "생분해성 평가"],
                    "duration": "6개월",
                    "status": "모집중"
                },
                {
                    "title": "나노복합재 전도성 향상 연구",
                    "skills": ["복합재료", "전기적 특성 분석"],
                    "duration": "3개월",
                    "status": "진행중"
                }
            ]
            
            for collab in collabs:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{collab['title']}**")
                        st.caption(f"필요 분야: {', '.join(collab['skills'])}")
                        st.caption(f"예상 기간: {collab['duration']}")
                    
                    with col2:
                        status_color = "🟢" if collab['status'] == "모집중" else "🟡"
                        st.write(f"{status_color} {collab['status']}")
                        
                        if st.button("참여 신청", key=f"join_{collab['title']}"):
                            st.success("참여 신청이 완료되었습니다!")
                    
                    st.divider()
            
            # 새 협업 요청
            if st.checkbox("🤝 새 협업 요청 등록"):
                with st.form("new_collaboration"):
                    project_title = st.text_input("프로젝트 제목")
                    project_desc = st.text_area("프로젝트 설명")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        required_skills = st.multiselect(
                            "필요한 전문 분야",
                            options=[
                                "고분자 합성", "고분자 가공", "복합재료",
                                "기계적 특성", "열적 특성", "전기적 특성",
                                "화학 분석", "물리 분석", "시뮬레이션"
                            ]
                        )
                    
                    with col2:
                        duration = st.selectbox(
                            "예상 기간",
                            options=["1개월", "3개월", "6개월", "1년", "1년 이상"]
                        )
                        
                        team_size = st.number_input(
                            "필요 인원",
                            min_value=1,
                            max_value=10,
                            value=2
                        )
                    
                    if st.form_submit_button("협업 요청 등록"):
                        st.success("협업 요청이 등록되었습니다!")
    
    def _show_api_settings(self):
        """API 설정 페이지"""
        st.title("🔑 API 설정")
        
        st.markdown("""
        ### 📋 API 키 관리
        
        Enhanced 기능을 사용하려면 해당 API 키를 입력해주세요.
        모든 키는 암호화되어 안전하게 저장됩니다.
        """)
        
        # API 상태 대시보드
        if self.enhanced_features and api_monitor:
            st.markdown("### 📊 API 상태 대시보드")
            
            # 전체 상태 요약
            api_status = api_monitor.get_all_status()
            
            if api_status:
                col1, col2, col3, col4 = st.columns(4)
                
                online_count = sum(1 for s in api_status.values() if s['status'] == APIStatus.ONLINE)
                error_count = sum(1 for s in api_status.values() if s['status'] == APIStatus.ERROR)
                slow_count = sum(1 for s in api_status.values() if s['status'] == APIStatus.SLOW)
                total_count = len(api_status)
                
                with col1:
                    st.metric("전체 API", total_count)
                with col2:
                    st.metric("정상", online_count, delta=f"{online_count/total_count*100:.0f}%")
                with col3:
                    st.metric("느림", slow_count)
                with col4:
                    st.metric("오류", error_count)
        
        # API 키 설정 섹션
        st.markdown("### 🤖 AI API 키")
        
        # AI API 탭
        ai_tabs = st.tabs(["Gemini", "Grok", "SambaNova", "DeepSeek", "Groq", "HuggingFace"])
        
        ai_configs = [
            ("gemini", "Gemini", "https://aistudio.google.com/app/apikey"),
            ("grok", "Grok", "https://x.ai/api"),
            ("sambanova", "SambaNova", "https://cloud.sambanova.ai/"),
            ("deepseek", "DeepSeek", "https://platform.deepseek.com/"),
            ("groq", "Groq", "https://console.groq.com/"),
            ("huggingface", "HuggingFace", "https://huggingface.co/settings/tokens")
        ]
        
        for tab, (key_id, name, url) in zip(ai_tabs, ai_configs):
            with tab:
                st.markdown(f"#### {name} API 설정")
                st.markdown(f"API 키 발급: [{url}]({url})")
                
                current_key = api_key_manager.get_key(key_id)
                
                new_key = st.text_input(
                    "API 키",
                    value=api_key_manager._mask_key(current_key) if current_key else "",
                    type="password",
                    key=f"api_{key_id}_key"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(f"저장", key=f"save_{key_id}"):
                        if new_key and new_key != api_key_manager._mask_key(current_key):
                            api_key_manager.set_key(key_id, new_key)
                            st.success(f"✅ {name} API 키가 저장되었습니다.")
                
                with col2:
                    if st.button(f"테스트", key=f"test_{key_id}"):
                        if current_key:
                            with st.spinner("연결 테스트 중..."):
                                # 동기적으로 실행
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                result = loop.run_until_complete(
                                    api_monitor.check_api_health(key_id)
                                )
                                loop.close()
                                
                                if result.success:
                                    st.success(f"✅ 연결 성공! (응답시간: {result.response_time:.2f}초)")
                                else:
                                    st.error(f"❌ 연결 실패: {result.error}")
                        else:
                            st.warning("먼저 API 키를 입력해주세요.")
        
        # 데이터베이스 API 섹션
        st.markdown("### 🗄️ 데이터베이스 API 키")
        
        db_tabs = st.tabs(["GitHub", "Materials Project", "기타 DB"])
        
        with db_tabs[0]:
            st.markdown("#### GitHub 설정")
            st.markdown("Personal Access Token 발급: [https://github.com/settings/tokens](https://github.com/settings/tokens)")
            
            current_key = api_key_manager.get_key('github')
            
            new_key = st.text_input(
                "GitHub Token",
                value=api_key_manager._mask_key(current_key) if current_key else "",
                type="password",
                key="api_github_token"
            )
            
            if st.button("저장 및 테스트", key="save_github"):
                if new_key and new_key != api_key_manager._mask_key(current_key):
                    api_key_manager.set_key('github', new_key)
                    st.success("✅ GitHub 토큰이 저장되었습니다.")
        
        with db_tabs[1]:
            st.markdown("#### Materials Project 설정")
            st.markdown("API 키 발급: [https://materialsproject.org/api](https://materialsproject.org/api)")
            
            current_key = api_key_manager.get_key('materials_project')
            
            new_key = st.text_input(
                "MP API Key",
                value=api_key_manager._mask_key(current_key) if current_key else "",
                type="password",
                key="api_mp_key"
            )
            
            if st.button("저장", key="save_mp"):
                if new_key:
                    api_key_manager.set_key('materials_project', new_key)
                    st.success("✅ Materials Project API 키가 저장되었습니다.")
        
        with db_tabs[2]:
            st.markdown("#### 기타 데이터베이스")
            
            # 추가 DB API 설정
            other_dbs = [
                ("materials_commons", "Materials Commons"),
                ("zenodo", "Zenodo"),
                ("protocols_io", "Protocols.io"),
                ("figshare", "Figshare")
            ]
            
            for key_id, name in other_dbs:
                with st.expander(name):
                    current_key = api_key_manager.get_key(key_id)
                    
                    new_key = st.text_input(
                        f"{name} API Key",
                        value=api_key_manager._mask_key(current_key) if current_key else "",
                        type="password",
                        key=f"api_{key_id}_key"
                    )
                    
                    if st.button(f"저장", key=f"save_{key_id}"):
                        if new_key:
                            api_key_manager.set_key(key_id, new_key)
                            st.success(f"✅ {name} API 키가 저장되었습니다.")
        
        # 일괄 테스트
        st.markdown("### 🧪 전체 API 테스트")
        
        if st.button("🔍 모든 API 연결 테스트", use_container_width=True):
            if api_monitor:
                with st.spinner("모든 API를 테스트하는 중..."):
                    # 프로그레스 바
                    progress_bar = st.progress(0)
                    status_container = st.empty()
                    
                    all_apis = list(api_key_manager.api_configs.keys())
                    results = {}
                    
                    for i, api_name in enumerate(all_apis):
                        status_container.text(f"테스트 중: {api_name}...")
                        progress_bar.progress((i + 1) / len(all_apis))
                        
                        # 테스트 실행
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(
                            api_monitor.check_api_health(api_name)
                        )
                        loop.close()
                        
                        results[api_name] = result
                    
                    # 결과 표시
                    progress_bar.empty()
                    status_container.empty()
                    
                    st.markdown("#### 테스트 결과")
                    
                    # 결과 테이블
                    result_data = []
                    for api_name, result in results.items():
                        config = api_key_manager.api_configs.get(api_name, {})
                        
                        result_data.append({
                            'API': config.get('name', api_name),
                            '카테고리': config.get('category', 'unknown'),
                            '상태': '✅ 정상' if result.success else '❌ 오류',
                            '응답시간': f"{result.response_time:.2f}s" if result.success else '-',
                            '오류': result.error if not result.success else ''
                        })
                    
                    result_df = pd.DataFrame(result_data)
                    st.dataframe(result_df)
            else:
                st.warning("API 모니터링 기능을 사용할 수 없습니다.")
        
        # API 사용량 통계
        if api_monitor and st.checkbox("📊 API 사용량 통계 보기"):
            st.markdown("### 📊 API 사용 통계")
            
            metrics = st.session_state.get('api_metrics', {})
            
            if metrics:
                # 사용량 차트
                usage_data = []
                for api_name, metric in metrics.items():
                    usage_data.append({
                        'API': api_name,
                        '총 호출': metric['total_calls'],
                        '성공': metric['success_calls'],
                        '성공률': metric['success_calls'] / metric['total_calls'] * 100 if metric['total_calls'] > 0 else 0
                    })
                
                if usage_data:
                    usage_df = pd.DataFrame(usage_data)
                    
                    # 막대 그래프
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=usage_df['API'],
                        y=usage_df['총 호출'],
                        name='총 호출',
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        x=usage_df['API'],
                        y=usage_df['성공'],
                        name='성공',
                        marker_color='green'
                    ))
                    fig.update_layout(
                        title="API 호출 통계",
                        xaxis_title="API",
                        yaxis_title="호출 수",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("아직 API 사용 통계가 없습니다.")

# ==================== 메인 실행 ====================
def main():
    st.set_page_config(
        page_title="🧬 고분자 실험 설계 플랫폼",
        page_icon="🧬",
        layout="wide"
    )
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 사이드바
    with st.sidebar:
        st.title("🧬 고분자 실험 설계")
        
        # 네비게이션
        pages = {
            'home': '🏠 홈',
            'project': '📋 프로젝트 설정',
            'design': '🔬 실험 설계',
            'analysis': '📊 결과 분석',
            'settings': '⚙️ 설정'
        }
        
        # 현재 페이지 선택
        current = st.session_state.current_page
        for page_id, page_name in pages.items():
            if st.button(page_name, use_container_width=True, 
                        type="primary" if current == page_id else "secondary"):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.divider()
        
        # 연결 상태
        st.markdown("### 연결 상태")
        
        # Google Sheets 연결 상태
        db_status = "🟢 연결됨" if st.session_state.db_manager.is_connected() else "🔴 미연결"
        st.info(f"Google Sheets: {db_status}")
        
        # API 상태 요약
        api_manager = st.session_state.api_key_manager
        configured_apis = sum(1 for status in api_manager.api_status.values() if status == 'configured')
        total_apis = len(api_manager.api_status)
        st.info(f"API: {configured_apis}/{total_apis} 설정됨")
    
    # 페이지 라우팅
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'project':
        show_project_page()
    elif st.session_state.current_page == 'design':
        show_design_page()
    elif st.session_state.current_page == 'analysis':
        show_analysis_page()
    elif st.session_state.current_page == 'settings':
        show_settings_page()

def show_home_page():
    """홈 페이지"""
    st.title("🧬 고분자 실험 설계 플랫폼")
    st.markdown("### AI 기반 스마트 실험 설계 시스템")
    
    # 환영 메시지
    st.info("Pectin, Cellulose 용해성 향상을 위한 DES 활용 실험 설계 플랫폼입니다.")
    
    # 빠른 시작
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🆕 새 프로젝트 시작", use_container_width=True):
            st.session_state.current_page = 'project'
            st.rerun()
    
    with col2:
        if st.button("📖 사용 가이드", use_container_width=True):
            with st.expander("플랫폼 사용 가이드", expanded=True):
                st.markdown("""
                1. **프로젝트 설정**: 실험 목표와 대상 고분자 선택
                2. **실험 설계**: AI 추천 또는 수동으로 실험 설계
                3. **결과 분석**: 실험 결과 업로드 및 분석
                4. **설정**: API 키 및 사용자 레벨 설정
                """)
    
    with col3:
        if st.button("⚙️ 설정", use_container_width=True):
            st.session_state.current_page = 'settings'
            st.rerun()

def show_project_page():
    """프로젝트 설정 페이지"""
    st.title("📋 프로젝트 설정")
    
    with st.form("project_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            project_name = st.text_input("프로젝트 이름", 
                                       placeholder="예: Pectin DES 용해성 향상")
            polymer_type = st.selectbox("대상 고분자",
                                      ["Pectin", "Cellulose", "Chitosan", "기타"])
        
        with col2:
            des_type = st.selectbox("DES 유형",
                                   ["ChCl-Urea", "ChCl-Glycerol", "ChCl-Ethylene glycol", "기타"])
            target_property = st.multiselect("목표 물성",
                                           ["용해도", "점도", "안정성", "투명도"])
        
        objectives = st.text_area("실험 목표", 
                                placeholder="달성하고자 하는 구체적인 목표를 입력하세요")
        
        submitted = st.form_submit_button("프로젝트 생성", use_container_width=True)
        
        if submitted:
            if project_name and polymer_type and des_type:
                project_data = {
                    'name': project_name,
                    'polymer': polymer_type,
                    'des': des_type,
                    'properties': ', '.join(target_property),
                    'objectives': objectives,
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.session_state.project_info = project_data
                
                # Google Sheets에 저장 시도
                if st.session_state.db_manager.save_project(project_data):
                    st.success("프로젝트가 생성되고 저장되었습니다!")
                else:
                    st.success("프로젝트가 생성되었습니다! (로컬 저장)")
                
                st.balloons()
            else:
                st.error("필수 항목을 모두 입력해주세요.")

def show_design_page():
    """실험 설계 페이지"""
    st.title("🔬 실험 설계")
    
    if not st.session_state.project_info:
        st.warning("먼저 프로젝트를 설정해주세요.")
        if st.button("프로젝트 설정으로 이동"):
            st.session_state.current_page = 'project'
            st.rerun()
        return
    
    # 프로젝트 정보 표시
    with st.expander("프로젝트 정보", expanded=False):
        st.json(st.session_state.project_info)
    
    # 실험 인자 설정
    st.markdown("### 실험 인자 설정")
    
    # 기본 인자들
    factors = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        temp_min = st.number_input("온도 최소값 (°C)", value=25, step=5)
        temp_max = st.number_input("온도 최대값 (°C)", value=50, step=5)
        temp_levels = st.number_input("온도 수준 수", value=2, min_value=2, max_value=5)
        
        if temp_levels == 2:
            factors['온도'] = [temp_min, temp_max]
        else:
            factors['온도'] = list(np.linspace(temp_min, temp_max, temp_levels))
    
    with col2:
        time_min = st.number_input("시간 최소값 (min)", value=30, step=10)
        time_max = st.number_input("시간 최대값 (min)", value=120, step=10)
        time_levels = st.number_input("시간 수준 수", value=2, min_value=2, max_value=5)
        
        if time_levels == 2:
            factors['시간'] = [time_min, time_max]
        else:
            factors['시간'] = list(np.linspace(time_min, time_max, time_levels))
    
    # 실험 설계 생성
    if st.button("실험 설계 생성", use_container_width=True):
        design_matrix = create_doe_matrix(factors)
        st.session_state.experiment_design = design_matrix
        
        st.success("실험 설계가 생성되었습니다!")
        
        # 설계 매트릭스 표시
        st.markdown("### 실험 설계 매트릭스")
        st.dataframe(design_matrix, use_container_width=True)
        
        # 다운로드 옵션
        col1, col2 = st.columns(2)
        with col1:
            download_csv(design_matrix, f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        with col2:
            try:
                download_excel(design_matrix, f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            except Exception as e:
                st.error(f"Excel 다운로드 오류: {e}")
    
    # 시각화
    if st.session_state.experiment_design is not None:
        st.markdown("### 실험점 시각화")
        show_experiment_visualization()

def show_analysis_page():
    """결과 분석 페이지"""
    st.title("📊 결과 분석")
    
    uploaded_file = st.file_uploader("실험 결과 CSV 파일 업로드", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.results_df = df
            
            st.success("파일이 업로드되었습니다!")
            
            # 데이터 미리보기
            st.markdown("### 데이터 미리보기")
            st.dataframe(df.head(), use_container_width=True)
            
            # 기본 통계
            st.markdown("### 기본 통계")
            st.dataframe(df.describe(), use_container_width=True)
            
            # 시각화
            st.markdown("### 데이터 시각화")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("X축", numeric_cols)
                
                with col2:
                    y_col = st.selectbox("Y축", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                
                if x_col and y_col:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

def show_settings_page():
    """설정 페이지"""
    st.title("⚙️ 설정")
    
    # 사용자 설정
    show_user_settings()
    
    st.divider()
    
    # API 설정
    show_api_status()
    
    st.divider()
    
    # Google Sheets 설정
    st.markdown("### 📊 Google Sheets 설정")
    
    if st.session_state.db_manager.is_connected():
        st.success("✅ Google Sheets가 연결되어 있습니다.")
    else:
        st.warning("Google Sheets가 연결되지 않았습니다.")
        
        with st.expander("설정 방법"):
            st.markdown("""
            1. Google Cloud Console에서 서비스 계정 생성
            2. Google Sheets API 활성화
            3. 서비스 계정 JSON 키 다운로드
            4. `.streamlit/secrets.toml`에 다음 형식으로 추가:
            
            ```toml
            private_gsheets_url = "YOUR_SHEET_URL"
            
            [gcp_service_account]
            type = "service_account"
            project_id = "your-project"
            # ... JSON 키 내용
            ```
            
            5. 생성한 시트에 서비스 계정 이메일 편집자 권한 부여
            """)

if __name__ == "__main__":
    main()
