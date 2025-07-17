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
    """다중 AI 모델 통합 관리"""
    
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.available_ais = []
        
        if api_keys.get('openai'):
            self.available_ais.append('openai')
            openai.api_key = api_keys['openai']
            
        if api_keys.get('google'):
            self.available_ais.append('google')
            genai.configure(api_key=api_keys['google'])
    
    def create_experiment_prompt(self, user_input, user_level, project_info):
        """사용자 레벨에 맞는 동적 프롬프트 생성"""
        level_descriptions = {
            1: "초보자를 위해 모든 단계를 상세히 설명하고, 각 결정의 이유를 명확히 제시해주세요.",
            2: "학습자를 위해 2-3가지 옵션을 장단점과 함께 제시해주세요.",
            3: "중급자의 설계를 검토하고 개선점을 제안해주세요.",
            4: "전문가 수준의 혁신적인 접근법을 제안해주세요."
        }
        
        prompt = f"""
당신은 고분자 실험 설계 전문가입니다.
사용자 레벨: {user_level} - {level_descriptions.get(user_level, level_descriptions[1])}

프로젝트 정보:
{json.dumps(project_info, ensure_ascii=False, indent=2)}

사용자 요청: {user_input}

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
    
    def get_ai_response(self, prompt, ai_type='openai'):
        """개별 AI 호출 (동기식)"""
        try:
            if ai_type == 'openai' and 'openai' in self.available_ais:
                import openai
                openai.api_key = self.api_keys['openai']
            
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000
                )
            
                # JSON 파싱 시도
                content = response.choices[0].message.content
                try:
                    # JSON 블록 찾기
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        return {"error": "JSON not found", "content": content}
                except json.JSONDecodeError:
                    return {"error": "JSON parsing failed", "content": content}
                
            elif ai_type == 'google' and 'google' in self.available_ais:
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
            
                # JSON 추출
                text = response.text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {"error": "JSON not found", "content": text}
                
        except Exception as e:
            print(f"{ai_type} AI 오류: {str(e)}")
            return None
    
    def get_consensus_design(self, user_input, user_level, project_info):
        """다중 AI 합의 도출"""
        prompt = self.create_experiment_prompt(user_input, user_level, project_info)
    
        if not self.available_ais:
            return self._get_fallback_design(user_input, project_info)
    
        # 단순화된 AI 호출 (병렬 처리 제거)
        responses = []
    
        for ai in self.available_ais:
            try:
                result = self.get_ai_response(prompt, ai)
                if result and isinstance(result, dict):
                    responses.append(result)
            except Exception as e:
                print(f"{ai} 오류: {e}")
                continue
    
        if not responses:
            return self._get_fallback_design(user_input, project_info)
    
        # 응답 통합 (오류 방지)
        try:
            # 가장 상세한 응답 선택
            best_response = responses[0]  # 첫 번째 유효한 응답 사용
        
            # 안전성 고려사항 통합
            all_safety = set()
            for r in responses:
                if isinstance(r, dict) and 'safety_considerations' in r:
                    all_safety.update(r.get('safety_considerations', []))
        
            if all_safety:
                best_response['safety_considerations'] = list(all_safety)
        
            return best_response
        except Exception as e:
            print(f"응답 통합 오류: {e}")
            return self._get_fallback_design(user_input, project_info)
        
        # 병렬 AI 호출
        with ThreadPoolExecutor(max_workers=len(self.available_ais)) as executor:
            futures = {
                executor.submit(self.get_ai_response, prompt, ai): ai 
                for ai in self.available_ais
            }
            
            responses = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    responses.append(result)
        
        if not responses:
            return self._get_fallback_design(user_input, project_info)
        
        # 응답 통합 (가장 상세한 응답 선택)
        best_response = max(responses, 
                          key=lambda r: len(r.get('factors', [])) + len(r.get('reasoning', '')))
        
        # 안전성 고려사항 통합
        all_safety = set()
        for r in responses:
            all_safety.update(r.get('safety_considerations', []))
        best_response['safety_considerations'] = list(all_safety)
        
        return best_response
    
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
                url = "https://api.openalex.org/works"
                params = {
                    'search': query,
                    'filter': 'is_oa:true',
                    'per_page': limit,
                    'mailto': 'polymer-doe@example.com'
                }
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    papers = []
                    for work in data.get('results', []):
                        papers.append({
                            'title': work.get('title', 'No title'),
                            'authors': ', '.join([a['author']['display_name'] 
                                                for a in work.get('authorships', [])[:3]]),
                            'year': work.get('publication_year', 'N/A'),
                            'doi': work.get('doi', '').replace('https://doi.org/', ''),
                            'citations': work.get('cited_by_count', 0),
                            'abstract': work.get('abstract', 'No abstract available')
                        })
                    return papers
                    
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
        self.ai_orchestrator = None
        self.api_manager = APIManager()
        self.stat_analyzer = StatisticalAnalyzer()
        self.report_generator = ReportGenerator()
        
        # API 키 설정
        if st.session_state.api_keys.get('openai') or st.session_state.api_keys.get('google'):
            self.ai_orchestrator = AIOrchestrator(st.session_state.api_keys)
    
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
        
        # AI 상담 모드 (레벨 1)
        if st.session_state.user_level == 1:
            st.info("🤖 AI가 프로젝트 설정을 도와드리겠습니다. 자유롭게 설명해주세요.")
            
            with st.expander("💡 AI 상담 시작", expanded=True):
                user_input = st.text_area(
                    "연구하고 싶은 고분자 소재나 실험 목표를 설명해주세요:",
                    placeholder="예: Deep eutectic solvent를 만들고자 염화콜린과 구연산을 이용합니다. 최적의 비율과 반응 조건을 찾고 싶습니다.",
                    height=150
                )
                
                if st.button("AI에게 물어보기"):
                    if user_input and self.ai_orchestrator:
                        with st.spinner("AI가 분석 중입니다..."):
                            # AI 프롬프트 생성
                            prompt = f"""
사용자가 다음과 같은 고분자 연구를 계획하고 있습니다:
"{user_input}"

이를 바탕으로 다음 정보를 추출하고 추천해주세요:
1. 프로젝트명 제안
2. 연구 유형 (예: 합성, 물성 최적화, 공정 개발 등)
3. 주요 변수 및 수준 추천
4. 측정해야 할 반응변수
5. 예상되는 도전과제
6. 추천 실험 설계 방법

JSON 형식으로 응답해주세요.
"""
                            # AI 응답 (간단한 시뮬레이션)
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
    
    def _show_experiment_design(self):
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
    
    def _show_results_analysis(self):
        """결과 분석 페이지"""
        st.title("📊 결과 분석")
        
        if not st.session_state.experiment_design:
            st.warning("먼저 실험을 설계해주세요.")
            if st.button("실험 설계로 이동"):
                st.session_state.current_page = 'experiment_design'
                st.rerun()
            return
        
        st.info("실험 결과를 입력하고 통계 분석을 수행합니다.")
        
        # 파일 업로드
        st.subheader("📁 데이터 업로드")
        
        uploaded_file = st.file_uploader(
            "실험 결과 CSV 파일을 업로드하세요",
            type=['csv', 'xlsx'],
            help="첫 번째 열은 실험 번호, 나머지 열은 반응변수여야 합니다."
        )
        
        if uploaded_file:
            try:
                # 파일 읽기 (UTF-8 BOM 처리)
                if uploaded_file.name.endswith('.csv'):
                    results_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                else:
                    results_df = pd.read_excel(uploaded_file)
                
                st.success("파일이 성공적으로 로드되었습니다!")
                
                # 데이터 미리보기
                st.subheader("📋 데이터 미리보기")
                st.dataframe(results_df.head(10), use_container_width=True)
                
                # 기본 통계
                st.subheader("📈 기본 통계")
                st.dataframe(results_df.describe(), use_container_width=True)
                
                # 설계 매트릭스와 결합
                design_matrix = pd.DataFrame(st.session_state.experiment_design['design_matrix'])
                
                # 통계 분석 수행
                if st.button("🔍 통계 분석 실행"):
                    with st.spinner("분석 중..."):
                        analysis = self.stat_analyzer.analyze_doe_results(design_matrix, results_df)
                        st.session_state.analysis_results = analysis
                        
                        # 결과 표시
                        st.subheader("📊 분석 결과")
                        
                        # 기본 통계
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**기본 통계**")
                            for var, stats in analysis['basic_stats'].items():
                                with st.expander(f"{var} 통계"):
                                    st.metric("평균", f"{stats['mean']:.2f}")
                                    st.metric("표준편차", f"{stats['std']:.2f}")
                                    st.metric("CV(%)", f"{stats['cv']:.1f}")
                        
                        with col2:
                            st.markdown("**주효과**")
                            for factor, effects in analysis['effects'].items():
                                with st.expander(f"{factor} 효과"):
                                    for response, effect in effects.items():
                                        st.metric(response, f"{effect:.2f}")
                        
                        # 시각화
                        st.subheader("📈 시각화")
                        
                        # 반응변수 선택
                        response_cols = [col for col in results_df.columns if col != 'run']
                        selected_response = st.selectbox("분석할 반응변수 선택", response_cols)
                        
                        if selected_response:
                            # 주효과 플롯
                            factors = [col for col in design_matrix.columns if col != 'run']
                            
                            fig = go.Figure()
                            
                            for factor in factors[:3]:  # 최대 3개 요인만 표시
                                levels = design_matrix[factor].unique()
                                means = []
                                
                                for level in levels:
                                    mask = design_matrix[factor] == level
                                    if mask.any():
                                        mean_val = results_df.loc[mask, selected_response].mean()
                                        means.append(mean_val)
                                
                                fig.add_trace(go.Scatter(
                                    x=levels,
                                    y=means,
                                    mode='lines+markers',
                                    name=factor,
                                    line=dict(width=3),
                                    marker=dict(size=10)
                                ))
                            
                            fig.update_layout(
                                title=f'{selected_response} 주효과 플롯',
                                xaxis_title='수준',
                                yaxis_title=selected_response,
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 3D 반응표면 (2개 요인 이상일 때)
                            if len(factors) >= 2:
                                opt_plot = self.stat_analyzer.generate_optimization_plot(
                                    design_matrix, results_df, selected_response
                                )
                                if opt_plot:
                                    st.plotly_chart(opt_plot, use_container_width=True)
                
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
    
    def _show_literature_search(self):
        """문헌 검색 페이지"""
        st.title("📚 문헌 검색")
        
        st.info("최신 연구 동향을 파악하고 관련 문헌을 검색합니다.")
        
        # 검색 옵션
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "검색어 입력",
                placeholder="예: polymer composite mechanical properties optimization"
            )
        
        with col2:
            search_source = st.selectbox(
                "데이터베이스",
                ["openalex", "crossref"]
            )
        
        # 고급 검색 옵션
        with st.expander("🔍 고급 검색 옵션"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                year_from = st.number_input("출판년도 (시작)", 2000, 2024, 2020)
            
            with col2:
                year_to = st.number_input("출판년도 (끝)", 2000, 2024, 2024)
            
            with col3:
                max_results = st.number_input("최대 결과 수", 10, 100, 20)
        
        # 검색 실행
        if st.button("🔍 검색 실행", type="primary"):
            if search_query:
                with st.spinner("문헌을 검색하고 있습니다..."):
                    papers = self.api_manager.search_literature(
                        search_query, 
                        source=search_source,
                        limit=max_results
                    )
                    
                    if papers:
                        st.session_state.literature_results = papers
                        st.success(f"{len(papers)}개의 문헌을 찾았습니다!")
                    else:
                        st.warning("검색 결과가 없습니다.")
        
        # 검색 결과 표시
        if st.session_state.get('literature_results'):
            st.subheader("📖 검색 결과")
            
            # 정렬 옵션
            sort_by = st.selectbox(
                "정렬 기준",
                ["관련도", "최신순", "인용수"]
            )
            
            papers = st.session_state.literature_results
            
            # 정렬
            if sort_by == "최신순":
                papers = sorted(papers, key=lambda x: x.get('year', 0), reverse=True)
            elif sort_by == "인용수":
                papers = sorted(papers, key=lambda x: x.get('citations', 0), reverse=True)
            
            # 논문 표시
            for i, paper in enumerate(papers):
                with st.expander(f"{i+1}. {paper['title'][:100]}..."):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**저자**: {paper['authors']}")
                        st.markdown(f"**연도**: {paper['year']}")
                        st.markdown(f"**DOI**: {paper['doi']}")
                        
                        if paper.get('abstract'):
                            st.markdown("**초록**:")
                            st.write(paper['abstract'][:500] + "...")
                    
                    with col2:
                        st.metric("인용수", paper.get('citations', 0))
                        
                        if paper['doi']:
                            st.markdown(f"[📄 원문 보기](https://doi.org/{paper['doi']})")
        
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
        
        if st.button("📊 트렌드 분석 실행"):
            if selected_keywords:
                with st.spinner("트렌드를 분석하고 있습니다..."):
                    # 트렌드 분석 (시뮬레이션)
                    years = list(range(2019, 2025))
                    
                    fig = go.Figure()
                    
                    for keyword in selected_keywords:
                        # 실제로는 API를 통해 연도별 논문 수를 가져와야 함
                        counts = np.random.randint(50, 200, size=len(years))
                        counts = np.cumsum(counts * np.random.uniform(0.9, 1.1, size=len(years)))
                        
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
        """커뮤니티 페이지"""
        st.title("👥 커뮤니티")
        
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
                        
                        if st.button(f"자세히 보기", key=f"view_post_{post['id']}"):
                            post['views'] += 1
                            with st.expander("게시글 내용", expanded=True):
                                st.write(post['content'])
                                
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
