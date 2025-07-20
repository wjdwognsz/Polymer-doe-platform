🧬 고분자 실험 설계 플랫폼 - 통합 실행 가이드
📋 체크리스트
1. 필수 패키지 설치
bashpip install streamlit pandas numpy plotly google-generativeai asyncio
2. 디렉토리 구조
polymer-doe-platform/
├── polymer_platform.py      # 메인 애플리케이션
├── project_setup_page.py    # 프로젝트 설정 페이지 (제공된 코드)
├── .streamlit/
│   └── secrets.toml        # API 키 설정
└── projects/               # 프로젝트 저장 폴더 (자동 생성)
3. secrets.toml 설정
.streamlit/secrets.toml 파일 생성:
toml# AI API Keys
google_gemini = "your-gemini-api-key"
xai_grok = ""  # 선택사항
sambanova = ""  # 선택사항

# 기타 서비스 (선택사항)
github = ""
materials_project = ""
🔧 기존 코드와의 통합
옵션 1: 독립 실행 (권장)
제공된 ProjectSetupPage 코드를 그대로 사용:
python# project_setup_page.py로 저장 후
streamlit run project_setup_page.py
옵션 2: 기존 앱에 통합
기존 polymer_platform.py에 통합하는 경우:

클래스 임포트 추가

python# polymer_platform.py 상단에 추가
from project_setup_page import ProjectSetupPage, setup_api_keys

UserInterfaceSystem 수정

pythonclass UserInterfaceSystem:
    def __init__(self):
        self.pages = {
            'home': HomePage(),
            'project_setup': ProjectSetupPage(),  # 새로운 페이지 사용
            # ... 나머지 페이지들
        }

API 초기화 추가

pythonasync def _initialize_systems(self):
    """시스템 초기화"""
    # API 키 설정
    api_keys = setup_api_keys()
    
    # AI Orchestrator 초기화
    if api_keys.get('google_gemini'):
        from ai_orchestrator import AIOrchestrator
        st.session_state.ai_orchestrator = AIOrchestrator(api_keys)
⚠️ 주의사항
1. 세션 상태 키 충돌

각 위젯의 key 파라미터가 고유해야 함
특히 user_level 같은 공통 키 주의

2. 비동기 함수 처리

Streamlit은 기본적으로 동기 실행
asyncio.run() 대신 제공된 패턴 사용

3. API 키 보안

.gitignore에 .streamlit/ 추가
실제 배포 시 Streamlit Cloud의 Secrets 사용

🎯 빠른 테스트

최소 설정으로 테스트

python# 간단한 테스트 (API 키 없이)
streamlit run project_setup_page.py

AI 기능 테스트


Google Gemini API 키만 설정
"AI 프로젝트 설계 도우미" 버튼 클릭


전체 워크플로우 테스트


프로젝트 정보 입력
템플릿 선택 및 적용
저장 버튼 클릭

📊 예상 결과
✅ 정상 작동 시:

모든 UI 요소가 정상 표시
입력값이 세션에 저장됨
페이지 전환 시 데이터 유지

❌ 문제 발생 시:

API 키 미설정: AI 기능만 비활성화
모듈 누락: 해당 기능만 비활성화
키 충돌: 위젯 동작 이상

🆘 트러블슈팅
"Module not found" 오류
bashpip install [누락된_패키지]
"Key already exists" 오류

위젯 키에 고유 접미사 추가
예: key="user_level" → key="project_setup_user_level"

AI 응답 없음

API 키 확인
네트워크 연결 확인
API 사용량 한도 확인

🎉 성공!
이제 고분자 실험 설계 플랫폼의 핵심 기능인 프로젝트 설정이 작동합니다!
