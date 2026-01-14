#!/usr/bin/env bash
set -e

ASIC_DIR=~/asic
REQ_FILE=$ASIC_DIR/requirements_venv.txt
PYTHON_VERSION=3.11.13
PIPENV_VERSION=2024.1.0


echo "[0] 기존 pipenv 및 캐시 정리 중..."
pyenv exec pipenv --rm || true
rm -rf ~/.local/share/virtualenvs
rm -rf ~/.cache/pipenv
rm -rf ~/.cache/pip
rm -rf $ASIC_DIR/Pipfile $ASIC_DIR/Pipfile.lock
echo "✅ 기존 환경 정리 완료."
echo ""

# 1. pyenv 초기화
export PATH="$HOME/.pyenv/bin:$PATH"

if ! command -v pyenv >/dev/null 2>&1; then
    echo "[1] pyenv 설치 중..."
    curl https://pyenv.run | bash
fi

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 2. Python 3.11.1 설치 확인
if ! pyenv versions --bare | grep -q "^$PYTHON_VERSION$"; then
    echo "[2] Python $PYTHON_VERSION 설치 중..."
    pyenv install $PYTHON_VERSION
fi

# 3. pyenv local 적용
cd $ASIC_DIR
pyenv local $PYTHON_VERSION

# 🔥 (NEW) 기존 pipenv 환경 완전 삭제
if pyenv exec pipenv --venv >/dev/null 2>&1; then
    echo "[3-1] 기존 pipenv 환경 삭제 중..."
    pyenv exec pipenv --rm || true
fi
rm -rf ~/.cache/pipenv ~/.local/share/virtualenvs || true

# 4. Pyenv 환경에서 pipenv 설치 (시스템 pipenv와 충돌 방지)
echo "[3] Pyenv 환경에서 pipenv 설치 중..."
export PIP_CONSTRAINT=
pyenv exec python -m pip install --upgrade pip
pyenv exec python -m pip install "pipenv==$PIPENV_VERSION"

# 5. pipenv 환경 생성 (Lock은 나중에)
echo "[4] pipenv 환경 생성 중..."
pyenv exec pipenv --python $PYTHON_VERSION install --skip-lock

# 6. requirements.txt 설치
if [ -f "$REQ_FILE" ]; then
    echo "[5] requirements.txt 설치 중..."
    pyenv exec pipenv install -r $REQ_FILE
else
    echo "⚠️ requirements 파일($REQ_FILE)이 없습니다."
    exit 1
fi

# 7. Pipfile.lock 생성
echo "[6] Pipfile.lock 생성 중..."
pyenv exec pipenv lock

# 8. 의존성 트리 생성 및 충돌 탐지
echo ""
echo "🔍 [의존성 충돌 자동 점검 중...]"
pyenv exec pipenv run python -m pip install --quiet pipdeptree
pyenv exec pipenv run pipdeptree --warn silence > /tmp/pipdeptree_output.txt
grep -i "numpy\|urllib3" /tmp/pipdeptree_output.txt > /tmp/conflicts.txt || true

if [ -s /tmp/conflicts.txt ]; then
    echo "⚠️ 다음 패키지들이 버전 충돌 가능성이 있습니다:"
    cat /tmp/conflicts.txt
else
    echo "✅ 주요 충돌(numpy, urllib3)은 감지되지 않았습니다."
fi

# 9. 전체 의존성 트리 저장
echo ""
echo "[📦 전체 패키지 트리 저장 중...]"
pyenv exec pipenv run pipdeptree > $ASIC_DIR/pip_dependency_tree.txt

echo ""
echo "[완료] pipenv 가상환경 생성 및 의존성 점검 완료!"
echo "👉 환경 진입: pyenv exec pipenv shell"
echo "👉 환경에서 실행: pyenv exec pipenv run python your_script.py"
