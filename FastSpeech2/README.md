# Conference

## Preprocess.py
- 데이터 전처리
- "lexicon, lab 파일 생성" 및 "data input 생성"
```
python preprocess.py --cfg <train_config_path> --generate_lab <true/false> --preprocessor <true/false>
```
- `--cfg`: configuration json 파일 경로 (default: train_config.json)
- `--generate_lab`: lab 파일 및 lexicon 파일 생성 함수 실행 (default: false)
  - mfa 학습하기 이전에 해당 코드를 통해 lexicon 및 lab 파일 생성해야 함
- `--preprocessor`: FastSpeech2 data input 생성 모듈 실행

## mfa/install_mfa.sh
- mfa 모델 설치
```
cd mfa
sh install_mfa.sh
```

## mfa/run_mfa.sh
- mfa 모델 학습 및 TextGrid 생성
```
cd mfa
sh run_mfa.sh <dir>
```
- `dir`: raw dataset 폴더 이름
