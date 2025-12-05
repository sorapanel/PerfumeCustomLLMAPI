# 香水チャットボット用 Custom LLM（OpenCALM-3B LoRA Fine-tuning）  
  
本プロジェクトは cyberagent/open-calm-3b をベースモデルとして、    
香水に特化した Q&A チャットボットを構築するためのファインチューニング環境です。  
  
---  
  
## 📦 Dataset Summary  
  
- 総データ数：26,075 件  
  - 学習データ：23,467 件  
  - テストデータ：2,608 件  
  
データ内容はすべて Q&A 形式で構成されており、  
香水名・シーン別おすすめ・香りの特徴などを含む多様な質問タイプに対応しています。  
  
---  
  
## 📁 File Description
  
getData.py         # スクレイピングによる香水データ収集  
arrangeData.py     # instructions / responses 形式への整形  
loadModel.py       # モデル学習処理（LoRA + OpenCALM）  
generate.py        # 推論用スクリプト  
  
---  
  
## 🚀 Training Pipeline 概要  
  
学習処理（`loadModel.py`）では以下の処理を行います：  
  
### 1. JSON データの読み込み・検証  
- 不正 JSON・欠損データなどを検出し除外  
  
### 2. Train / Test への分割  
- train：23,467  
- test：2,608  
  
### 3. OpenCALM-3B のロード  
- `"cyberagent/open-calm-3b"` を使用  
- pad_token を追加（LoRA学習安定化のため必須）  
  
### 4. LoRA 設定  
- `r = 16`  
- `lora_alpha = 32`  
- `lora_dropout = 0.05`  
- `task_type = "CAUSAL_LM"`  

### 5. TRL SFTTrainer による学習  
- instruction → response 形式の supervised fine-tuning  
- GPU メモリ節約のため勾配蓄積を併用  
  
### 6. モデル保存  
- `fine_tuned_open_calm/` ディレクトリに保存  
  
---  
  
## ⚙️ TrainingArguments（最適化設定）  
  
学習では以下の設定を使用しています：  
  
| パラメータ | 値 |
|-----------|-----|
| scheduler | cosine_with_restarts |
| batch size | 4 × accumulation 4（実質 16） |
| learning rate | 7e-6 |
| warmup_ratio | 0.1 |
| max_grad_norm | 0.3 |
| weight_decay | 0.05 |
| epoch | 4.5 |
  
低学習率 + 長めのウォームアップ + LoRA で過学習を抑えつつ安定した収束を狙っています。  
  
---  
  
## 🧪 推論サンプル（generate.py）  
  
学習済みモデルを使用した推論結果は以下の通りです。 

### demo video   
https://drive.google.com/file/d/1-6tn6dS3C8FD7ldOvTYeuYHn9GdrXLrO/view?usp=sharing  
  
---  
  
### Q：メゾンマルジェラの香水を教えてください。  
  
🧴 生成された回答：  
  
Maison Margielaの香りでは、ローザがおすすめです。【トップノート】オレンジ, レモン【ミドル&ラスト】「マルセル・ワンダース」は「女性たちのためのフェミニンな香り」。フレッシュで軽やかな印象を与えるレモンやベルガモットが組み合わさることで温かみのあるウッディムスクに変わります。「オードパルファン ド ヴィサージュ オーデトワレ(オスマンサス)とは?」のような香水です。  
  
---  

### Q：男性向けの香水を教えてください。  
  
🧴 生成された回答：  
  
メンズ向けでは、ヴァレンタイン コレクションがおすすめです。【トップノート】オレンジ, オレンジブロッサム【ミドルエレガンス】、ラストはサンタル(ホワイトムスク), ムスキーです。, 【ベースオードパルファン】, シトラス  
  
---  
  
### Q：春に相応しい香水を教えてください。  
  
🧴 生成された回答：  
  
春でしたら、ザ バスクがおすすめです。【トップノート】マンダリン, オレンジブロッサム、【ミドルスパイス】、ペッパー,【ラスト(基剤)】アンバーグリス  
