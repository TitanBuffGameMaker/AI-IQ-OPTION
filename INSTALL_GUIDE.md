# คู่มือติดตั้งและใช้งาน IQ Option AI Trading
## ระบบ AI เทรด IQ Option แบบเรียนรู้อัตโนมัติ

---

## สารบัญ

1. [ความต้องการของระบบ](#1-ความต้องการของระบบ)
2. [ติดตั้ง Python และ Prerequisites](#2-ติดตั้ง-python-และ-prerequisites)
3. [ดาวน์โหลดโปรแกรม](#3-ดาวน์โหลดโปรแกรม)
4. [ติดตั้ง Dependencies](#4-ติดตั้ง-dependencies)
5. [ตั้งค่า .env](#5-ตั้งค่า-env)
6. [ตั้งค่า IQ Option](#6-ตั้งค่า-iq-option)
7. [ทดสอบ Backtest ก่อนใช้งานจริง](#7-ทดสอบ-backtest-ก่อนใช้งานจริง)
8. [เปิดโปรแกรม Web Dashboard](#8-เปิดโปรแกรม-web-dashboard)
9. [วิธีใช้งาน Dashboard](#9-วิธีใช้งาน-dashboard)
10. [ทำความเข้าใจ Brain Age](#10-ทำความเข้าใจ-brain-age)
11. [เมื่อ AI พร้อม — เปลี่ยนเป็นเงินจริง](#11-เมื่อ-ai-พร้อม--เปลี่ยนเป็นเงินจริง)
12. [แก้ปัญหาที่พบบ่อย](#12-แก้ปัญหาที่พบบ่อย)

---

## 1. ความต้องการของระบบ

| รายการ | ขั้นต่ำ | แนะนำ |
|--------|---------|-------|
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.11 หรือ 3.12 |
| RAM | 4 GB | 8 GB ขึ้นไป |
| CPU | 4 core | 8 core (GPU ไม่จำเป็น) |
| พื้นที่ดิสก์ | 3 GB | 5 GB |
| อินเทอร์เน็ต | จำเป็น | ความเร็วสูง (AI ค้นข่าวทุก 30 นาที) |
| บัญชี IQ Option | จำเป็น | เปิดบัญชีที่ iqoption.com |

> **GPU (NVIDIA CUDA):** ไม่จำเป็น แต่ถ้ามีจะเร็วขึ้น 2-3 เท่า PyTorch จะตรวจหา GPU อัตโนมัติ

---

## 2. ติดตั้ง Python และ Prerequisites

### Windows

1. ดาวน์โหลด Python 3.11 จาก https://python.org/downloads
2. ติดตั้ง — **ติ๊ก "Add Python to PATH"** ก่อนกด Install
3. เปิด Command Prompt (Win+R → พิมพ์ `cmd`) แล้วตรวจสอบ:
   ```
   python --version
   ```
   ต้องขึ้น `Python 3.11.x`

4. ติดตั้ง Tesseract OCR (สำหรับตรวจสอบหน้าจอ IQ Option):
   - ดาวน์โหลด: https://github.com/UB-Mannheim/tesseract/wiki
   - ติดตั้งไปที่ `C:\Program Files\Tesseract-OCR\`
   - เพิ่ม PATH: Control Panel → System → Advanced → Environment Variables → Path → เพิ่ม `C:\Program Files\Tesseract-OCR`

5. ติดตั้ง Git:
   - ดาวน์โหลด: https://git-scm.com/download/win

### Ubuntu / Debian Linux

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git
sudo apt install -y tesseract-ocr tesseract-ocr-tha
sudo apt install -y libgl1-mesa-glx  # สำหรับ opencv
```

### macOS

```bash
# ต้องมี Homebrew ก่อน (https://brew.sh)
brew install python@3.11 git tesseract
```

---

## 3. ดาวน์โหลดโปรแกรม

```bash
# Clone โปรเจกต์
git clone https://github.com/titanbuffgamemaker/titan-fall-the-world.git
cd titan-fall-the-world

# หรือถ้ามีโฟลเดอร์อยู่แล้ว
cd /path/to/Titan-Fall-The-World
```

---

## 4. ติดตั้ง Dependencies

### สร้าง Virtual Environment (แนะนำมาก — แยกสภาพแวดล้อม)

```bash
# สร้าง virtual env
python -m venv venv

# เปิดใช้งาน (Windows)
venv\Scripts\activate

# เปิดใช้งาน (Linux/macOS)
source venv/bin/activate
```

> หลังจากนี้ทุกคำสั่งต้องอยู่ใน virtual env (จะเห็น `(venv)` นำหน้า)

### ติดตั้ง packages ทั้งหมด

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> การติดตั้งใช้เวลา **5-15 นาที** เพราะต้องดาวน์โหลด PyTorch (~2GB)

### ตรวจสอบว่าติดตั้งสำเร็จ

```bash
python -c "import torch; import fastapi; import iqoptionapi; print('OK')"
```

ต้องขึ้น `OK` — ถ้า error ดูหัวข้อ [แก้ปัญหา](#12-แก้ปัญหาที่พบบ่อย)

---

## 5. ตั้งค่า .env

### คัดลอกไฟล์ตัวอย่าง

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

### แก้ไขไฟล์ .env

เปิดไฟล์ `.env` ด้วย Notepad (Windows) หรือ nano (Linux):

```bash
# Linux/macOS
nano .env

# Windows — เปิด Notepad แล้วลากไฟล์ .env เข้าไป
```

**ตัวอย่างการตั้งค่า:**

```env
# ข้อมูลล็อกอิน IQ Option
IQ_EMAIL=your-email@gmail.com
IQ_PASSWORD=your-iq-option-password

# ประเภทบัญชี — เริ่มต้นใช้ PRACTICE ก่อนเสมอ!
IQ_ACCOUNT_TYPE=PRACTICE

# สินทรัพย์หลัก (AI จะเทรด EUR/USD OTC เป็นหลัก)
ASSET=EURUSD

# จำนวนเงินต่อไม้ (USD) — เริ่มต้น $1
TRADE_AMOUNT=1.0

# ระยะเวลาออเดอร์ (นาที) — 1 นาที ตามที่ตั้งใน IQ Option
TRADE_DURATION=1

# แท่งเทียน 30 วินาที (ใช้ดูราคา real-time)
CANDLE_TIMEFRAME=30

# โฟลเดอร์เก็บสมอง AI
MODEL_DIR=./knowledge

# ขาดทุนสูงสุดต่อวัน (USD) — AI จะหยุดเองเมื่อถึง limit
DAILY_LOSS_LIMIT=20.0
```

> **สำคัญ:** ไฟล์ `.env` มีรหัสผ่าน อย่า commit ขึ้น Git เด็ดขาด

---

## 6. ตั้งค่า IQ Option

ก่อนเปิด AI **ต้องตั้งค่า IQ Option ให้ถูกต้อง** เพราะ AI จะตรวจสอบอัตโนมัติ:

### เปิด IQ Option แล้วทำตามขั้นตอน:

**ขั้นตอนที่ 1 — เลือกสินทรัพย์ OTC ทั้ง 4 ตัว**

กราฟทั้ง 4 ต้องเป็น:
- EUR/USD (OTC)
- GBP/USD (OTC)
- AUD/USD (OTC)
- GBP/JPY (OTC)

> OTC = Over The Counter คือหุ้นที่ IQ Option สร้างขึ้นเอง ซื้อขายได้ตลอด 24 ชั่วโมง

**ขั้นตอนที่ 2 — ตั้ง Timeframe เป็น 30 นาที**

คลิกที่กราฟ → เลือก timeframe → **30M**

**ขั้นตอนที่ 3 — ตั้ง Duration เป็น 1 นาที**

ที่กล่องเลือก duration ด้านล่าง → เลือก **1M**

**ขั้นตอนที่ 4 — ใช้บัญชี Practice**

มุมบนขวา → เลือก **Practice account** (เงินปลอม $10,000)

---

## 7. ทดสอบ Backtest ก่อนใช้งานจริง

Backtest คือการฝึก AI บนข้อมูลเก่า (ไม่เสียเงินจริง) เพื่อให้ AI มีความรู้พื้นฐานก่อน

### ดาวน์โหลดข้อมูล Historical (ตัวอย่าง CSV)

ดาวน์โหลด EURUSD OHLCV จาก https://www.histdata.com/download-free-forex-historical-data/

หรือสร้างข้อมูลจำลองเพื่อทดสอบ:

```bash
python -c "
import pandas as pd, numpy as np
np.random.seed(42)
n = 5000
price = 1.08 + np.cumsum(np.random.randn(n) * 0.0001)
df = pd.DataFrame({
    'time': pd.date_range('2024-01-01', periods=n, freq='1min'),
    'open':  price,
    'high':  price + np.abs(np.random.randn(n) * 0.0002),
    'low':   price - np.abs(np.random.randn(n) * 0.0002),
    'close': price + np.random.randn(n) * 0.00005,
    'volume': np.random.randint(100, 1000, n)
})
df.to_csv('test_data.csv', index=False)
print('สร้าง test_data.csv สำเร็จ', len(df), 'แท่ง')
"
```

### รัน Backtest

```bash
python -m trading_ai.backtest --csv test_data.csv --episodes 20
```

ผลลัพธ์ที่ควรเห็น:
```
Episode  1/20 | Trades:  45 | Win: 51.1% | PnL: -$1.23 | Nodes: 12
Episode  5/20 | Trades: 223 | Win: 53.4% | PnL: +$2.18 | Nodes: 31
Episode 10/20 | Trades: 451 | Win: 55.8% | PnL: +$8.45 | Nodes: 58
Episode 15/20 | Trades: 678 | Win: 57.2% | PnL: +$14.3 | Nodes: 76
Episode 20/20 | Trades: 902 | Win: 58.9% | PnL: +$22.1 | Nodes: 94
Brain saved → ./knowledge/brain_best.pt
```

> ถ้า Win Rate ขึ้นไปเรื่อยๆ แสดงว่า AI กำลังเรียนรู้ได้ถูกต้อง

---

## 8. เปิดโปรแกรม Web Dashboard

```bash
# ต้องอยู่ใน virtual env และ root ของโปรเจกต์
python -m trading_ai.web
```

เปิด browser แล้วไปที่: **http://localhost:8000**

ถ้าสำเร็จจะเห็น:
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     WebSocket: /ws
[IQ] กำลังเชื่อมต่อ IQ Option...
[IQ] เชื่อมต่อสำเร็จ — PRACTICE account
```

> **หมายเหตุ:** IQ Option ต้องเปิดอยู่ก่อนรัน AI (ต้องล็อกอินไว้แล้ว)

---

## 9. วิธีใช้งาน Dashboard

### หน้าหลัก (4 Chart Grid)

```
┌─────────────────────────────────────────────────────┐
│  IQ Option AI  🟡 กำลังเชื่อมต่อ  💰 $10,000  🧠 94 nodes │
│                                  TF: 30M | DUR: 1M  │
│                               [▶ START AI]          │
├───────────────┬───────────────┬─────────────────────┤
│ EUR/USD (OTC) │ GBP/USD (OTC) │    🧠 Brain         │
│  1.08542 ↑   │  1.27103 →   │    📋 History       │
│  [กราฟ]      │  [กราฟ]      │    ⚙️ Settings      │
│ ▲BUY  ▼SELL  │ ▲BUY  ▼SELL  │                     │
├───────────────┼───────────────┤                     │
│ AUD/USD (OTC) │ GBP/JPY (OTC)│                     │
│  0.65218 ↓   │  193.421 ↑   │                     │
│  [กราฟ]      │  [กราฟ]      │                     │
│ ▲BUY  ▼SELL  │ ▲BUY  ▼SELL  │                     │
└───────────────┴───────────────┴─────────────────────┘
```

### ขั้นตอนการใช้งาน

#### ขั้นตอนที่ 1 — ตรวจสอบการเชื่อมต่อ

แถบด้านบน (Check Banner) จะแสดงสถานะ:
- ✅ OTC Assets — ชื่อหุ้น 4 ตัวถูกต้อง
- ✅ Timeframe 30M — แท่งเทียนถูกต้อง
- ✅ Duration 1M — ระยะเวลาออเดอร์ถูกต้อง

ถ้า ❌ ให้กลับไปตั้งค่า IQ Option ตาม [ขั้นตอนที่ 6](#6-ตั้งค่า-iq-option)

#### ขั้นตอนที่ 2 — ตั้งค่าในแท็บ Settings

คลิก **⚙️ Settings** ด้านขวา:

| การตั้งค่า | ค่าที่แนะนำ | คำอธิบาย |
|------------|-------------|----------|
| Timeframe | 30 นาที | ขนาดแท่งเทียนที่ AI วิเคราะห์ |
| Duration | 1 นาที | ระยะเวลาออเดอร์แต่ละไม้ |
| Amount | $1 | เงินต่อไม้ — เพิ่มทีหลังเมื่อ AI พิสูจน์ตัวเอง |
| Account | PRACTICE | **ใช้ PRACTICE ก่อนเสมอ** |
| Min. Confidence | 60% | AI จะเทรดเมื่อมั่นใจ 60%+ |

กด **💾 บันทึกการตั้งค่า**

#### ขั้นตอนที่ 3 — เริ่ม AI

กดปุ่ม **▶ START AI** มุมบนขวา

ปุ่มจะเปลี่ยนเป็น **⏹ STOP AI** — AI เริ่มทำงานแล้ว

สิ่งที่จะเห็น:
- กราฟเคลื่อนไหวตาม real-time ราคา OTC
- แถบสัญญาณบนกราฟจะแสดง `▲ BUY` หรือ `▼ SELL`
- AI confidence bar แสดงความมั่นใจ 0-100%
- Status bar ล่างสุดแสดง trades / win rate / P&L

#### ขั้นตอนที่ 4 — ดูสมอง AI (แท็บ Brain)

คลิก **🧠 Brain** ด้านขวา เพื่อดู:

- **Brain Age** — อายุสมอง AI เทียบกับมนุษย์
- **Knowledge Nodes** — จำนวน "ราก" ความรู้
- **Branches** — เส้นเชื่อมระหว่างความรู้
- **Win Rate** — อัตราชนะล่าสุด
- **Breakdown** — คะแนนแต่ละด้าน (ความรู้, ผลงาน, ประสบการณ์ ฯลฯ)

#### ขั้นตอนที่ 5 — เทรดด้วยตัวเอง (Optional)

ถ้าต้องการเทรดเอง ไม่รอ AI:
- กดปุ่ม **▲ BUY ขึ้น** เมื่อคิดว่าราคาจะขึ้น
- กดปุ่ม **▼ SELL ลง** เมื่อคิดว่าราคาจะลง

AI จะเรียนรู้จากผลการเทรดของคุณด้วย

---

## 10. ทำความเข้าใจ Brain Age

Brain Age คือการวัดความฉลาดของ AI เทียบกับอายุมนุษย์:

| อายุ | ระยะ | คำอธิบาย | ทำได้ |
|------|------|----------|-------|
| 0-3 ปี | 👶 ทารก | เพิ่งเกิด | เรียนรู้พื้นฐาน |
| 3-10 ปี | 🧒 เด็กเล็ก | เริ่มเข้าใจ | จดจำ pattern เบื้องต้น |
| 10-18 ปี | 🧑 วัยรุ่น | กำลังพัฒนา | วิเคราะห์ indicator ได้ |
| 18-30 ปี | 👨 หนุ่มสาว | มีประสบการณ์ | อ่านตลาดได้ดี |
| 30-45 ปี | 👨‍💼 มืออาชีพ | เชี่ยวชาญ | ชนะได้สม่ำเสมอ |
| 45-60 ปี | 🧓 ผู้เชี่ยวชาญ | ชำนาญมาก | win rate 60%+ |
| 60-80 ปี | 👴 ปรมาจารย์ | ระดับสูง | ปรับตัวกับตลาดได้ |
| 80+ ปี | 🏆 ตำนาน | สูงสุด | AI ระดับ elite |

**คะแนน Brain Age (0-100):**
- 🌳 ความรู้ (25 คะแนน) — จำนวน knowledge nodes
- 🎯 ผลงาน (30 คะแนน) — win rate
- 📚 ประสบการณ์ (25 คะแนน) — จำนวนเทรด
- 🧬 การเรียนรู้ (10 คะแนน) — PPO updates
- 🎲 ความมั่นใจ (10 คะแนน) — calibrated confidence

> AI ใหม่ที่ยังไม่เคยเทรดจะอยู่ที่ ~0 ปี หลังเทรด 100-200 ไม้จะขึ้นเป็น 3-5 ปี

---

## 11. เมื่อ AI พร้อม — เปลี่ยนเป็นเงินจริง

**เงื่อนไขก่อนเปลี่ยนเป็น REAL account:**

1. ✅ Brain Age ≥ 18 ปี (หนุ่มสาว ขึ้นไป)
2. ✅ Win Rate ≥ 60% ในช่วง 100 ไม้ล่าสุด
3. ✅ เทรด PRACTICE แล้ว ≥ 500 ไม้
4. ✅ ผ่าน 7 วัน (สมองเรียนรู้ news + economic calendar)
5. ✅ ตั้ง DAILY_LOSS_LIMIT ใน .env แล้ว

**ขั้นตอน:**
1. ไปแท็บ **⚙️ Settings**
2. เปลี่ยน Account จาก `PRACTICE` เป็น `REAL`
3. ตั้ง Amount ให้เหมาะสม (ไม่เกิน 1-2% ของ balance)
4. กด **💾 บันทึก**

> ⚠️ เงินจริงมีความเสี่ยง AI ฉลาดแค่ไหนก็ยังขาดทุนได้ ควรเริ่มด้วย $1-5 ต่อไม้

---

## 12. แก้ปัญหาที่พบบ่อย

### ❌ `ModuleNotFoundError: No module named 'iqoptionapi'`

```bash
pip install iqoptionapi websocket-client
```

### ❌ `Connection failed` หรือ `Login error`

- ตรวจสอบ email/password ใน `.env`
- ลองล็อกอิน IQ Option ผ่านเว็บ browser ก่อน (อาจต้อง verify email)
- ถ้า IQ Option ขอ 2FA ให้ approve ในแอปมือถือก่อน

### ❌ `torch.cuda` errors

```bash
# ถ้าไม่มี GPU — ติดตั้ง CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### ❌ กราฟไม่แสดง / ขึ้น `Fallback chart`

- เกิดจาก CDN `lightweight-charts` โหลดช้า
- ลองรีเฟรช browser (Ctrl+F5)
- ถ้า offline จะใช้ canvas fallback อัตโนมัติ (ทำงานได้เหมือนกัน)

### ❌ `OSError: [Errno 98] Address already in use`

```bash
# มี server รันอยู่แล้ว — หา process และปิด
# Linux/macOS
lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### ❌ ราคาไม่อัพเดท (กราฟนิ่ง)

- ตรวจสอบว่า IQ Option เปิดอยู่และล็อกอินแล้ว
- ดู terminal — ถ้าขึ้น `[IQ] เชื่อมต่อสำเร็จ` แต่กราฟนิ่ง ให้รอ ~30 วินาที

### ❌ Brain Age ไม่ขยับ

Brain Age คำนวณจากผลเทรดจริง — ต้องให้ AI เทรดก่อน (อย่างน้อย 20-30 ไม้) ถึงจะเห็นความเปลี่ยนแปลง

### ❌ `tesseract: command not found`

```bash
# Ubuntu
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

---

## สรุปขั้นตอนแบบสั้น (Quick Start)

```bash
# 1. Clone + เข้าโฟลเดอร์
git clone https://github.com/titanbuffgamemaker/titan-fall-the-world.git
cd titan-fall-the-world

# 2. สร้าง virtual env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. ติดตั้ง
pip install -r requirements.txt

# 4. ตั้งค่า
cp .env.example .env
# แก้ไข .env ใส่ email/password IQ Option

# 5. Backtest ก่อน (ฝึก AI บนข้อมูลเก่า)
python -m trading_ai.backtest --csv your_data.csv --episodes 20

# 6. เปิด Dashboard
python -m trading_ai.web

# 7. เปิด browser
# http://localhost:8000
```

---

## โครงสร้างไฟล์สำคัญ

```
Titan-Fall-The-World/
├── .env                    ← ข้อมูลล็อกอิน (สร้างเอง จาก .env.example)
├── .env.example            ← ตัวอย่าง config
├── requirements.txt        ← รายการ packages
├── knowledge/              ← สมอง AI (สร้างอัตโนมัติ)
│   ├── brain_best.pt       ← Neural network weights
│   └── brain_graph.json    ← Knowledge graph (รากไม้)
└── trading_ai/
    ├── web/                ← Web Dashboard
    │   ├── server.py       ← FastAPI + WebSocket server
    │   └── templates/      ← HTML หน้าเว็บ
    ├── brain/              ← ระบบสมอง AI
    │   ├── brain_age.py    ← คำนวณอายุสมอง
    │   ├── brain_core.py   ← ควบคุมสมองทั้งหมด
    │   └── knowledge_graph.py ← รากไม้ความรู้
    ├── models/ppo_agent.py ← PPO RL Agent (Neural Network)
    ├── core/trading_env.py ← Environment การเทรด
    ├── indicators/         ← Indicators 18 ตัว
    └── backtest.py         ← ทดสอบบนข้อมูลเก่า
```

---

*คู่มือนี้ครอบคลุมการใช้งานทั้งหมด หากพบปัญหาอื่น ดู error message ใน terminal และตรวจสอบว่า `.env` ถูกต้อง*
