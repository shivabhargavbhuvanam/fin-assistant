# Financial Assistant

> **AI-powered personal financial analysis platform with intelligent spending insights, ML anomaly detection, and conversational coaching**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2-61dafb?style=flat-square&logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-3178c6?style=flat-square&logo=typescript)](https://www.typescriptlang.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python)](https://www.python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-10a37f?style=flat-square&logo=openai)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

## Overview

**Fin Assistant** is an intelligent financial analysis platform that combines machine learning, natural language processing, and conversational AI to help users:

- **Automatically categorize** financial transactions using hybrid AI + rule-based systems
- **Detect unusual spending** with Isolation Forest machine learning algorithm
- **Identify subscriptions** and forgotten recurring charges (gray charges)
- **Chat with an AI coach** that has real-time access to your financial data
- **Receive personalized insights** and actionable savings recommendations
- **Set and achieve** savings goals with AI-powered forecasting

### Why Fin Assistant?

Traditional budgeting apps lack intelligence and personalization. Fin Assistant goes beyond simple expense tracking by:

- Training ML models **on your individual data** for personalized anomaly detection  
- Using **GPT-4o with function calling** for natural language financial queries  
- Providing **explainable AI** - every insight includes reasoning and confidence scores  
- Offering **hybrid categorization** - 70% rule-based speed + 30% AI accuracy  
- Detecting **gray charges** - small forgotten subscriptions costing $100s annually  

---

##  Features

###  Smart Categorization
- **Hybrid Approach**: Rule-based patterns (~70% coverage) + GPT-4o for complex transactions
- **12 Default Categories**: Housing, Groceries, Dining, Transportation, Healthcare, Shopping, Entertainment, Subscriptions, Utilities, Travel, Income, Other

###  Anomaly Detection
- **ML-Powered**: Isolation Forest algorithm trained on YOUR data (personalized)

###  Recurring Charge Detection
- **Subscription Tracking**: Automatically identifies monthly and weekly patterns
- **Gray Charge Alerts**: Flags small, unknown recurring charges you might have forgotten

### AI-Powered Insights
- **Personalized Advice**: GPT-4o generates actionable recommendations
- **Pattern Recognition**: Detects merchant habits, weekend splurges, payday impulse spending

###  Conversational AI Coach
- **Natural Language Queries**: Ask questions like "What are my biggest expenses?"
- **Function Calling**: 7 tools for real-time data access (spending summary, anomalies, recurring charges, insights, category details, transaction search, month comparison)
- **Streaming Responses**: Real-time token-by-token generation
  
### 🥠 Fortune Cookie Insights
- **Fun Financial Wisdom**: Get personalized fortune cookie messages based on your spending patterns
- **AI-Generated**: Uses GPT-4o to create witty, motivational financial advice
- **Contextual**: Messages reflect your actual financial behavior and insights

---

## 🏗️ High Level Architecture
<img width="2115" height="2233" alt="smart_financial_coach_–_high_level_architecture_(deployed_on_gcp)" src="https://github.com/user-attachments/assets/5d634e9a-a49e-4a3b-87cf-9787e809ef30" />



Fin Assistant uses a **microservice architecture** with distinct layers for frontend, API gateway, application logic, data storage, and AI/ML services.

```mermaid
flowchart TD
    User[User] --> Clerk[Clerk Authentication]
    Clerk --> Welcome[Welcome Page]
    Welcome --> Upload[Upload CSV Option]
    
    Upload --> CSV[CSV Processor Service]
    CSV --> DB[(SQLite Database<br/>Stores CSV Data)]
    
    DB --> Dashboard[Main Dashboard]
    
    Dashboard --> Cat[Categorizer]
    Dashboard --> Anom[Anomaly Detector]
    Dashboard --> Rec[Recurring Detector]
    Dashboard --> Insight[Insight Generator]
    Dashboard --> Chat[Chat Service]
    
    Cat --> DB
    Anom --> DB
    Rec --> DB
    Insight --> DB
    Chat --> DB
    
    Cat --> OpenAI[OpenAI GPT-4o]
    Insight --> OpenAI
    Chat --> OpenAI
    
    Anom --> ML[ML Model Training<br/>Isolation Forest]
    
    style Clerk fill:#6c47ff
    style Welcome fill:#e1f5ff
    style Upload fill:#fff4e6
    style CSV fill:#ffecb3
    style DB fill:#f3e5f5
    style Dashboard fill:#c8e6c9
    style ML fill:#ffccbc
    style OpenAI fill:#b2dfdb
```

### Architecture Layers Explained

#### 1. **Client Layer**
- Users access via modern web browsers (Chrome, Firefox, Safari)
- Responsive design works on desktop, tablet, mobile

#### 2. **Presentation Layer (React Frontend)**
- **Technology**: React 18.2 + TypeScript 5.3 + Tailwind CSS + Vite
- **Components**: Upload, Dashboard, Chat, Insights, Anomalies, Recurring Charges, Goals
- **State Management**: React hooks (useState, useEffect)
- **Styling**: Utility-first Tailwind CSS with warm neutral palette

#### 3. **API Gateway Layer**
- **Nginx**: Reverse proxy for frontend serving and API routing
- **Clerk Auth**: JWT token validation on protected routes
- **Rate Limiter**: Sliding window algorithm (30 chat/min, 100 API/min)

#### 4. **Application Layer (FastAPI Backend)**
- **FastAPI**: High-performance async Python API framework
- **7 Service Modules**: Modular, reusable components for each feature
- **15+ API Endpoints**: RESTful design with automatic OpenAPI docs

#### 5. **Data Layer (SQLite)**
- **8 Tables**: Sessions, Transactions, Categories, Anomalies, Recurring Charges, Insights, Deltas, Conversations
- **SQLAlchemy ORM**: Type-safe database operations
- **Foreign Keys**: Maintains referential integrity

#### 6. **AI/ML Layer**
- **OpenAI GPT-4o**: Transaction categorization, insight generation, chat
- **Isolation Forest**: ML anomaly detection (≥50 transactions)
- **Statistical Analysis**: Z-score fallback (<50 transactions)

#### 7. **Cloud Infrastructure**
- **GCP Compute Engine**: e2-medium VM (2 vCPU, 4GB RAM)
- **Ubuntu 22.04 LTS**: Stable Linux distribution
- **Docker**: Containerized deployment

---

## 🛠️ Technology Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Core language |
| FastAPI | 0.109 | REST API framework |
| SQLAlchemy | 2.0 | Database ORM |
| Pydantic | 2.5 | Data validation |
| Pandas | 2.2 | CSV processing |
| scikit-learn | 1.4+ | ML (Isolation Forest) |
| OpenAI | 1.12+ | GPT-4o integration |
| Uvicorn | 0.27 | ASGI server |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2 | UI framework |
| TypeScript | 5.3 | Type safety |
| Vite | 5.0 | Build tool |
| Tailwind CSS | 3.4 | Styling |
| Recharts | 2.12 | Charts/visualizations |

### Infrastructure
| Technology | Version | Purpose |
|------------|---------|---------|
| Docker | Latest | Containerization |
| Docker Compose | Latest | Service orchestration |
| Nginx | 1.18+ | Reverse proxy |
| SQLite | 3 | Database |
| GCP Compute Engine | - | Cloud hosting |

### Authentication & Security
| Technology | Version | Purpose |
|------------|---------|---------|
| Clerk | Latest | User authentication |
| PyJWT | 2.8+ | JWT token handling |
| python-dotenv | 1.0 | Environment variables |

---

## Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18 or higher
- **npm**: 9 or higher
- **OpenAI API Key**: Get one at [platform.openai.com](https://platform.openai.com)
- **Clerk Account**: Sign up at [clerk.com](https://clerk.com)

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Fin Assistant.git
cd Fin Assistant
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
# Navigate to frontend
cd ../frontend

# Install dependencies
npm install
```

### 4. Environment Configuration

Create `.env` file in project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o

# Database
DATABASE_URL=sqlite:///./financial_coach.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Clerk Authentication
VITE_CLERK_PUBLISHABLE_KEY=pk_test_your-clerk-key
```

### 5. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
# Or: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 6. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## Installation

### Development Installation

```bash
# Clone repository
git clone https://github.com/your-username/Fin Assistant.git
cd FinAssistant

# Setup backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup frontend
cd ../frontend
npm install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run development servers
# Terminal 1:
cd backend && python main.py
# Terminal 2:
cd frontend && npm run dev
```

### Production Installation (Docker)

```bash
# Build containers
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```
---

## Deployment

### Deploy to Google Cloud Platform

#### 1. Create VM Instance

```bash
gcloud compute instances create Fin-Assistant \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --tags=http-server,https-server
```

#### 2. SSH into VM

```bash
gcloud compute ssh Fin-assistant --zone=us-central1-a
```

#### 3. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install -y python3.10 python3.10-venv python3-pip

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Nginx
sudo apt install -y nginx
```

#### 4. Deploy Application

```bash
# Clone repository
cd /var/www
sudo mkdir -p Fin-Assistant
sudo chown $USER:$USER Fin-Assistant
git clone
cd fin-asistant

# Setup backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup frontend
cd ../frontend
npm install
npm run build

# Configure .env
cd ..
cp .env.example .env
nano .env  # Add your API keys
```

#### 5. Configure systemd Service

Create `/etc/systemd/system/fin-asistant-backend.service`:

```ini
[Unit]
Description=Finance assistant Backend
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/var/www/fin-asistant/backend
Environment="PATH=/var/www/fin-asistant/venv/bin"
ExecStart=/var/www/fin-asistant/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable fin-asistant
sudo systemctl start fin-asistant
```

#### 6. Configure Nginx

Create `/etc/nginx/sites-available/Fin-Assistant`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        root /var/www/fin-asistant/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/fin-asistant /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 7. Configure Firewall

```bash
gcloud compute firewall-rules create allow-http --allow tcp:80
gcloud compute firewall-rules create allow-https --allow tcp:443
gcloud compute firewall-rules create allow-backend --allow tcp:8000
```

#### 8. Test Deployment

```bash
curl http://VM_EXTERNAL_IP:8000/health
```

---

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style

**Python:**
- Follow PEP 8 guidelines
- Use type hints
- Document functions with docstrings

**TypeScript/React:**
- Follow ESLint rules
- Use functional components with hooks
- Add JSDoc comments for complex functions
- Run linter: `npm run lint`

### Testing

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
npm test
```
---

###  Sample Data & Test Cases Pre-loaded test datasets in the `/test_csvs` folder. Feel free to quick start analyzing without connecting your bank accounts :p

## 📜 License

MIT License – Feel free to fork, extend, and build upon TravelBuddy.

---
## 🤖 AI Usage Disclosure

AI tools such as OpenAI GPT, LangChain, and GitHub Copilot were used for implementing multi-agent travel assistance, generating responses, and supporting development tasks like code scaffolding and prompt tuning.


