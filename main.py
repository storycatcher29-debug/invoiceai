"""
InvoiceAI SaaS v4.2 — Production Ready
Groq AI (ücretsiz) + SQLite/PostgreSQL + JWT Auth + Stripe + Rate Limiting
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.hash import bcrypt
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os, json, io, re, logging
import pdfplumber
import stripe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="InvoiceAI SaaS v4.1", version="4.1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Ortam Değişkenleri ────────────────────────────────────
SECRET_KEY       = os.environ.get("SECRET_KEY", "dev-secret-change-me-12345")
ALGORITHM        = "HS256"
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
STRIPE_SECRET    = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID  = os.environ.get("STRIPE_PRICE_ID", "")
STRIPE_WH_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
FREE_LIMIT       = 5

# ── Veritabanı ────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./invoiceai.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

groq_client = Groq(api_key=GROQ_API_KEY)
stripe.api_key = STRIPE_SECRET

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── Veritabanı Modelleri ──────────────────────────────────
class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True)
    email    = Column(String, unique=True, index=True)
    password = Column(String)
    plan     = Column(String, default="free")
    usage    = Column(Integer, default=0)

class Invoice(Base):
    __tablename__ = "invoices"
    id             = Column(Integer, primary_key=True)
    user_id        = Column(Integer, index=True)
    vendor         = Column(String)
    amount         = Column(Float)
    currency       = Column(String)
    date           = Column(String)
    due_date       = Column(String)
    vat_amount     = Column(Float)
    vat_rate       = Column(Float)
    invoice_number = Column(String)
    category       = Column(String)
    summary_tr     = Column(Text)
    summary_en     = Column(Text)
    raw_result     = Column(Text)
    created_at     = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

Base.metadata.create_all(bind=engine)

# ── JWT Yardımcıları ──────────────────────────────────────
def create_token(user_id: int) -> str:
    payload = {"user_id": user_id, "exp": datetime.now(timezone.utc) + timedelta(days=7)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def require_user(authorization: str = Header(None), db: Session = Depends(get_db)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Yetkilendirme gerekli")
    token = authorization.removeprefix("Bearer ")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
    except JWTError:
        raise HTTPException(status_code=401, detail="Geçersiz veya süresi dolmuş token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Kullanıcı bulunamadı")
    return user

# ── Pydantic Modelleri ────────────────────────────────────
class AuthIn(BaseModel):
    email: str
    password: str

class EmailIn(BaseModel):
    subject: str
    body: str

# ── AI Analiz ─────────────────────────────────────────────
EXTRACT_PROMPT = """
Extract invoice data from the text below. Return ONLY valid JSON, no explanation.
TEXT:
{text}
JSON format:
{{
  "vendor": null,
  "amount": null,
  "currency": null,
  "date": null,
  "due_date": null,
  "vat_amount": null,
  "vat_rate": null,
  "invoice_number": null,
  "category": null,
  "summary_tr": "",
  "summary_en": ""
}}
"""

def analyze_text(text: str) -> tuple[dict, str]:
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text[:6000])}],
            temperature=0,
        )
        raw = res.choices[0].message.content or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("AI JSON formatında yanıt üretmedi")
        return json.loads(match.group(0)), raw
    except Exception as e:
        logger.error(f"Groq Analiz Hatası: {e}")
        raise HTTPException(500, f"AI Analiz hatası: {str(e)}")

# ── Endpoint'ler ──────────────────────────────────────────
@app.post("/register")
@limiter.limit("5/minute")
def register(request: Request, data: AuthIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "Bu email zaten kayıtlı")
    user = User(email=data.email, password=bcrypt.hash(data.password[:72]))
    db.add(user)
    db.commit()
    return {"success": True}

@app.post("/login")
@limiter.limit("10/minute")
def login(request: Request, data: AuthIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not bcrypt.verify(data.password[:72], user.password):
        raise HTTPException(401, "E-posta veya şifre hatalı")
    return {"token": create_token(user.id)}

@app.get("/me")
def me(user: User = Depends(require_user)):
    remaining = None if user.plan == "pro" else max(0, FREE_LIMIT - user.usage)
    return {"email": user.email, "plan": user.plan, "usage": user.usage, "remaining": remaining}

@app.get("/invoices")
def list_invoices(user: User = Depends(require_user), db: Session = Depends(get_db)):
    invoices = db.query(Invoice).filter(Invoice.user_id == user.id).order_by(Invoice.id.desc()).all()
    return invoices

@app.post("/analyze-email")
@limiter.limit("20/minute")
def analyze_email(request: Request, data: EmailIn,
                  user: User = Depends(require_user), db: Session = Depends(get_db)):
    if user.plan != "pro" and user.usage >= FREE_LIMIT:
        raise HTTPException(402, "Ücretsiz limitiniz doldu. Lütfen Pro'ya geçin.")
    content = f"Subject: {data.subject}\n\n{data.body}"
    invoice_data, raw = analyze_text(content)
    new_inv = Invoice(user_id=user.id, raw_result=raw,
                      **{k: v for k, v in invoice_data.items() if hasattr(Invoice, k)})
    user.usage += 1
    db.add(new_inv)
    db.commit()
    db.refresh(new_inv)
    return {"success": True, "invoice": new_inv}

@app.post("/analyze-pdf")
@limiter.limit("20/minute")
async def analyze_pdf(request: Request, file: UploadFile = File(...),
                      user: User = Depends(require_user), db: Session = Depends(get_db)):
    if user.plan != "pro" and user.usage >= FREE_LIMIT:
        raise HTTPException(402, "Ücretsiz limitiniz doldu. Lütfen Pro'ya geçin.")
    try:
        contents = await file.read()
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if not text.strip():
            raise HTTPException(422, "PDF metni okunamadı. Dijital PDF yükleyin.")
        invoice_data, raw = analyze_text(text)
        new_inv = Invoice(user_id=user.id, raw_result=raw,
                          **{k: v for k, v in invoice_data.items() if hasattr(Invoice, k)})
        user.usage += 1
        db.add(new_inv)
        db.commit()
        db.refresh(new_inv)
        return {"success": True, "invoice": new_inv}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"İşlem hatası: {str(e)}")

@app.post("/checkout")
def checkout(user: User = Depends(require_user)):
    if not STRIPE_SECRET or not STRIPE_PRICE_ID:
        raise HTTPException(500, "Stripe ödeme sistemi şu an aktif değil.")
    try:
        host = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "localhost")
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=f"https://{host}/?upgraded=1",
            cancel_url=f"https://{host}/",
            metadata={"user_id": str(user.id)},
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(500, f"Stripe oturumu oluşturulamadı: {str(e)}")

@app.post("/stripe-webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WH_SECRET)
    except Exception as e:
        logger.error(f"Webhook Hatası: {e}")
        raise HTTPException(400, "Geçersiz Webhook Talebi")
    if event["type"] == "checkout.session.completed":
        uid = event["data"]["object"]["metadata"].get("user_id")
        if uid:
            user = db.query(User).filter(User.id == int(uid)).first()
            if user:
                user.plan = "pro"
                db.commit()
                logger.info(f"Kullanıcı {uid} PRO plana yükseltildi.")
    return {"status": "success"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# ── Frontend ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>InvoiceAI — Fatura Otomasyonu</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f8f9fa; color: #1a1a2e; min-height: 100vh; }
  .hero { background: linear-gradient(135deg, #534AB7 0%, #3d3891 100%);
          color: white; padding: 60px 20px; text-align: center; }
  .hero h1 { font-size: 2.5rem; margin-bottom: 12px; }
  .hero p { font-size: 1.1rem; opacity: 0.85; margin-bottom: 32px; }
  .btn { display: inline-block; padding: 14px 32px; border-radius: 8px;
         font-size: 1rem; font-weight: 600; cursor: pointer; border: none;
         text-decoration: none; transition: all .2s; }
  .btn-white { background: white; color: #534AB7; }
  .btn-white:hover { background: #f0efff; }
  .btn-outline { background: transparent; color: white;
                 border: 2px solid white; margin-left: 12px; }
  .btn-outline:hover { background: rgba(255,255,255,0.1); }
  .container { max-width: 900px; margin: 0 auto; padding: 0 20px; }
  .features { padding: 60px 20px; }
  .features h2 { text-align: center; font-size: 1.8rem; margin-bottom: 40px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 24px; }
  .card { background: white; border-radius: 12px; padding: 28px;
          box-shadow: 0 2px 12px rgba(0,0,0,0.07); }
  .card-icon { font-size: 2rem; margin-bottom: 12px; }
  .card h3 { font-size: 1.1rem; margin-bottom: 8px; }
  .card p { font-size: 0.9rem; color: #666; line-height: 1.6; }
  .auth-section { background: white; border-radius: 16px; padding: 40px;
                  max-width: 420px; margin: 0 auto 60px; box-shadow: 0 4px 24px rgba(0,0,0,0.1); }
  .auth-section h2 { margin-bottom: 24px; font-size: 1.4rem; }
  .tabs { display: flex; gap: 8px; margin-bottom: 24px; }
  .tab { flex: 1; padding: 10px; border: 1.5px solid #e0e0e0; border-radius: 8px;
         background: white; cursor: pointer; font-size: 14px; font-weight: 500;
         color: #666; transition: all .15s; }
  .tab.active { border-color: #534AB7; color: #534AB7; background: #f0efff; }
  input { width: 100%; padding: 12px 14px; border: 1.5px solid #e0e0e0;
          border-radius: 8px; font-size: 14px; margin-bottom: 12px;
          outline: none; transition: border-color .15s; }
  input:focus { border-color: #534AB7; }
  .btn-primary { width: 100%; background: #534AB7; color: white; padding: 13px;
                 border-radius: 8px; border: none; font-size: 15px; font-weight: 600;
                 cursor: pointer; transition: background .2s; }
  .btn-primary:hover { background: #3d3891; }
  .msg { padding: 10px 14px; border-radius: 8px; font-size: 13px;
         margin-bottom: 12px; display: none; }
  .msg.success { background: #EAF3DE; color: #27500A; display: block; }
  .msg.error { background: #FCEBEB; color: #791F1F; display: block; }
  .dashboard { display: none; }
  .dash-card { background: white; border-radius: 12px; padding: 24px;
               box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 20px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 99px;
           font-size: 12px; font-weight: 600; }
  .badge-free { background: #E6F1FB; color: #0C447C; }
  .badge-pro { background: #EAF3DE; color: #27500A; }
  .invoice-list { margin-top: 16px; }
  .inv-item { padding: 12px 0; border-bottom: 1px solid #f0f0f0;
              display: flex; justify-content: space-between; align-items: center; }
  .inv-item:last-child { border-bottom: none; }
  .inv-vendor { font-weight: 500; font-size: 14px; }
  .inv-amount { font-size: 14px; color: #534AB7; font-weight: 600; }
  .inv-cat { font-size: 11px; color: #888; margin-top: 2px; }
  #upload-area { border: 2px dashed #d0d0d0; border-radius: 10px; padding: 32px;
                 text-align: center; cursor: pointer; transition: all .2s; margin-bottom: 12px; }
  #upload-area:hover { border-color: #534AB7; background: #f8f7ff; }
  #upload-area p { color: #888; font-size: 14px; }
  .spinner { display: none; text-align: center; padding: 12px;
             color: #534AB7; font-size: 14px; }
</style>
</head>
<body>

<div class="hero">
  <div class="container">
    <h1>⚡ InvoiceAI</h1>
    <p>Faturalarını yükle, AI vergi özetini saniyeler içinde çıkarsın.</p>
    <a href="#auth" class="btn btn-white">Ücretsiz Başla</a>
    <a href="/docs" class="btn btn-outline">API Docs</a>
  </div>
</div>

<div class="features">
  <div class="container">
    <h2>Neden InvoiceAI?</h2>
    <div class="grid">
      <div class="card">
        <div class="card-icon">📄</div>
        <h3>PDF Analizi</h3>
        <p>Fatura PDF'ini yükle, AI tüm bilgileri otomatik çıkarsın.</p>
      </div>
      <div class="card">
        <div class="card-icon">🤖</div>
        <h3>AI Destekli</h3>
        <p>Türkçe ve İngilizce faturalar desteklenir. KDV, tutar, tarih otomatik.</p>
      </div>
      <div class="card">
        <div class="card-icon">💰</div>
        <h3>Vergi Özeti</h3>
        <p>Aylık harcama ve KDV özeti. Muhasebecine hazır rapor.</p>
      </div>
    </div>
  </div>
</div>

<div class="container" id="auth">
  <div class="auth-section" id="auth-box">
    <h2>Hesabına Giriş Yap</h2>
    <div class="tabs">
      <button class="tab active" onclick="switchTab('login')">Giriş Yap</button>
      <button class="tab" onclick="switchTab('register')">Kayıt Ol</button>
    </div>
    <div id="auth-msg" class="msg"></div>
    <input type="email" id="email" placeholder="E-posta adresin">
    <input type="password" id="password" placeholder="Şifren">
    <button class="btn-primary" id="auth-btn" onclick="doAuth()">Giriş Yap</button>
  </div>

  <div class="dashboard" id="dashboard">
    <div class="dash-card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="font-size:1.1rem;font-weight:600" id="user-email">—</div>
          <span class="badge badge-free" id="plan-badge">free</span>
          <span style="font-size:13px;color:#888;margin-left:8px" id="usage-info"></span>
        </div>
        <div style="display:flex;gap:8px">
          <button class="btn-primary" style="width:auto;padding:8px 16px;font-size:13px"
                  onclick="goPro()">Pro'ya Geç</button>
          <button onclick="logout()"
                  style="padding:8px 16px;border-radius:8px;border:1px solid #ddd;
                         background:white;cursor:pointer;font-size:13px">Çıkış</button>
        </div>
      </div>
    </div>

    <div class="dash-card">
      <h3 style="margin-bottom:16px">📤 Fatura Yükle</h3>
      <div id="upload-area" onclick="document.getElementById('file-input').click()">
        <p>📄 PDF faturanı buraya sürükle veya tıkla</p>
        <p style="font-size:12px;margin-top:4px">Maks. 10 MB</p>
      </div>
      <input type="file" id="file-input" accept=".pdf" style="display:none" onchange="uploadPDF()">
      <div class="spinner" id="spinner">⏳ AI analiz ediyor...</div>
      <div id="result-msg" class="msg"></div>
    </div>

    <div class="dash-card">
      <h3 style="margin-bottom:4px">📋 Faturalarım</h3>
      <div class="invoice-list" id="invoice-list">
        <p style="color:#888;font-size:13px;padding:16px 0">Henüz fatura yok.</p>
      </div>
    </div>
  </div>
</div>

<script>
let token = localStorage.getItem("invoiceai_token");
let currentTab = "login";
const API = "";

if (token) loadDashboard();

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll(".tab").forEach((t,i) => {
    t.classList.toggle("active", (i===0 && tab==="login") || (i===1 && tab==="register"));
  });
  document.getElementById("auth-btn").textContent = tab === "login" ? "Giriş Yap" : "Kayıt Ol";
  document.getElementById("auth-msg").className = "msg";
}

async function doAuth() {
  const email = document.getElementById("email").value;
  const pass  = document.getElementById("password").value;
  if (!email || !pass) return showMsg("auth-msg", "E-posta ve şifre girin.", "error");
  const endpoint = currentTab === "login" ? "/login" : "/register";
  try {
    const res = await fetch(API + endpoint, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({email, password: pass})
    });
    const data = await res.json();
    if (!res.ok) return showMsg("auth-msg", data.detail || "Hata oluştu.", "error");
    if (currentTab === "register") {
      showMsg("auth-msg", "Kayıt başarılı! Giriş yapabilirsiniz.", "success");
      switchTab("login");
    } else {
      localStorage.setItem("invoiceai_token", data.token);
      token = data.token;
      loadDashboard();
    }
  } catch { showMsg("auth-msg", "Sunucuya ulaşılamadı.", "error"); }
}

async function loadDashboard() {
  try {
    const res = await fetch(API + "/me", {headers: {"Authorization": "Bearer " + token}});
    if (!res.ok) { logout(); return; }
    const u = await res.json();
    document.getElementById("auth-box").style.display = "none";
    document.getElementById("dashboard").style.display = "block";
    document.getElementById("user-email").textContent = u.email;
    const badge = document.getElementById("plan-badge");
    badge.textContent = u.plan;
    badge.className = "badge " + (u.plan === "pro" ? "badge-pro" : "badge-free");
    document.getElementById("usage-info").textContent =
      u.plan === "pro" ? "Sınırsız kullanım" : `${u.remaining} analiz hakkı kaldı`;
    loadInvoices();
  } catch { logout(); }
}

async function loadInvoices() {
  const res = await fetch(API + "/invoices", {headers:{"Authorization":"Bearer "+token}});
  const list = await res.json();
  const el = document.getElementById("invoice-list");
  if (!list.length) { el.innerHTML = '<p style="color:#888;font-size:13px;padding:16px 0">Henüz fatura yok.</p>'; return; }
  el.innerHTML = list.map(inv => `
    <div class="inv-item">
      <div>
        <div class="inv-vendor">${inv.vendor || "Bilinmeyen"}</div>
        <div class="inv-cat">${inv.category || ""} · ${inv.date || ""}</div>
      </div>
      <div class="inv-amount">${inv.amount ? inv.amount.toLocaleString("tr-TR") : "—"} ${inv.currency || ""}</div>
    </div>`).join("");
}

async function uploadPDF() {
  const file = document.getElementById("file-input").files[0];
  if (!file) return;
  document.getElementById("spinner").style.display = "block";
  document.getElementById("result-msg").className = "msg";
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch(API + "/analyze-pdf", {
      method: "POST", headers: {"Authorization": "Bearer " + token}, body: form
    });
    const data = await res.json();
    document.getElementById("spinner").style.display = "none";
    if (!res.ok) {
      showMsg("result-msg", data.detail || "Hata oluştu.", "error");
    } else {
      const inv = data.invoice;
      showMsg("result-msg",
        `✅ ${inv.vendor || "Fatura"} — ${inv.amount || "?"} ${inv.currency || ""} · ${inv.summary_tr || ""}`,
        "success");
      loadInvoices();
      loadDashboard();
    }
  } catch {
    document.getElementById("spinner").style.display = "none";
    showMsg("result-msg", "Yükleme hatası.", "error");
  }
  document.getElementById("file-input").value = "";
}

async function goPro() {
  try {
    const res = await fetch(API + "/checkout", {
      method:"POST", headers:{"Authorization":"Bearer "+token}
    });
    const data = await res.json();
    if (data.url) window.location.href = data.url;
    else alert(data.detail || "Ödeme sistemi aktif değil.");
  } catch { alert("Sunucuya ulaşılamadı."); }
}

function logout() {
  localStorage.removeItem("invoiceai_token");
  token = null;
  document.getElementById("auth-box").style.display = "block";
  document.getElementById("dashboard").style.display = "none";
}

function showMsg(id, text, type) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = "msg " + type;
}

document.addEventListener("keydown", e => {
  if (e.key === "Enter" && document.getElementById("auth-box").style.display !== "none") doAuth();
});
</script>
</body>
</html>"""
