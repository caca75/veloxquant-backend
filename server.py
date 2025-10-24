# server.py — VeloxQuant (FastAPI + MySQL / SQLAlchemy)

from fastapi import (
    FastAPI, APIRouter, Depends, HTTPException,
    UploadFile, File, Form
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext

from typing import Optional, List
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import uuid
import shutil
import json
import logging

from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, Boolean, Text,
    ForeignKey, or_
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session


# -----------------------------------------------------------------------------#
# ENV & CONSTANTS
# -----------------------------------------------------------------------------#
ROOT_DIR = Path(__file__).parent.resolve()
load_dotenv(ROOT_DIR / ".env")

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "veloxquant_db")

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-please")
JWT_ALG = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "4320"))  # 3 jours

CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o]

PAY_ADDR_BTC = os.getenv("PAY_ADDR_BTC", "")
PAY_ADDR_TRX = os.getenv("PAY_ADDR_TRX", "")
PAY_ADDR_USDT = os.getenv("PAY_ADDR_USDT", "")

UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------#
# LOGGING
# -----------------------------------------------------------------------------#
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("veloxquant")


# -----------------------------------------------------------------------------#
# DB (SQLAlchemy)
# -----------------------------------------------------------------------------#
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------------------------------------#
# SECURITY
# -----------------------------------------------------------------------------#
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer()

def hash_password(p: str) -> str:
    return pwd_context.hash(p)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(payload: dict) -> str:
    to_encode = payload.copy()
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


# -----------------------------------------------------------------------------#
# MODELS
# -----------------------------------------------------------------------------#
class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(190), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")  # user/admin
    balance = Column(Float, default=0.0)       # soldes disponibles
    profit_total = Column(Float, default=0.0)  # cumul des profits
    referral_code = Column(String(16), unique=True, index=True)
    referred_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    has_first_payment = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Plan(Base):
    __tablename__ = "plans"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    name_fr = Column(String(100), nullable=False)
    name_es = Column(String(100), nullable=False)
    price_usd = Column(Float, nullable=False)
    features = Column(Text, default="[]")      # JSON string list
    features_fr = Column(Text, default="[]")
    features_es = Column(Text, default="[]")
    max_cycles_per_day = Column(Integer, default=1)


class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    plan_id = Column(String(36), ForeignKey("plans.id"), nullable=False)
    status = Column(String(20), default="active")  # active/expired/cancelled
    start_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_date = Column(DateTime, nullable=False)


class ManualPayment(Base):
    __tablename__ = "manual_payments"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    plan_id = Column(String(36), ForeignKey("plans.id"), nullable=False)
    currency = Column(String(10), nullable=False)  # BTC/TRX/USDT
    address = Column(String(255), nullable=False)
    amount_usd = Column(Float, nullable=False)
    tx_hash = Column(String(255), nullable=False)
    screenshot_url = Column(String(255), nullable=False)
    status = Column(String(20), default="PENDING")  # PENDING/APPROVED/REJECTED
    rejection_reason = Column(Text, nullable=True)
    reviewed_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Cycle(Base):
    __tablename__ = "cycles"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    start_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_time = Column(DateTime, nullable=True)
    initial_balance = Column(Float, default=1000.0)
    final_balance = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    status = Column(String(20), default="active")  # active/completed


class Withdrawal(Base):
    __tablename__ = "withdrawals"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    crypto_address = Column(String(255), nullable=False)
    crypto_type = Column(String(10), nullable=False)  # BTC/USDT/TRX
    status = Column(String(20), default="PENDING")
    # PENDING/APPROVED/REJECTED
    rejection_reason = Column(Text, nullable=True)
    reviewed_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# -----------------------------------------------------------------------------#
# SCHEMAS
# -----------------------------------------------------------------------------#
class UserOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    balance: float
    profit_total: float
    referral_code: Optional[str] = None
    referred_by: Optional[str] = None
    has_first_payment: bool
    created_at: datetime

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    referral_code: Optional[str] = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class CycleCreateIn(BaseModel):
    initial_balance: float = 1000.0

class WithdrawalCreateIn(BaseModel):
    amount: float
    crypto_address: str
    crypto_type: str


# -----------------------------------------------------------------------------#
# APP & ROUTER
# -----------------------------------------------------------------------------#
app = FastAPI(title="VeloxQuant API")
api = APIRouter(prefix="/api")

# static for uploads
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------#
# HELPERS (AUTH)
# -----------------------------------------------------------------------------#
def issue_token_for_user(u: User) -> str:
    return create_access_token({"sub": u.id, "email": u.email, "role": u.role})

def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if not dt:
        return dt
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def _user_to_out(u: User) -> UserOut:
    return UserOut(
        id=u.id,
        email=u.email,
        role=u.role,
        balance=float(u.balance or 0.0),
        profit_total=float(u.profit_total or 0.0),
        referral_code=u.referral_code,
        referred_by=u.referred_by,
        has_first_payment=bool(u.has_first_payment),
        created_at=_ensure_utc(u.created_at),
    )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALG])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token invalide")
        u = db.get(User, user_id)
        if not u:
            raise HTTPException(status_code=401, detail="Utilisateur introuvable")
        return u
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide")

async def get_admin_user(u: User = Depends(get_current_user)) -> User:
    if u.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")
    return u


# -----------------------------------------------------------------------------#
# AUTH
# -----------------------------------------------------------------------------#
@api.post("/auth/register", response_model=TokenOut)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email déjà enregistré")
    ref_by = None
    if payload.referral_code:
        ref_user = db.query(User).filter(User.referral_code == payload.referral_code.upper()).first()
        if ref_user:
            ref_by = ref_user.id
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        role="user",
        referral_code=str(uuid.uuid4())[:8].upper(),
        referred_by=ref_by,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = issue_token_for_user(user)
    return TokenOut(access_token=token, user=_user_to_out(user))

@api.post("/auth/login", response_model=TokenOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == payload.email).first()
    if not u or not verify_password(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    token = issue_token_for_user(u)
    return TokenOut(access_token=token, user=_user_to_out(u))

@api.get("/auth/me", response_model=UserOut)
def me(u: User = Depends(get_current_user)):
    return _user_to_out(u)


# -----------------------------------------------------------------------------#
# PLANS
# -----------------------------------------------------------------------------#
@api.get("/plans")
def get_plans(db: Session = Depends(get_db)):
    rows = db.query(Plan).all()
    out = []
    for p in rows:
        out.append({
            "id": p.id,
            "name": p.name,
            "name_fr": p.name_fr,
            "name_es": p.name_es,
            "price_usd": p.price_usd,
            "features": json.loads(p.features or "[]"),
            "features_fr": json.loads(p.features_fr or "[]"),
            "features_es": json.loads(p.features_es or "[]"),
            "max_cycles_per_day": p.max_cycles_per_day,
        })
    return out


# -----------------------------------------------------------------------------#
# BILLING — PAYMENT ADDRESSES
# -----------------------------------------------------------------------------#
@api.get("/billing/addresses")
def get_addresses():
    return {
        "btc": PAY_ADDR_BTC or "",
        "trx": PAY_ADDR_TRX or "",
        "usdt": PAY_ADDR_USDT or "",
    }


# -----------------------------------------------------------------------------#
# BILLING — MANUAL PAYMENTS
# -----------------------------------------------------------------------------#
@api.post("/billing/manual/submit")
def submit_manual_payment(
    plan_id: str = Form(...),
    currency: str = Form(...),
    amount_usd: float = Form(...),
    tx_hash: str = Form(...),
    screenshot: UploadFile = File(...),
    u: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    currency = currency.upper()
    if currency not in {"BTC", "TRX", "USDT"}:
        raise HTTPException(status_code=400, detail="Devise invalide")

    plan = db.get(Plan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan introuvable")

    # Save screenshot
    ext = (screenshot.filename or "").split(".")[-1].lower()
    fname = f"{uuid.uuid4()}.{ext or 'png'}"
    fpath = UPLOAD_DIR / fname
    with fpath.open("wb") as f:
        shutil.copyfileobj(screenshot.file, f)

    addr_map = {"BTC": PAY_ADDR_BTC, "TRX": PAY_ADDR_TRX, "USDT": PAY_ADDR_USDT}
    pay = ManualPayment(
        user_id=u.id,
        plan_id=plan_id,
        currency=currency,
        address=addr_map[currency] or "",
        amount_usd=amount_usd,
        tx_hash=tx_hash,
        screenshot_url=f"/uploads/{fname}",
        status="PENDING",
    )
    db.add(pay)
    db.commit()
    return {"ok": True, "id": pay.id, "message": "Paiement soumis pour vérification"}

@api.get("/billing/my-payments")
def my_payments(u: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(ManualPayment)
        .filter(ManualPayment.user_id == u.id)
        .order_by(ManualPayment.created_at.desc())
        .all()
    )
    def row_to_dict(x: ManualPayment):
        return {
            "id": x.id,
            "plan_id": x.plan_id,
            "currency": x.currency,
            "address": x.address,
            "amount_usd": x.amount_usd,
            "tx_hash": x.tx_hash,
            "screenshot_url": x.screenshot_url,
            "status": x.status,
            "rejection_reason": x.rejection_reason,
            "reviewed_by": x.reviewed_by,
            "created_at": x.created_at,
        }
    return [row_to_dict(r) for r in rows]


# -----------------------------------------------------------------------------#
# SUBSCRIPTIONS
# -----------------------------------------------------------------------------#
@api.get("/subscriptions/active")
def active_subscription(u: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sub = (
        db.query(Subscription)
        .filter(Subscription.user_id == u.id, Subscription.status == "active")
        .order_by(Subscription.start_date.desc())
        .first()
    )
    if not sub:
        return None
    plan = db.get(Plan, sub.plan_id)
    return {
        "subscription": {
            "id": sub.id,
            "user_id": sub.user_id,
            "plan_id": sub.plan_id,
            "status": sub.status,
            "start_date": _ensure_utc(sub.start_date),
            "end_date": _ensure_utc(sub.end_date),
        },
        "plan": {
            "id": plan.id,
            "name": plan.name,
            "name_fr": plan.name_fr,
            "name_es": plan.name_es,
            "price_usd": plan.price_usd,
            "max_cycles_per_day": plan.max_cycles_per_day,
        } if plan else None
    }


# -----------------------------------------------------------------------------#
# CYCLES
# -----------------------------------------------------------------------------#
@api.post("/cycles/start")
def start_cycle(payload: CycleCreateIn, u: User = Depends(get_current_user), db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)

    # 1) Abonnement actif requis
    sub = (
        db.query(Subscription)
        .filter(Subscription.user_id == u.id, Subscription.status == "active")
        .order_by(Subscription.start_date.desc())
        .first()
    )
    if not sub:
        raise HTTPException(
            status_code=403,
            detail="Abonnement actif requis (validation par l’administrateur)."
        )

    # 2) Expiration
    end = _ensure_utc(sub.end_date)
    if end < now:
        raise HTTPException(
            status_code=403,
            detail="Abonnement expiré. Merci de renouveler votre plan."
        )

    # 3) Limite quotidienne de cycles
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    cycles_today = (
        db.query(Cycle)
        .filter(Cycle.user_id == u.id, Cycle.start_time >= today_start)
        .count()
    )
    plan = db.get(Plan, sub.plan_id)
    if not plan:
        raise HTTPException(status_code=400, detail="Plan introuvable.")
    if cycles_today >= (plan.max_cycles_per_day or 0):
        raise HTTPException(status_code=403, detail="Limite quotidienne de cycles atteinte.")

    # 4) Démarrage du cycle
    initial = float(payload.initial_balance or 1000.0)
    cyc = Cycle(user_id=u.id, initial_balance=initial, status="active")
    db.add(cyc)
    db.commit()
    db.refresh(cyc)
    return {
        "ok": True,
        "cycle": {
            "id": cyc.id,
            "start_time": cyc.start_time,
            "initial_balance": cyc.initial_balance,
            "status": cyc.status,
        }
    }

@api.get("/cycles/my-cycles")
def my_cycles(u: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Cycle)
        .filter(Cycle.user_id == u.id)
        .order_by(Cycle.start_time.desc())
        .all()
    )
    def to_dict(c: Cycle):
        return {
            "id": c.id,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "initial_balance": c.initial_balance,
            "final_balance": c.final_balance,
            "profit": c.profit,
            "status": c.status,
        }
    return [to_dict(r) for r in rows]

@api.post("/cycles/{cycle_id}/complete")
def complete_cycle(cycle_id: str, u: User = Depends(get_current_user), db: Session = Depends(get_db)):
    c = db.query(Cycle).filter(Cycle.id == cycle_id, Cycle.user_id == u.id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Cycle introuvable")
    if c.status == "completed":
        return {"ok": True, "profit": c.profit, "final_balance": c.final_balance}

    import random
    pct = random.uniform(0.01, 0.05)
    profit = round((c.initial_balance * pct), 2)
    c.profit = profit
    c.final_balance = c.initial_balance + profit
    c.status = "completed"
    c.end_time = datetime.now(timezone.utc)
    db.commit()
    return {"ok": True, "profit": profit, "final_balance": c.final_balance}


# -----------------------------------------------------------------------------#
# WITHDRAWALS
# -----------------------------------------------------------------------------#
@api.post("/withdrawals/request")
def request_withdrawal(payload: WithdrawalCreateIn, u: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if payload.amount <= 0:
        raise HTTPException(status_code=400, detail="Montant invalide")
    if payload.amount > (u.balance or 0.0):
        raise HTTPException(status_code=400, detail=f"Solde insuffisant. Disponible: ${u.balance:.2f}")
    if payload.crypto_type.upper() not in {"BTC", "USDT", "TRX"}:
        raise HTTPException(status_code=400, detail="Type de crypto invalide")

    w = Withdrawal(
        user_id=u.id,
        amount=payload.amount,
        crypto_address=payload.crypto_address,
        crypto_type=payload.crypto_type.upper(),
        status="PENDING",
    )
    db.add(w)
    db.commit()
    return {"ok": True, "withdrawal_id": w.id}

@api.get("/withdrawals/my-requests")
def my_withdrawals(u: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Withdrawal)
        .filter(Withdrawal.user_id == u.id)
        .order_by(Withdrawal.created_at.desc())
        .all()
    )
    def to_dict(w: Withdrawal):
        return {
            "id": w.id,
            "amount": w.amount,
            "crypto_address": w.crypto_address,
            "crypto_type": w.crypto_type,
            "status": w.status,
            "rejection_reason": w.rejection_reason,
            "created_at": w.created_at,
        }
    return [to_dict(r) for r in rows]


# -----------------------------------------------------------------------------#
# ADMIN
# -----------------------------------------------------------------------------#
@api.get("/admin/users")
def admin_users(_: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    rows = db.query(User).order_by(User.created_at.desc()).all()
    # On renvoie un dict explicite (compat Pydantic v1/v2) AVEC balance/profit_total
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "email": r.email,
            "role": r.role,
            "balance": float(r.balance or 0.0),
            "profit_total": float(r.profit_total or 0.0),
            "referral_code": r.referral_code,
            "referred_by": r.referred_by,
            "has_first_payment": bool(r.has_first_payment),
            "created_at": _ensure_utc(r.created_at),
        })
    return out

@api.post("/admin/users/add-funds")
def admin_add_funds(
    user_identifier: str = Form(...),  # email OU id
    amount: float = Form(...),
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Montant invalide")
    u = db.query(User).filter(or_(User.email == user_identifier, User.id == user_identifier)).first()
    if not u:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    u.balance = float(u.balance or 0.0) + float(amount)
    db.commit()
    db.refresh(u)  # <<< garantit que la valeur renvoyée est mise à jour

    log.info(f"[ADMIN] {admin.email} +{amount}$ au solde de {u.email} => {u.balance}")
    return {"ok": True, "user_email": u.email, "amount_added": amount, "new_balance": float(u.balance or 0.0)}

@api.post("/admin/users/add-profit")
def admin_add_profit(
    user_identifier: str = Form(...),  # email OU id
    amount: float = Form(...),         # positif ou négatif
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    u = db.query(User).filter(or_(User.email == user_identifier, User.id == user_identifier)).first()
    if not u:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    u.profit_total = float(u.profit_total or 0.0) + float(amount)
    db.commit()
    db.refresh(u)  # <<< idem

    log.info(f"[ADMIN] {admin.email} profit_total {amount:+} pour {u.email} => {u.profit_total}")
    return {"ok": True, "user_email": u.email, "delta_profit": amount, "profit_total": float(u.profit_total or 0.0)}

@api.get("/admin/manual-payments")
def admin_list_payments(
    status_filter: Optional[str] = None,
    _: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    q = db.query(ManualPayment)
    if status_filter:
        q = q.filter(ManualPayment.status == status_filter.upper())
    rows = q.order_by(ManualPayment.created_at.desc()).all()

    def to_dict(p: ManualPayment):
        user = db.get(User, p.user_id)
        plan = db.get(Plan, p.plan_id)
        return {
            "id": p.id,
            "user_email": user.email if user else "Unknown",
            "plan_name": plan.name if plan else "Unknown",
            "currency": p.currency,
            "address": p.address,
            "amount_usd": p.amount_usd,
            "tx_hash": p.tx_hash,
            "screenshot_url": p.screenshot_url,
            "status": p.status,
            "rejection_reason": p.rejection_reason,
            "reviewed_by": p.reviewed_by,
            "created_at": p.created_at,
        }
    return [to_dict(r) for r in rows]

@api.post("/admin/manual-payments/{payment_id}/approve")
def admin_approve_payment(
    payment_id: str,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    p = db.get(ManualPayment, payment_id)
    if not p:
        raise HTTPException(status_code=404, detail="Paiement introuvable")
    if p.status != "PENDING":
        raise HTTPException(statuscode=400, detail="Déjà traité")

    p.status = "APPROVED"
    p.reviewed_by = admin.id
    db.commit()

    # Commission parrainage (50% 1er paiement)
    user = db.get(User, p.user_id)
    if user and (not user.has_first_payment) and user.referred_by:
        ref = db.get(User, user.referred_by)
        if ref:
            ref.balance = float(ref.balance or 0.0) + float(p.amount_usd) * 0.5
            db.commit()
            db.refresh(ref)  # <<< pour que le front voie tout de suite le nouveau solde
            log.info(f"[REF] +{p.amount_usd*0.5}$ ajouté au parrain {ref.email}")
        user.has_first_payment = True
        db.commit()

    # Abonnement 30j (création/extension)
    now = datetime.now(timezone.utc)
    sub = (
        db.query(Subscription)
        .filter(Subscription.user_id == p.user_id, Subscription.status == "active")
        .first()
    )
    if sub:
        end = _ensure_utc(sub.end_date)
        sub.end_date = end + timedelta(days=30)
        sub.plan_id = p.plan_id
    else:
        db.add(Subscription(
            user_id=p.user_id,
            plan_id=p.plan_id,
            status="active",
            start_date=now,
            end_date=now + timedelta(days=30),
        ))
    db.commit()

    return {"ok": True, "message": "Paiement approuvé et abonnement activé"}

@api.post("/admin/manual-payments/{payment_id}/reject")
def admin_reject_payment(
    payment_id: str,
    reason: str = Form(...),
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    p = db.get(ManualPayment, payment_id)
    if not p:
        raise HTTPException(status_code=404, detail="Paiement introuvable")
    if p.status != "PENDING":
        raise HTTPException(status_code=400, detail="Déjà traité")

    p.status = "REJECTED"
    p.rejection_reason = reason
    p.reviewed_by = admin.id
    db.commit()
    return {"ok": True, "message": "Paiement rejeté"}

@api.get("/admin/withdrawals")
def admin_list_withdrawals(
    status_filter: Optional[str] = None,
    _: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    q = db.query(Withdrawal)
    if status_filter:
        q = q.filter(Withdrawal.status == status_filter.upper())
    rows = q.order_by(Withdrawal.created_at.desc()).all()

    def to_dict(w: Withdrawal):
        user = db.get(User, w.user_id)
        return {
            "id": w.id,
            "user_email": user.email if user else "Unknown",
            "user_balance": float(user.balance or 0.0) if user else 0.0,
            "amount": w.amount,
            "crypto_address": w.crypto_address,
            "crypto_type": w.crypto_type,
            "status": w.status,
            "rejection_reason": w.rejection_reason,
            "created_at": w.created_at,
        }
    return [to_dict(r) for r in rows]

@api.post("/admin/withdrawals/{withdrawal_id}/approve")
def admin_approve_withdrawal(
    withdrawal_id: str,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    w = db.get(Withdrawal, withdrawal_id)
    if not w:
        raise HTTPException(status_code=404, detail="Retrait introuvable")
    if w.status != "PENDING":
        raise HTTPException(statuscode=400, detail="Déjà traité")

    user = db.get(User, w.user_id)
    if (user.balance or 0.0) < w.amount:
        raise HTTPException(status_code=400, detail="Solde insuffisant")

    user.balance = float(user.balance or 0.0) - float(w.amount)
    w.status = "APPROVED"
    w.reviewed_by = admin.id
    db.commit()
    db.refresh(user)  # <<< pour refléter le solde à jour

    return {"ok": True, "message": "Retrait approuvé", "new_user_balance": float(user.balance or 0.0)}

@api.post("/admin/withdrawals/{withdrawal_id}/reject")
def admin_reject_withdrawal(
    withdrawal_id: str,
    reason: str = Form(...),
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    w = db.get(Withdrawal, withdrawal_id)
    if not w:
        raise HTTPException(status_code=404, detail="Retrait introuvable")
    if w.status != "PENDING":
        raise HTTPException(status_code=400, detail="Déjà traité")

    w.status = "REJECTED"
    w.rejection_reason = reason
    w.reviewed_by = admin.id
    db.commit()
    return {"ok": True, "message": "Retrait rejeté"}


# -----------------------------------------------------------------------------#
# SEED (plans + admin)
# -----------------------------------------------------------------------------#
@api.post("/admin/seed")
def seed(db: Session = Depends(get_db)):
    # Plans
    if db.query(Plan).count() == 0:
        def j(x: list) -> str: return json.dumps(x, ensure_ascii=False)
        plans = [
            Plan(
                name="Starter", name_fr="Débutant", name_es="Inicial",
                price_usd=300,
                features=j(["1 cycle per day", "Basic support", "Trading simulator"]),
                features_fr=j(["1 cycle par jour", "Support de base", "Simulateur de trading"]),
                features_es=j(["1 ciclo por día", "Soporte básico", "Simulador de trading"]),
                max_cycles_per_day=1,
            ),
            Plan(
                name="Pro", name_fr="Pro", name_es="Pro",
                price_usd=550,
                features=j(["3 cycles per day", "Priority support", "Advanced analytics"]),
                features_fr=j(["3 cycles par jour", "Support prioritaire", "Analyses avancées"]),
                features_es=j(["3 ciclos por día", "Soporte prioritario", "Análisis avanzado"]),
                max_cycles_per_day=3,
            ),
            Plan(
                name="Elite", name_fr="Élite", name_es="Elite",
                price_usd=1000,
                features=j(["Unlimited cycles", "24/7 VIP support", "Custom strategies"]),
                features_fr=j(["Cycles illimités", "Support VIP 24/7", "Stratégies personnalisées"]),
                features_es=j(["Ciclos ilimitados", "Soporte VIP 24/7", "Estrategias personalizadas"]),
                max_cycles_per_day=999,
            ),
        ]
        db.add_all(plans)
        db.commit()

    # Admin
    if not db.query(User).filter(User.email == "admin@veloxquant.com").first():
        admin = User(
            email="admin@veloxquant.com",
            password_hash=hash_password("admin123"),
            role="admin",
            referral_code=str(uuid.uuid4())[:8].upper(),
        )
        db.add(admin)
        db.commit()

    return {"message": "Seed OK"}


# -----------------------------------------------------------------------------#
# INCLUDE ROUTER + TABLES + HEALTH
# -----------------------------------------------------------------------------#
app.include_router(api)
Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}
