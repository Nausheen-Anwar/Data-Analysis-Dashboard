import hashlib
import re
import hmac
import secrets
import sqlite3
from pathlib import Path
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "users.db"

def init_auth_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        n=16384, r=8, p=1
    )
    return digest.hex(), salt

def verify_password(password, stored_hash, salt):
    calculated, _ = hash_password(password, salt)
    return hmac.compare_digest(calculated, stored_hash)

def validate_user_id(user_id):
    """Validate a new account User ID."""
    user_id = user_id.strip()

    if not 6 <= len(user_id) <= 20:
        return False, "User ID must be between 6 and 20 characters."
    if not user_id[0].isalpha():
        return False, "User ID must start with a letter."
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", user_id):
        return False, "Use only letters, numbers, dots (.) and underscores (_)."
    if not re.search(r"[A-Za-z]", user_id) or not re.search(r"\d", user_id):
        return False, "User ID must contain at least one letter and one number."

    return True, ""


def validate_password(password):
    """Validate a new account password."""
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    if len(password) > 64:
        return False, "Password cannot be longer than 64 characters."
    if any(char.isspace() for char in password):
        return False, "Password cannot contain spaces."
    if not re.search(r"[A-Z]", password):
        return False, "Add at least one uppercase letter (A-Z)."
    if not re.search(r"[a-z]", password):
        return False, "Add at least one lowercase letter (a-z)."
    if not re.search(r"\d", password):
        return False, "Add at least one number (0-9)."
    if not re.search(r"[^A-Za-z0-9\s]", password):
        return False, "Add at least one special character (for example, @, #, $, !)."

    return True, ""


def create_user(user_id, password):
    user_id = user_id.strip()

    valid, message = validate_user_id(user_id)
    if not valid:
        return False, message

    valid, message = validate_password(password)
    if not valid:
        return False, message

    password_hash, salt = hash_password(password)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Prevent duplicate IDs even when letter casing is different.
            existing = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ? COLLATE NOCASE",
                (user_id,)
            ).fetchone()
            if existing:
                return False, "This User ID already exists."

            conn.execute(
                "INSERT INTO users(user_id, password_hash, salt) VALUES (?, ?, ?)",
                (user_id, password_hash, salt)
            )
            conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "This User ID already exists."

def authenticate(user_id, password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE user_id = ? COLLATE NOCASE",
            (user_id.strip(),)
        ).fetchone()
    if row and verify_password(password, row[0], row[1]):
        return True
    return False

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def login_screen():
    init_auth_db()
    _, centre, _ = st.columns([1, 1.4, 1])
    centre.empty()
    logo = Path(__file__).resolve().parent.parent / "assets" / "ust_logo.jpg"
    st.markdown('<div class="ust-title">UST Software India Pvt Ltd</div>', unsafe_allow_html=True)
    if logo.exists():
        st.image(str(logo), width=70)
    st.markdown('<div class ="ust-title">Data Analysis Using Streamlit in Python</div>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>Login</h2>", unsafe_allow_html=True)
    st.caption("Sign in to continue.")

    tab_login, tab_register = st.tabs(["Login", "Create account"])

    with tab_login:
        user_id = st.text_input("User ID", key="login_user")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary", use_container_width=True):
            if authenticate(user_id, password):
                st.session_state.logged_in = True
                st.session_state.user_id = user_id.strip()
                st.rerun()
            else:
                st.error("Invalid User ID or password.")

    with tab_register:
        st.caption(
            "User ID: 6-20 characters; start with a letter and include at least "
            "one number. Letters, numbers, dots and underscores are allowed."
        )
        new_id = st.text_input("User ID", key="register_user")
        st.caption(
            "Password: 8-64 characters with uppercase, lowercase, a number "
            "and a special character. Spaces are not allowed."
        )
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm = st.text_input("Confirm password", type="password", key="register_confirm")
        if st.button("Create account", use_container_width=True):
            if new_password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message = create_user(new_id, new_password)
                if ok:
                    st.toast("Account created successfully. Continue with login.", icon="✅")
                else:
                    st.error(message)
