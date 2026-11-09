"""
SmartPose - User Authentication & Session Management Module
Handles SQLite-backed user registration, password hashing with PBKDF2/SHA256,
login verification, and a modern Streamlit authentication interface.
"""

import os
import sqlite3
import hashlib
import secrets
import datetime
import streamlit as st


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.db")


def get_db_connection():
    """Returns a connection to the SQLite users database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the users table and seeds a demo account if none exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    # Check if demo account exists, if not seed it
    cursor.execute("SELECT id FROM users WHERE username = ?", ("demo",))
    if not cursor.fetchone():
        salt = secrets.token_hex(16)
        pwd_hash = hash_password("password123", salt)
        cursor.execute(
            """
            INSERT INTO users (username, name, email, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "demo",
                "Demo Athlete",
                "demo@smartpose.ai",
                pwd_hash,
                salt,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
    conn.close()


def hash_password(password: str, salt: str) -> str:
    """Hashes a password with a cryptographic salt using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
    ).hex()


def register_user(username: str, name: str, email: str, password: str) -> tuple[bool, str]:
    """Registers a new user into the database."""
    username = username.strip().lower()
    name = name.strip()
    email = email.strip().lower()

    if not username or not name or not email or not password:
        return False, "All fields are required."

    if len(username) < 3:
        return False, "Username must be at least 3 characters long."

    if "@" not in email or "." not in email:
        return False, "Please enter a valid email address."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        salt = secrets.token_hex(16)
        pwd_hash = hash_password(password, salt)
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO users (username, name, email, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username, name, email, pwd_hash, salt, created_at),
        )
        conn.commit()
        return True, "Account created successfully! You can now sign in."
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            return False, "This username is already taken. Please choose another."
        elif "email" in err_msg:
            return False, "An account with this email already exists."
        return False, "Account creation failed due to duplicate credentials."
    finally:
        conn.close()


def authenticate_user(username_or_email: str, password: str) -> tuple[bool, dict, str]:
    """Authenticates credentials against the users database."""
    identifier = username_or_email.strip().lower()
    if not identifier or not password:
        return False, {}, "Please provide both your username/email and password."

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, name, email, password_hash, salt, created_at 
        FROM users 
        WHERE username = ? OR email = ?
        """,
        (identifier, identifier),
    )
    user_row = cursor.fetchone()
    conn.close()

    if not user_row:
        return False, {}, "Account not found. Please check your credentials or sign up."

    expected_hash = hash_password(password, user_row["salt"])
    if secrets.compare_digest(expected_hash, user_row["password_hash"]):
        user_data = {
            "id": user_row["id"],
            "username": user_row["username"],
            "name": user_row["name"],
            "email": user_row["email"],
            "created_at": user_row["created_at"],
        }
        return True, user_data, "Login successful!"
    else:
        return False, {}, "Incorrect password. Please try again."


def render_auth_page():
    """Renders a sleek, modern authentication interface with Login and Sign Up tabs."""
    init_db()

    st.markdown(
        """
        <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 2rem;">
            <div style="font-size: 3.5rem; line-height: 1;">🧘</div>
            <div style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(120deg, #1E88E5 0%, #00C9FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                SMARTPOSE
            </div>
            <div style="color: #64748B; font-size: 1.05rem; font-weight: 500;">
                AI Powered Pose Suggestion & Alignment System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered container for the login/register card
    _, col_center, _ = st.columns([1, 1.4, 1])

    with col_center:
        auth_tab1, auth_tab2 = st.tabs(["🔐 Sign In", "📝 Create Account"])

        # -------------------------------------------------------------
        # TAB 1: SIGN IN
        # -------------------------------------------------------------
        with auth_tab1:
            st.markdown("#### Welcome Back")
            st.caption("Sign in to your SmartPose account to track your posture and alignment history.")

            with st.form("login_form", clear_on_submit=False):
                login_user = st.text_input("Username or Email", placeholder="e.g. demo or demo@smartpose.ai")
                login_pwd = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submit_login:
                success, user_data, msg = authenticate_user(login_user, login_pwd)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = user_data
                    st.success(f"Welcome back, **{user_data['name']}**!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)

            st.markdown("---")
            st.markdown(
                """
                <div style="background-color: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #334155;">🔑 Quick Demo Credentials:</div>
                    <div style="font-size: 0.8rem; color: #64748B;">Username: <code>demo</code> | Password: <code>password123</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Quick Demo Login One-Click Button
            if st.button("⚡ Instant Demo Login", use_container_width=True):
                success, user_data, _ = authenticate_user("demo", "password123")
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = user_data
                    st.rerun()

        # -------------------------------------------------------------
        # TAB 2: REGISTER
        # -------------------------------------------------------------
        with auth_tab2:
            st.markdown("#### Create New Account")
            st.caption("Get started with AI pose coaching and save your personal exercise progress.")

            with st.form("register_form", clear_on_submit=False):
                reg_name = st.text_input("Full Name", placeholder="e.g. Alex Johnson")
                reg_user = st.text_input("Username", placeholder="e.g. alexj")
                reg_email = st.text_input("Email Address", placeholder="e.g. alex@example.com")
                reg_pwd = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
                reg_pwd_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                submit_register = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if submit_register:
                if reg_pwd != reg_pwd_confirm:
                    st.error("Passwords do not match. Please re-enter carefully.")
                else:
                    success, msg = register_user(reg_user, reg_name, reg_email, reg_pwd)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
