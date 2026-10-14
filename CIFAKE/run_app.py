"""
CIFAKE Application Launcher
Ensures Streamlit executes under the current active Python interpreter.
"""
import sys
import os
import subprocess

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(repo_dir, "app.py")

    print("=" * 65)
    print("🔬 CIFAKE: AI-Generated vs Real Image Forensics Platform")
    print(f"🐍 Python Interpreter: {sys.executable} ({sys.version.split()[0]})")
    print("⚡ Launching Streamlit Web Application (Figures 2 - 7)...")
    print("=" * 65)

    # Run through the current python interpreter to avoid Anaconda/Global PATH collisions
    cmd = [sys.executable, "-m", "streamlit", "run", app_path] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[CIFAKE] Streamlit server stopped.")

if __name__ == "__main__":
    main()
