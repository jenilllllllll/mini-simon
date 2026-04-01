import sys
import importlib

def check_package(package_name):
    try:
        module = importlib.import_module(package_name)
        print(f"✅ {package_name} is installed (version: {getattr(module, '__version__', 'unknown')})")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

def main():
    print(f"Python interpreter: {sys.executable}")
    print(f"Python version: {sys.version}")
    print("-" * 30)
    
    packages = ["uvicorn", "fastapi", "pandas", "fyers_apiv3", "jinja2"]
    all_ok = True
    for pkg in packages:
        if not check_package(pkg):
            all_ok = False
    
    print("-" * 30)
    if all_ok:
        print("🚀 All core dependencies are correctly installed and importable!")
    else:
        print("⚠️ Some dependencies are missing. Please run 'pip install -r requirements.txt'")

if __name__ == "__main__":
    main()
