#!/usr/bin/env python3
"""
Comprehensive project verification script.
Checks for errors, missing dependencies, and configuration issues.
"""

import sys
import os
from pathlib import Path
import importlib.util

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}[OK]{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[FAIL]{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.BLUE}i{Colors.END} {msg}")

def check_python_version():
    """Check Python version."""
    print("\n" + "="*70)
    print("1. PYTHON VERSION CHECK")
    print("="*70)
    version = sys.version_info
    print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible (3.8+)")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False

def check_required_packages():
    """Check if required packages are installed."""
    print("\n" + "="*70)
    print("2. REQUIRED PACKAGES CHECK")
    print("="*70)
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'torch',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'dotenv': 'python-dotenv',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
        'tqdm': 'tqdm',
        'joblib': 'joblib',
    }
    
    optional = {
        'yfinance': 'yfinance',
        'transformers': 'transformers',
        'sentencepiece': 'sentencepiece',
        'requests': 'requests',
        'bs4': 'beautifulsoup4',
    }
    
    missing_required = []
    missing_optional = []
    
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            print_success(f"{package_name} is installed")
        except ImportError:
            print_error(f"{package_name} is NOT installed")
            missing_required.append(package_name)
    
    for module_name, package_name in optional.items():
        try:
            __import__(module_name)
            print_success(f"{package_name} is installed (optional)")
        except ImportError:
            print_warning(f"{package_name} is NOT installed (optional)")
            missing_optional.append(package_name)
    
    if missing_required:
        print_error(f"\nMissing required packages: {', '.join(missing_required)}")
        print_info("Install with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print_warning(f"\nMissing optional packages: {', '.join(missing_optional)}")
        print_info("These are optional but recommended for full functionality")
    
    return True

def check_project_structure():
    """Check project directory structure."""
    print("\n" + "="*70)
    print("3. PROJECT STRUCTURE CHECK")
    print("="*70)
    
    required_dirs = [
        'src',
        'src/data',
        'src/models',
        'src/training',
        'src/utils',
        'src/interpretability',
        'config',
        'data/raw',
        'data/processed',
        'data/models',
        'data/figures',
        'notebooks',
        'dashboard',
    ]
    
    required_files = [
        'requirements.txt',
        'config/config.yaml',
        'main.py',
        'README.md',
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/training/__init__.py',
        'src/utils/__init__.py',
        'src/interpretability/__init__.py',
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_success(f"Directory exists: {dir_path}/")
        else:
            print_error(f"Directory missing: {dir_path}/")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"File exists: {file_path}")
        else:
            print_error(f"File missing: {file_path}")
            all_good = False
    
    return all_good

def check_imports():
    """Check if all source modules can be imported."""
    print("\n" + "="*70)
    print("4. MODULE IMPORT CHECK")
    print("="*70)
    
    modules_to_check = [
        'src.data.collectors',
        'src.data.preprocessor',
        'src.data.validator',
        'src.data.feature_engineering',
        'src.data.augmentation',
        'src.models.multimodal_model',
        'src.models.baselines',
        'src.models.encoders',
        'src.models.attention',
        'src.training.trainer',
        'src.training.evaluator',
        'src.utils.helpers',
        'src.utils.logger',
        'src.utils.model_scaler',
        'src.interpretability.attention_viz',
        'src.interpretability.ablation',
    ]
    
    failed_imports = []
    
    for module_name in modules_to_check:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                print_error(f"Cannot find module: {module_name}")
                failed_imports.append(module_name)
            else:
                # Try to actually import it
                try:
                    __import__(module_name)
                    print_success(f"Module imports: {module_name}")
                except Exception as e:
                    print_error(f"Module import failed: {module_name} - {str(e)[:100]}")
                    failed_imports.append(module_name)
        except Exception as e:
            print_error(f"Error checking module: {module_name} - {str(e)[:100]}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print_error(f"\nFailed to import {len(failed_imports)} module(s)")
        return False
    
    return True

def check_config_file():
    """Check configuration file."""
    print("\n" + "="*70)
    print("5. CONFIGURATION FILE CHECK")
    print("="*70)
    
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        print_error("config/config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['data', 'model', 'training']
        for key in required_keys:
            if key in config:
                print_success(f"Config section '{key}' exists")
            else:
                print_error(f"Config section '{key}' missing")
                return False
        
        # Check specific values
        if 'commodity' in config.get('data', {}):
            print_success(f"Commodity: {config['data']['commodity']}")
        if 'location' in config.get('data', {}):
            print_success(f"Location: {config['data']['location']}")
        if 'encoder_type' in config.get('model', {}):
            print_success(f"Encoder type: {config['model']['encoder_type']}")
        
        return True
    except Exception as e:
        print_error(f"Error reading config: {e}")
        return False

def check_syntax():
    """Check Python syntax in all source files."""
    print("\n" + "="*70)
    print("6. SYNTAX CHECK")
    print("="*70)
    
    import ast
    
    python_files = []
    for root, dirs, files in os.walk('src'):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    # Also check main files
    for file in ['main.py', 'collect_daily_data.py', 'test_apis.py', 'diagnose_data.py']:
        if Path(file).exists():
            python_files.append(Path(file))
    
    syntax_errors = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code, filename=str(file_path))
            print_success(f"Syntax OK: {file_path}")
        except SyntaxError as e:
            print_error(f"Syntax error in {file_path}: {e}")
            syntax_errors.append((file_path, e))
        except Exception as e:
            print_warning(f"Could not check {file_path}: {e}")
    
    if syntax_errors:
        print_error(f"\nFound {len(syntax_errors)} syntax error(s)")
        return False
    
    return True

def main():
    """Run all checks."""
    print("\n" + "="*70)
    print("PROJECT VERIFICATION REPORT")
    print("="*70)
    
    results = {
        'Python Version': check_python_version(),
        'Required Packages': check_required_packages(),
        'Project Structure': check_project_structure(),
        'Module Imports': check_imports(),
        'Configuration': check_config_file(),
        'Syntax': check_syntax(),
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print_success("ALL CHECKS PASSED - Project is ready!")
    else:
        print_error("SOME CHECKS FAILED - Please fix the issues above")
        print_info("\nTo install dependencies:")
        print_info("  pip install -r requirements.txt")
        print_info("\nOr using conda:")
        print_info("  conda install --file requirements.txt")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

