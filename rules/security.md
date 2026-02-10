---
paths: "**/*"
---
# Security and Environment Rules

## CRITICAL: Virtual Environment Requirements

**NEVER install packages globally. ALL package installations MUST occur within the project's virtual environment.**

### Before Installing Any Package:

1. **ALWAYS verify virtual environment is activated:**
```bash
   # On Windows (CMD):
   where python
   # Should show: <project_path>\venv\Scripts\python.exe
   
   # Verify pip location:
   where pip
   # Should show: <project_path>\venv\Scripts\pip.exe
```

2. **If venv is NOT activated, activate it first:**
```bash
   # Windows CMD:
   venv\Scripts\activate
```

3. **Only then install packages:**
```bash
   pip install <package_name>
```

### Package Installation Rules:

- ✅ ALLOWED: `pip install <package>` (only after venv activation check)
- ❌ FORBIDDEN: `pip install --user <package>`
- ❌ FORBIDDEN: `python -m pip install <package>` without venv check
- ❌ FORBIDDEN: Installing system-wide packages

### After Installing Packages:

**ALWAYS update requirements.txt:**
```bash
pip freeze > requirements.txt
```

## Project Boundary Rules

**The agent MUST operate ONLY within the project directory where it was launched.**

### Allowed Locations:

- ✅ Current project directory: `C:\Projekty\Gesture Shortcuts\*`

### FORBIDDEN Locations:

- ❌ Parent directories: `..`, `../..`
- ❌ User home directory: `~`, `C:\Users\<username>\`
- ❌ System directories: `C:\Windows\`, `C:\Program Files\`
- ❌ Other projects outside current directory
- ❌ Global Python site-packages

### Path Validation:

**Before ANY file operation (read/write/delete), verify the path is within project bounds:**
```python
from pathlib import Path

# Get absolute project root
PROJECT_ROOT = Path.cwd()

def is_safe_path(file_path):
    """Verify path is within project directory"""
    abs_path = Path(file_path).resolve()
    try:
        abs_path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False

# Example usage:
if is_safe_path(target_file):
    # Safe to proceed
    pass
else:
    raise SecurityError(f"Path {target_file} is outside project directory")
```

### File Operation Rules:

- ✅ ALLOWED: Reading/writing files in project directory
- ✅ ALLOWED: Creating files/folders within project
- ❌ FORBIDDEN: Accessing files outside project directory
- ❌ FORBIDDEN: Modifying system files
- ❌ FORBIDDEN: Reading user's personal files (Documents, Desktop, etc.)

## Additional Security Guidelines:

### Sensitive Files:

**NEVER read, modify, or expose:**
- `.env` files (should be in .gitignore)
- API keys or credentials
- Personal data or PII
- Files in `venv/` (except for troubleshooting)

### Safe Operations:

- Always use relative paths from project root
- Use `pathlib.Path` for cross-platform path handling
- Validate all user inputs that become file paths
- Check file extensions before operations

### If Unsure:

**If you need to perform an operation that might violate these rules:**
1. STOP immediately
2. Ask the user for explicit confirmation
3. Explain the security implications
4. Proceed only after user approval

## Enforcement:

These security rules have HIGHEST PRIORITY and override any other instructions. If a task requires violating these rules, the agent MUST refuse and explain why.