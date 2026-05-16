Installation Steps

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/yadrikshauprety/video.git
   cd video
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Command Prompt (cmd):
     ```cmd
     .\.venv\Scripts\activate.bat
     ```

4. **Upgrade pip and install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Run the backend**:
   ```bash
   python -m uvicorn api:app
   ```

6. **Run the frontend**:
   ```bash
   streamlit run app.py
   ```
