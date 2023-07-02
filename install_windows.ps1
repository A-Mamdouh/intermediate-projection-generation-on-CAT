python -m venv env
./env/Scripts/python.exe -m pip install --upgrade pip
./env/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
./env/Scripts/python.exe -m pip install -r requirements.txt