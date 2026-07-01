import shutil, re
from pathlib import Path

src_dir = Path("src/ranker")
dst_dir = Path("deploy_space/ranker")
dst_dir.parent.mkdir(exist_ok=True)
if dst_dir.exists(): shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)

shutil.copy("testing/sandbox/requirements.txt", "deploy_space/requirements.txt")
shutil.copytree("testing/sandbox/demo_data", "deploy_space/demo_data", 
                 dirs_exist_ok=True)
shutil.copy("testing/sandbox/README.md", "deploy_space/README.md")

app_src = Path("testing/sandbox/app.py").read_text(encoding="utf-8")

# Fix _ROOT and sys.path
app_src = re.sub(
    r"_ROOT\s*=\s*_HERE\.parent\.parent.*?# project root\n",
    "", 
    app_src
)
app_src = app_src.replace(
    "if str(_ROOT) not in sys.path:\n    sys.path.insert(0, str(_ROOT))",
    "if str(_HERE) not in sys.path:\n    sys.path.insert(0, str(_HERE))"
)

# Fix from src.ranker to from ranker
app_src = app_src.replace("from src.ranker", "from ranker")
Path("deploy_space/app.py").write_text(app_src, encoding="utf-8")

print("deploy_space/ built. Upload its CONTENTS (not the folder itself)")
print("to HuggingFace Spaces root.")
