#!/bin/bash

source /home/wyf/paper/.env

# 定义工作目录和 Git 仓库
WORK_DIR="/home/wyf/paper"
REPO_URL="https://github.com/302178/paper.git"
DEST_DIR="tools"  # 目标文件夹



# 进入工作目录
cd "$WORK_DIR" || { echo "Failed to navigate to $WORK_DIR"; exit 1; }

# 打印当前目录
echo "Current working directory: $(pwd)"

# 如果已经存在 git 仓库，执行 pull 更新
cd /home/wyf/paper/paper
echo "Repository found. Pulling latest changes..."
# git pull origin main
# git fetch origin
# git reset --hard origin/main
# 初始化失败计数器
FAIL_COUNT=0
MAX_FAIL_COUNT=20

# 循环尝试 git pull，直到成功或失败次数达到上限
while true; do
    git fetch origin
    git reset --hard origin/main
    if [ $? -eq 0 ]; then
        echo "Git pull successful."
        break
    else
        ((FAIL_COUNT++))
        echo "Git pull failed. Attempt $FAIL_COUNT of $MAX_FAIL_COUNT. Retrying in 5 minutes..."
        if [ $FAIL_COUNT -ge $MAX_FAIL_COUNT ]; then
            echo "Git pull failed $MAX_FAIL_COUNT times. Exiting script."
            exit 1
        fi
        sleep 5m  # 等待 5 分钟后重试
    fi
done



# 确保所需的文件存在
if [ ! -f "$WORK_DIR/paper/$DEST_DIR/keyword-accept.csv" ] || [ ! -f "$WORK_DIR/paper/$DEST_DIR/keyword-reject.csv" ] || [ ! -f "$WORK_DIR/paper/$DEST_DIR/paper.py" ]; then
    echo "Required files (keyword-accept.csv, keyword-reject.csv, paper.py) not found in the '$DEST_DIR' folder."
    exit 1
fi

# 激活 conda 环境
echo "Activating conda environment 'paper'..."
source /home/share/anaconda3/etc/profile.d/conda.sh  # 确保 conda 命令可以执行（如果你是使用 Conda）
conda activate paper || { echo "Failed to activate conda environment"; exit 1; }

# 运行 paper.py 文件
cd tools
echo "Running paper.py..."
python "paper.py" || { echo "Failed to run paper.py"; exit 1; }

echo "Changes detected in daily-arxiv-llm.md. Committing changes..."
    
# 添加修改到 git
git add "$WORK_DIR/paper/daily-arxiv-llm.md"
    
# 提交修改
git commit -m "Update daily-arxiv-llm.md with new papers"

# 推送到 GitHub
# git push origin main || { echo "Failed to push changes to GitHub"; exit 1; }
git push https://$GITHUB_USERNAME:$GITHUB_PAT@github.com/302178/paper.git main || { echo "Failed to push changes to GitHub"; exit 1; }

echo "Changes successfully pushed to GitHub."

echo "Script completed successfully."
