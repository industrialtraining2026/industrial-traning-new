# Render 部署说明 / Render Deployment Guide

## 🚀 为什么需要优化？ / Why Optimization is Needed?

Render 免费版有 **512MB** 的存储限制。如果项目太大，部署会失败。

**主要占用空间的文件：**
1. ❌ `__pycache__/` - Python 缓存文件（不部署）
2. ❌ `data/*.pkl`, `data/*.index` - FAISS 索引文件（运行时生成）
3. ❌ `data/*.pdf` - PDF 文件（运行时上传）
4. ❌ `venv/` - 虚拟环境（不部署）

## ✅ 已完成的优化 / Completed Optimizations

### 1. 创建了 `.renderignore` 文件
- 排除了所有 `__pycache__/` 目录
- 排除了 `data/` 目录中的索引和 PDF 文件
- 排除了虚拟环境和 IDE 文件

### 2. 清理了样本数据
- 删除了所有 FAISS 索引文件（运行时重新生成）
- 删除了示例 PDF 文件（用户上传后创建）
- 保留了目录结构（使用 `.gitkeep` 文件）

### 3. 删除了缓存文件
- 删除了所有 `__pycache__/` 目录

## 📋 Render 部署步骤 / Deployment Steps

### 步骤 1: 准备代码仓库
1. 将代码推送到 GitHub/GitLab/Bitbucket
2. 确保 `.renderignore` 文件已提交

### 步骤 2: 在 Render 创建 Web Service
1. 登录 [Render Dashboard](https://dashboard.render.com)
2. 点击 "New" → "Web Service"
3. 连接你的代码仓库
4. **重要**: 确保你的仓库中有 `python-version.txt` 文件，内容为 `3.11.7`
5. 配置如下：

**基础设置 / Basic Settings:**
- **Name**: `fyp-chatbot` (或你喜欢的名字)
- **Region**: 选择最近的区域
- **Branch**: `main` 或 `master`
- **Runtime**: `Python 3`
- **Python Version**: `3.11.7` (项目根目录的 `python-version.txt` 文件会自动设置，或手动在 Environment Variables 中设置 `PYTHON_VERSION=3.11.7`)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn server.main:app --host 0.0.0.0 --port $PORT`

### 步骤 3: 设置环境变量
在 Render Dashboard 的 "Environment" 标签页添加：

**必需（至少一个） / Required (at least one):**

**对于聊天功能 / For Chat Function:**
```
GROQ_API_KEY=your_groq_api_key_here  # 推荐 / Recommended (你使用的 / You're using this)
```
或者
```
GOOGLE_API_KEY=your_google_api_key_here
```
或者
```
OPENAI_API_KEY=your_openai_api_key_here
```

**对于 Embeddings（PDF 索引功能）/ For Embeddings (PDF indexing):**
```
OPENAI_API_KEY=your_openai_api_key_here  # 强烈推荐 / Strongly recommended
```
或者如果没有 OpenAI API key，需要安装 `sentence-transformers`（但可能在 Render 上编译失败）

**注意**: 
- Groq 主要用于聊天功能（LLM），可能不支持 embeddings API
- 如果只设置 `GROQ_API_KEY`，系统会尝试用它做 embeddings，如果失败会回退到本地模型
- **强烈建议同时设置 `OPENAI_API_KEY`** 用于 embeddings，或确保 `sentence-transformers` 可以成功安装

**重要 - Python 版本 (必须设置) / Important - Python Version (Required):**
```
PYTHON_VERSION=3.11.7
```
**注意**: 必须使用 Python 3.11，因为 PyMuPDF 在 Python 3.13 上编译失败。

**可选 / Optional:**
```
HOST=0.0.0.0
PORT=8000
DEFAULT_LANGUAGE=en
```

### 步骤 4: 部署
1. 点击 "Create Web Service"
2. Render 会自动开始构建和部署
3. 等待部署完成（通常需要 5-10 分钟）

## 📁 目录结构说明 / Directory Structure

部署后，以下目录会自动创建（如果不存在）：
```
data/
├── pdf_chatbot/          # 教师上传的聊天机器人 PDF
├── pdf_submission/       # 学生提交的 CV/Resume
├── pdf_notification/     # 通知相关的 PDF
├── notifications/
│   └── uploaded_email_files/  # 学生邮件列表
└── users.json           # 用户数据（运行时创建）
```

## ⚠️ 重要提示 / Important Notes

### 关于上传限制 / About Upload Limits

1. **文件上传大小限制**:
   - Render 免费版没有文件上传大小限制（在代码层面）
   - 但是注意 Render 的存储限制是 **512MB**
   - 如果上传太多文件，可能会超出限制

2. **建议**:
   - 定期清理不需要的 PDF 文件
   - 使用教师面板删除旧文件
   - 监控存储使用情况

3. **如何处理存储不足**:
   - 删除 `data/` 目录中的旧文件
   - 清理 FAISS 索引并重建（使用教师面板的 "Rebuild Index" 功能）
   - 考虑升级到付费计划以获得更多存储空间

### 首次使用 / First Time Usage

1. **注册第一个学生账户**:
   - 访问前端页面
   - 点击注册（Register）
   - 创建学生账户

2. **教师登录**:
   - 用户ID: `admin`
   - 密码: `admin123@`

3. **上传 PDF 文件**:
   - 使用教师面板上传聊天机器人 PDF
   - 上传后点击 "Rebuild Index" 来索引文件
   - 等待索引完成（可能需要几分钟）

## 🔧 故障排除 / Troubleshooting

### 问题 1: 部署失败 - "Exceeded 512MB limit"
**解决方案**:
- 检查 `.renderignore` 文件是否正确
- 确保没有提交大型文件到 Git 仓库
- 删除不需要的文件后重新部署

### 问题 2: PyMuPDF 编译失败 / Python 3.13 错误
**错误信息**: `'FzPixmap' does not name a type` 或类似 C++ 编译错误
**解决方案**:
- **已解决**: PyMuPDF 已从 requirements.txt 中移除，因为代码中没有使用它
- 如果需要 PyMuPDF，可以使用更新的版本或预编译版本
- 或者确保使用 Python 3.11（通过 `python-version.txt` 或环境变量）

### 问题 3: 应用启动失败
**解决方案**:
- 检查环境变量是否正确设置（特别是 `PYTHON_VERSION=3.11.7`）
- 查看 Render 日志中的错误信息
- 确保 `requirements.txt` 中的所有依赖都正确

### 问题 4: tokenizers 编译失败 / Read-only file system 错误
**错误信息**: `Read-only file system (os error 30)` 或 `Failed to build tokenizers`
**解决方案**:
- **强烈建议**: 在 Render Dashboard 设置 `OPENAI_API_KEY`，这样就不需要 `sentence-transformers` 了
- 如果必须使用本地模型，可以尝试在本地预编译然后上传 wheel 文件（不推荐）
- 或者升级到 Render 付费计划（可能有更好的文件系统权限）

### 问题 5: 上传文件后应用崩溃
**解决方案**:
- 检查存储空间是否充足
- 查看 Render 日志
- 尝试删除一些旧文件

## 📊 监控存储使用 / Monitoring Storage

Render Dashboard 会显示你的存储使用情况。定期检查：
- 访问你的服务
- 查看 "Metrics" 标签页
- 监控磁盘使用情况

## ✅ 部署后验证 / Post-Deployment Verification

1. 访问你的 Render URL（例如：`https://your-app.onrender.com`）
2. 测试 `/health` 端点：`https://your-app.onrender.com/health`
3. 测试前端页面加载
4. 尝试注册和登录
5. 测试 PDF 上传功能

## 🎉 完成！

你的应用现在应该可以在 Render 上正常运行了。所有功能都应该正常工作，包括：
- ✅ 用户注册和登录
- ✅ PDF 上传和管理
- ✅ 聊天机器人功能
- ✅ CV 检查和提交
- ✅ 通知系统

如果遇到任何问题，请查看 Render 日志或联系支持。

