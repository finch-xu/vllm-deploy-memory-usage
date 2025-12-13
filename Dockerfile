# 使用官方 Python 运行时作为基础镜像
FROM python:3.14-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和静态文件
COPY main.py .
COPY index.html .
COPY robots.txt .

# 暴露端口
EXPOSE 8018

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8018"]
