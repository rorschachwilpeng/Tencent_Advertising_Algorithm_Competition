#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# 多模态缓存配置（唯一配置入口，可被外部覆盖）
export MM_CACHE_ENABLED=${MM_CACHE_ENABLED:-1}
export MM_CACHE_MAX_ENTRIES=${MM_CACHE_MAX_ENTRIES:-12}
echo "[cfg] MM_CACHE_ENABLED=${MM_CACHE_ENABLED}"
echo "[cfg] MM_CACHE_MAX_ENTRIES=${MM_CACHE_MAX_ENTRIES}"

# write your code below
python -u main.py

