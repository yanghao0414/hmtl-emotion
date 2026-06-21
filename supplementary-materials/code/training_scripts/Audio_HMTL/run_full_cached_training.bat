@echo off
chcp 65001 >nul
echo ========================================
echo 🚀 音频HMTL模型 - 缓存加速训练（优化版）
echo ========================================
echo.
echo 📊 权重配置：
echo   - L_4 (4核心分类): 1.0
echo   - L_3 (3极性分类): 0.2
echo   - L_A (Arousal):   1.0 (降低33%%)
echo   - L_V (Valence):   0.3 (降低40%%)
echo.
echo ========================================
echo.

REM 检查缓存文件
if exist "05_数据文件\audio_features_cache.pt" (
    echo ✅ 缓存文件已存在，跳过预处理
    goto :train
)

echo [步骤 1/2] 预处理音频特征
echo ----------------------------------------
echo ⏱️  预计时间：10-20 分钟
echo 📦 将生成约 1.8GB 缓存文件
echo.
python modules/audio_hmtl/preprocess_features.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 预处理失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo ✅ 预处理完成！
echo.

:train
echo [步骤 2/2] 开始快速训练（使用优化权重）
echo ----------------------------------------
echo ⏱️  预计时间：10-15 分钟 (10 epochs)
echo 🎯 预期准确率：75-85%%
echo.
python modules/audio_hmtl/train_audio_hmtl_cached.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 训练失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 全部完成！
echo 📁 模型保存在: 06_模型文件\audio_best_hmtl.pt
echo ========================================
pause
