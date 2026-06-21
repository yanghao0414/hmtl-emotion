@echo off
echo ========================================
echo 音频HMTL模型 - 缓存加速训练
echo ========================================
echo.

echo [步骤 1/2] 预处理音频特征 (预计 10-20 分钟)
echo ----------------------------------------
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
echo [步骤 2/2] 开始快速训练 (预计 10-15 分钟)
echo ----------------------------------------
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
echo 模型保存在: 06_模型文件/audio_best_hmtl.pt
echo ========================================
pause
