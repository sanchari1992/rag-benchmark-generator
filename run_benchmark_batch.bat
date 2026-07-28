@echo off
setlocal

echo ===========================================
echo Running Benchmark Generation for All RAGs
echo ===========================================

for %%R in (neural ragify wizard wiki covid) do (
    for %%S in (ragas ares datamorgana) do (
        echo.
        echo ===========================================
        echo RAG: %%R    Strategy: %%S
        echo ===========================================
        python run_benchmark.py --rag %%R --strategy %%S
        if errorlevel 1 (
            echo ERROR: Failed for %%R with %%S
            pause
            exit /b 1
        )
    )
)

echo.
echo ===========================================
echo All benchmark generation completed!
echo ===========================================

pause