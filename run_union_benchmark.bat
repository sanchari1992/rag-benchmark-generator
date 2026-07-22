@echo off
setlocal

echo =====================================
echo Starting Union Benchmark Pipeline
echo =====================================

REM =========================================================
REM Generate ratio_100 results
REM =========================================================

echo Generating ratio_100 results...

for %%R in (neural ragify wizard wiki covid) do (
    python generate_answers.py .\union_benchmarks\ratio_100\%%R_benchmark.json --rag %%R --output .\union_benchmarks\ratio_100\results\%%R_benchmark.json
    python .\evaluators\evaluator.py .\union_benchmarks\ratio_100\%%R_benchmark.json .\union_benchmarks\ratio_100\results\%%R_benchmark.json --output .\union_benchmarks\ratio_100\results\%%R
)


REM =========================================================
REM Compression
REM =========================================================

echo Compressing benchmarks...


for %%P in (90 75 50 25 10) do (

    echo ================================
    echo Compression ratio %%P%%
    echo ================================

    mkdir .\union_benchmarks\ratio_%%P\results 2>nul

    for %%R in (neural ragify wizard wiki covid) do (

        python .\main_compressor.py ^
        --input union_benchmarks\ratio_100\%%R_benchmark.json ^
        --output union_benchmarks\ratio_%%P\%%R_benchmark.json ^
        --report union_benchmarks\ratio_%%P\%%R_report_%%P.json ^
        --ratio 0.%%P ^
        --semantic-weight 0.7

    )
)


REM =========================================================
REM Generate answers and evaluate compressed benchmarks
REM =========================================================


for %%P in (90 75 50 25 10) do (

    echo ====================================
    echo Running ratio_%%P evaluation
    echo ====================================

    mkdir .\union_benchmarks\ratio_%%P\results 2>nul


    for %%R in (neural ragify wizard wiki covid) do (

        python generate_answers.py ^
        .\union_benchmarks\ratio_%%P\%%R_benchmark.json ^
        --rag %%R ^
        --output .\union_benchmarks\ratio_%%P\results\%%R_benchmark.json


        python .\evaluators\evaluator.py ^
        .\union_benchmarks\ratio_%%P\%%R_benchmark.json ^
        .\union_benchmarks\ratio_%%P\results\%%R_benchmark.json ^
        --output .\union_benchmarks\ratio_%%P\results\%%R

    )
)


echo =====================================
echo ALL BENCHMARK EXPERIMENTS COMPLETE
echo =====================================

pause