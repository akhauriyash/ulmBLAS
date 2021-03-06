BENCH_CMD            = xdl3blastst
BENCH_OPT            = -N 100 4000 100
BENCH_SUMMARY_SUFFIX = gemm
BENCH_TITLE          = "Compute C=A*B"

EXTRACT_VARIANTS     = MFLOPS TIME TIMELOGLOG

EXTRACT_MFLOPS_X     = 4
EXTRACT_MFLOPS_Y     = 13

EXTRACT_TIME_X       = 4
EXTRACT_TIME_Y       = 12

EXTRACT_TIMELOGLOG_X = 4
EXTRACT_TIMELOGLOG_Y = 12

GPS_VARIANTS         = $(EXTRACT_VARIANTS)

GPS_TITLE            = "GEMM: C=A*B"
GPS_PLOT_OPT         = using 1:2 with linespoints lt 4 lw 3 title "$(#CONFIG_NAME)"

GPS_XLABEL           = "Matrix Dimension (M=K=N)"

GPS_MFLOPS_YLABEL    = "MFLOPS"

GPS_TIME_YLABEL      = "TIME (in seconds)"
GPS_TIME_YRANGE_MIN  =
GPS_TIME_YRANGE_MAX  =

GPS_TIMELOGLOG_YLABEL      = "TIME (in seconds)"
GPS_TIMELOGLOG_YRANGE_MIN  =
GPS_TIMELOGLOG_YRANGE_MAX  =
GPS_TIMELOGLOG_LOGSCALEX   = set logscale x
GPS_TIMELOGLOG_LOGSCALEY   = set logscale y
